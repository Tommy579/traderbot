"""Passerelle d'exécution Alpaca, verrouillée sur le paper trading.

Cette couche est la seule à parler au courtier. Elle n'accepte que des
:class:`~trading_bot.risk.engine.OrderIntent` scellés par le moteur de risque,
vérifie la validité du jeton, refuse les rejeux (idempotence par
``decision_id``) et refuse de démarrer si l'endpoint n'est pas celui du paper
trading Alpaca.

Les clients Alpaca sont injectables : les tests unitaires peuvent vérifier la
construction exacte des requêtes sans réseau ni credentials.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Any, Final, Protocol, cast
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetCalendarRequest,
    GetOrdersRequest,
    MarketOrderRequest,
    StopOrderRequest,
)

from trading_bot.config import Settings, is_paper_endpoint
from trading_bot.data.loader import BarFrame, DataError, normalize_ohlcv
from trading_bot.risk.engine import OrderIntent, OrderType, Position, Side

LOGGER = logging.getLogger(__name__)

_TERMINAL_FAILURES: Final[frozenset[str]] = frozenset(
    {"canceled", "expired", "rejected", "replaced", "done_for_day"}
)


class BrokerError(RuntimeError):
    """Erreur de communication ou de contrat avec le courtier."""


class PaperTradingViolation(BrokerError):
    """Tentative d'utiliser autre chose que l'endpoint paper Alpaca."""


class OrderTimeoutError(BrokerError):
    """L'ordre n'a pas atteint un état final dans le délai imparti."""


class OrderRejectedError(BrokerError):
    """Le courtier a rejeté, annulé ou expiré l'ordre."""


class MarketClosure(StrEnum):
    """Cause d'une fermeture de marché constatée auprès d'Alpaca."""

    WEEKEND = "week-end"
    HOLIDAY = "jour férié"
    OUTSIDE_HOURS = "hors des heures de séance"


@dataclass(frozen=True, slots=True)
class MarketStatus:
    """État du marché, cause de fermeture comprise.

    L'horloge Alpaca donne ``is_open`` mais jamais le motif. Le croisement avec
    le calendrier des séances permet de distinguer un week-end, un jour férié
    et un simple hors-séance (avant l'ouverture ou après la clôture).
    """

    is_open: bool
    closure: MarketClosure | None = None
    next_open: datetime | None = None


class TradingApi(Protocol):
    """Sous-ensemble de ``alpaca.trading.client.TradingClient`` utilisé ici."""

    def get_account(self) -> Any: ...

    def get_all_positions(self) -> Sequence[Any]: ...

    def get_clock(self) -> Any: ...

    def get_calendar(self, filters: Any = None) -> Sequence[Any]: ...

    def submit_order(self, order_data: Any) -> Any: ...

    def get_order_by_id(self, order_id: Any) -> Any: ...

    def get_orders(self, filter: Any = None) -> Sequence[Any]: ...

    def get_asset(self, symbol_or_asset_id: Any) -> Any: ...

    def cancel_order_by_id(self, order_id: Any) -> Any: ...

    def cancel_orders(self) -> Any: ...

    def close_all_positions(self, cancel_orders: bool = True) -> Any: ...


class DataApi(Protocol):
    """Sous-ensemble de ``StockHistoricalDataClient`` utilisé ici."""

    def get_stock_latest_trade(self, request: Any) -> Mapping[str, Any]: ...

    def get_stock_bars(self, request: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """Vue normalisée du compte Alpaca."""

    equity: float
    cash: float
    buying_power: float


@dataclass(frozen=True, slots=True)
class AssetInfo:
    """Identité d'un instrument telle que déclarée par Alpaca.

    Sert à montrer à l'utilisateur ce qu'il vient réellement de saisir :
    « BTC » est un ETF Grayscale, pas le Bitcoin.
    """

    symbol: str
    name: str
    asset_class: str
    tradable: bool

    @property
    def is_crypto(self) -> bool:
        """``True`` si l'instrument est une paire crypto (``BTC/USD``)."""
        return self.asset_class.lower() == "crypto"


@dataclass(frozen=True, slots=True)
class OrderReceipt:
    """Vue normalisée d'un ordre Alpaca."""

    order_id: str
    client_order_id: str
    symbol: str
    side: Side
    qty: float
    status: str
    filled_qty: float
    avg_fill_price: float | None
    submitted_at: datetime | None

    @property
    def is_filled(self) -> bool:
        """``True`` si l'ordre est entièrement exécuté."""
        return self.status == "filled" and self.filled_qty > 0.0

    @property
    def is_terminal_failure(self) -> bool:
        """``True`` si l'ordre ne peut plus être exécuté."""
        return self.status in _TERMINAL_FAILURES

    @property
    def is_open(self) -> bool:
        """``True`` si l'ordre peut encore évoluer."""
        return not self.is_filled and not self.is_terminal_failure


def timeframe_to_alpaca(timeframe: str) -> TimeFrame:
    """Convertit un timeframe de configuration en ``TimeFrame`` Alpaca.

    Raises:
        BrokerError: timeframe non supporté.
    """
    mapping: dict[str, TimeFrame] = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "30Min": TimeFrame(30, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day": TimeFrame.Day,
    }
    try:
        return mapping[timeframe]
    except KeyError as exc:
        raise BrokerError(f"timeframe Alpaca non supporté: {timeframe!r}") from exc


_UNKNOWN_ASSET_CODE: Final[int] = 40410000


def _is_unknown_asset(exc: Exception) -> bool:
    """``True`` si Alpaca a répondu « cet actif n'existe pas ».

    Le code d'erreur est propre à l'API ; le message est testé en secours car
    il varie selon la version du client.
    """
    if getattr(exc, "code", None) == _UNKNOWN_ASSET_CODE:
        return True
    text = str(exc)
    return str(_UNKNOWN_ASSET_CODE) in text or "asset not found" in text


class AlpacaPaperClient:
    """Wrapper minimal et sûr de l'API Alpaca Paper.

    Args:
        settings: Configuration (endpoint paper, credentials, feed).
        trading_client: Client de trading injectable (tests).
        data_client: Client de données injectable (tests).

    Raises:
        PaperTradingViolation: si l'URL configurée n'est pas celle du paper.
        ConfigurationError: si les credentials sont absents.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        trading_client: TradingApi | None = None,
        data_client: DataApi | None = None,
    ) -> None:
        if not is_paper_endpoint(settings.alpaca_trading_url) or not settings.is_paper:
            raise PaperTradingViolation(
                "seul l'endpoint paper Alpaca est autorisé "
                f"(reçu: {settings.alpaca_trading_url!r})"
            )
        api_key, secret_key = settings.require_credentials()
        self._settings = settings
        self._feed = DataFeed(settings.data_feed)
        self._trading: TradingApi = cast(
            TradingApi,
            trading_client
            or TradingClient(
                api_key,
                secret_key,
                paper=True,
                url_override=settings.alpaca_trading_url,
            ),
        )
        self._data: DataApi = data_client or StockHistoricalDataClient(
            api_key,
            secret_key,
        )
        self._consumed_decisions: set[str] = set()

    def __repr__(self) -> str:
        """Représentation sans secret."""
        return (
            f"AlpacaPaperClient(endpoint={self._settings.alpaca_trading_url!r}, "
            f"feed={self._settings.data_feed!r})"
        )

    def get_account(self) -> AccountSnapshot:
        """Retourne l'équity, le cash et le buying power du compte paper."""
        try:
            account = self._trading.get_account()
            return AccountSnapshot(
                equity=float(account.equity),
                cash=float(account.cash),
                buying_power=float(account.buying_power),
            )
        except Exception as exc:
            raise BrokerError("lecture du compte Alpaca impossible") from exc

    def get_positions(self) -> dict[str, Position]:
        """Retourne les positions ouvertes, sans métadonnées de stop locales."""
        try:
            raw_positions = self._trading.get_all_positions()
        except Exception as exc:
            raise BrokerError("lecture des positions Alpaca impossible") from exc

        positions: dict[str, Position] = {}
        for raw in raw_positions:
            qty = float(raw.qty)
            if qty <= 0:
                continue
            entry = float(raw.avg_entry_price)
            current = float(raw.current_price) if raw.current_price else entry
            positions[raw.symbol] = Position(
                symbol=raw.symbol,
                qty=qty,
                avg_entry_price=entry,
                current_price=current,
                highest_price=max(entry, current),
            )
        return positions

    def get_latest_prices(self, symbols: Sequence[str]) -> dict[str, float]:
        """Retourne le dernier prix négocié de chaque symbole disponible."""
        unique = sorted({symbol for symbol in symbols if symbol})
        if not unique:
            return {}
        request = StockLatestTradeRequest(symbol_or_symbols=unique, feed=self._feed)
        try:
            trades = self._data.get_stock_latest_trade(request)
        except Exception as exc:
            raise BrokerError("lecture des derniers prix impossible") from exc

        prices: dict[str, float] = {}
        for symbol, trade in trades.items():
            price = float(trade.price)
            if math.isfinite(price) and price > 0:
                prices[symbol] = price
        missing = sorted(set(unique) - set(prices))
        if missing:
            LOGGER.warning("aucun dernier prix pour: %s", ",".join(missing))
        return prices

    def describe_asset(self, symbol: str) -> AssetInfo | None:
        """Identité du symbole selon Alpaca, ou ``None`` s'il est inconnu.

        Args:
            symbol: Symbole à résoudre (``AAPL``, ``BTC/USD``…).

        Returns:
            L'identité de l'instrument, ou ``None`` si Alpaca ne le connaît pas.

        Raises:
            BrokerError: si la question n'a pas pu être posée au courtier.
        """
        if not symbol:
            return None
        try:
            asset = self._trading.get_asset(symbol)
        except Exception as exc:
            if _is_unknown_asset(exc):
                return None
            raise BrokerError(f"identité de {symbol} illisible") from exc
        return AssetInfo(
            symbol=str(asset.symbol),
            name=str(getattr(asset, "name", "") or asset.symbol),
            asset_class=str(
                getattr(getattr(asset, "asset_class", ""), "value", "")
                or getattr(asset, "asset_class", "")
            ),
            tradable=bool(getattr(asset, "tradable", False)),
        )

    def fetch_bars(
        self,
        symbol: str,
        *,
        timeframe: str = "1Day",
        lookback_days: int = 400,
        now: datetime | None = None,
    ) -> BarFrame:
        """Récupère les barres Alpaca et les normalise au schéma du projet."""
        end = now or datetime.now().astimezone()
        start = end - timedelta(days=lookback_days)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe_to_alpaca(timeframe),
            start=start,
            end=end,
            feed=self._feed,
        )
        try:
            bars = self._data.get_stock_bars(request)
        except Exception as exc:
            raise BrokerError(f"lecture des barres {symbol} impossible") from exc

        frame = bars.df
        if frame is None or frame.empty:
            raise DataError(f"{symbol}: aucune barre Alpaca")
        frame = frame.copy()
        if isinstance(frame.index, pd.MultiIndex):
            frame = frame.xs(symbol, level="symbol")
        return normalize_ohlcv(frame, symbol)

    def is_market_open(self) -> bool:
        """Indique si le marché actions est ouvert (horloge Alpaca)."""
        try:
            clock = self._trading.get_clock()
            return bool(clock.is_open)
        except Exception as exc:
            raise BrokerError("lecture de l'horloge de marché impossible") from exc

    def market_status(self) -> MarketStatus:
        """État du marché, motif de fermeture compris (horloge + calendrier).

        L'horloge Alpaca ne dit que si la séance est ouverte, jamais pourquoi.
        On la croise avec le calendrier des séances : une séance existe ce
        jour-là mais l'horloge dit « fermé » = hors-séance ; aucune séance un
        jour de semaine = jour férié ; sinon week-end.

        Returns:
            L'état du marché et, le cas échéant, le motif de fermeture et le
            prochain instant d'ouverture rapportés par Alpaca.

        Raises:
            BrokerError: si l'état n'a pas pu être lu auprès du courtier.
        """
        try:
            clock = self._trading.get_clock()
        except Exception as exc:
            raise BrokerError("lecture de l'horloge de marché impossible") from exc
        if bool(clock.is_open):
            return MarketStatus(is_open=True)

        market_tz = ZoneInfo(self._settings.market_timezone)
        moment = getattr(clock, "timestamp", None)
        if isinstance(moment, datetime):
            moment = moment.astimezone(market_tz)
        else:
            moment = datetime.now(market_tz)
        next_open = getattr(clock, "next_open", None)
        return MarketStatus(
            is_open=False,
            closure=self._closure_reason(moment.date()),
            next_open=next_open if isinstance(next_open, datetime) else None,
        )

    def _closure_reason(self, session_day: date) -> MarketClosure:
        """Motif d'une fermeture, d'après le calendrier des séances Alpaca."""
        try:
            sessions = self._trading.get_calendar(
                GetCalendarRequest(start=session_day, end=session_day)
            )
        except Exception as exc:  # noqa: BLE001 - fermeture best-effort, jamais bloquante
            # Calendrier illisible : le seul motif déductible est le week-end.
            LOGGER.warning("calendrier Alpaca illisible: %s", exc)
            return (
                MarketClosure.WEEKEND
                if session_day.weekday() >= 5
                else MarketClosure.OUTSIDE_HOURS
            )
        if sessions:
            return MarketClosure.OUTSIDE_HOURS
        if session_day.weekday() >= 5:
            return MarketClosure.WEEKEND
        return MarketClosure.HOLIDAY

    def submit(self, intent: OrderIntent, *, now: datetime | None = None) -> OrderReceipt:
        """Transmet un ordre approuvé, une seule fois.

        Les remplacements de stop sont des ordres stop GTC nus : la jambe de
        protection vit dans le carnet d'Alpaca et survit à un crash du process.
        Les ordres stop doivent être scellés par le moteur de risque
        (``OrderType.STOP`` avec ``stop_price``).

        Args:
            intent: Ordre scellé par le moteur de risque.
            now: Instant de validation du jeton (défaut : maintenant UTC).

        Raises:
            BrokerError: jeton invalide, rejoué, ou échec de transmission.
            PaperTradingViolation: si l'ordre n'est pas un ordre paper.
        """
        if not is_paper_endpoint(self._settings.alpaca_trading_url):
            raise PaperTradingViolation("endpoint non paper : ordre refusé")
        instant = now or datetime.now().astimezone()
        if not intent.is_executable(instant):
            raise BrokerError(
                f"jeton d'approbation expiré ou invalide pour {intent.symbol}"
            )
        if intent.token.decision_id in self._consumed_decisions:
            raise BrokerError(
                f"décision {intent.token.decision_id} déjà transmise (idempotence)"
            )

        side = OrderSide.BUY if intent.side is Side.BUY else OrderSide.SELL
        if intent.order_type is OrderType.STOP:
            if intent.stop_price is None:
                raise BrokerError(f"ordre stop sans stop_price pour {intent.symbol}")
            request: Any = StopOrderRequest(
                symbol=intent.symbol,
                qty=intent.qty,
                side=side,
                time_in_force=TimeInForce.GTC,
                stop_price=intent.stop_price,
                client_order_id=intent.client_order_id,
            )
        else:
            request = MarketOrderRequest(
                symbol=intent.symbol,
                qty=intent.qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                client_order_id=intent.client_order_id,
            )
        try:
            order = self._trading.submit_order(order_data=request)
        except Exception as exc:
            raise BrokerError(f"rejet de l'ordre {intent.symbol}") from exc

        self._consumed_decisions.add(intent.token.decision_id)
        receipt = self._to_receipt(order)
        LOGGER.info(
            "ordre transmis %s %s %.4f (client_order_id=%s, statut=%s)",
            receipt.side.value,
            receipt.symbol,
            receipt.qty,
            receipt.client_order_id,
            receipt.status,
        )
        return receipt

    def wait_for_fill(
        self,
        order_id: str,
        *,
        timeout_seconds: float = 30.0,
        poll_seconds: float = 0.25,
    ) -> OrderReceipt:
        """Attend l'exécution complète d'un ordre.

        Raises:
            OrderTimeoutError: si l'ordre reste ouvert au-delà du délai.
            OrderRejectedError: si l'ordre est rejeté, annulé ou expiré.
        """
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds doit être > 0")
        deadline = time.monotonic() + timeout_seconds
        receipt: OrderReceipt | None = None
        while time.monotonic() < deadline:
            try:
                order = self._trading.get_order_by_id(order_id)
            except Exception as exc:
                raise BrokerError(f"suivi de l'ordre {order_id} impossible") from exc
            receipt = self._to_receipt(order)
            if receipt.is_filled:
                return receipt
            if receipt.is_terminal_failure:
                raise OrderRejectedError(
                    f"ordre {order_id} terminé en échec (statut={receipt.status})"
                )
            time.sleep(poll_seconds)
        last_status = receipt.status if receipt else "inconnu"
        raise OrderTimeoutError(
            f"ordre {order_id} non exécuté après {timeout_seconds:.0f}s "
            f"(dernier statut: {last_status})"
        )

    def cancel_open_orders(self, symbol: str) -> int:
        """Annule tous les ordres ouverts d'un symbole (jambes de stop incluses).

        Utilisé avant une liquidation manuelle : une jambe de stop orpheline
        laisserait le compte court sur le symbole si elle se déclenchait après
        la vente. Retourne le nombre d'ordres annulés.
        """
        if not symbol:
            return 0
        request = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[symbol],
            nested=True,
        )
        try:
            orders = self._trading.get_orders(request)
        except Exception as exc:
            raise BrokerError(f"lecture des ordres ouverts {symbol} impossible") from exc
        cancelled = 0
        for order in orders:
            order_id = str(getattr(order, "id", ""))
            if not order_id:
                raise BrokerError(
                    f"ordre ouvert sans identifiant pour {symbol}; annulation non vérifiable"
                )
            try:
                self._trading.cancel_order_by_id(order_id)
            except Exception as exc:
                raise BrokerError(
                    f"annulation de l'ordre ouvert {order_id} ({symbol}) impossible"
                ) from exc
            cancelled += 1

        # Alpaca accepte une requête d'annulation de façon asynchrone. Ne pas
        # supposer qu'elle est effective avant d'avoir relu les ordres ouverts.
        try:
            remaining = self._trading.get_orders(request)
        except Exception as exc:
            raise BrokerError(
                f"vérification des ordres ouverts {symbol} impossible après annulation"
            ) from exc
        remaining_ids = [
            str(getattr(order, "id", "sans-identifiant")) for order in remaining
        ]
        if remaining_ids:
            raise BrokerError(
                f"ordres encore ouverts pour {symbol} après annulation: "
                f"{', '.join(remaining_ids)}"
            )
        if cancelled:
            LOGGER.info(
                "%d ordre(s) ouvert(s) annulé(s) et vérifiés pour %s",
                cancelled,
                symbol,
            )
        return cancelled

    def cancel_all_orders(self) -> None:
        """Annule tous les ordres ouverts du compte paper."""
        try:
            self._trading.cancel_orders()
        except Exception as exc:
            raise BrokerError("annulation des ordres impossible") from exc

    def close_all_positions(self) -> None:
        """Liquide toutes les positions (procédure d'urgence)."""
        try:
            self._trading.close_all_positions(cancel_orders=True)
        except Exception as exc:
            raise BrokerError("liquidation des positions impossible") from exc

    @staticmethod
    def _to_receipt(order: Any) -> OrderReceipt:
        """Normalise un objet ``Order`` Alpaca en :class:`OrderReceipt`."""
        filled_qty = float(order.filled_qty or 0.0)
        average_price = (
            float(order.filled_avg_price) if order.filled_avg_price is not None else None
        )
        side = Side(str(getattr(order.side, "value", order.side)))
        return OrderReceipt(
            order_id=str(order.id),
            client_order_id=str(order.client_order_id),
            symbol=str(order.symbol),
            side=side,
            qty=float(order.qty),
            status=str(getattr(order.status, "value", order.status)),
            filled_qty=filled_qty,
            avg_fill_price=average_price,
            submitted_at=getattr(order, "submitted_at", None),
        )


__all__ = [
    "AccountSnapshot",
    "AlpacaPaperClient",
    "AssetInfo",
    "BrokerError",
    "DataApi",
    "OrderReceipt",
    "OrderRejectedError",
    "OrderTimeoutError",
    "PaperTradingViolation",
    "TradingApi",
    "timeframe_to_alpaca",
]
