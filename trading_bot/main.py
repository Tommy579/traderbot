"""Point d'entrée : boucle de paper trading et CLI de backtest.

Trois commandes :

* ``python -m trading_bot.main live`` — boucle paper trading (ou ``--once``) ;
* ``python -m trading_bot.main backtest`` — backtest déterministe yfinance ;
* ``python -m trading_bot.main check`` — diagnostic lecture seule (compte,
  horloge, données, signal courant), sans aucun ordre.

La boucle live réconcilie l'état du courtier à chaque cycle, restaure les stops
persistés après un redémarrage, et n'envoie un ordre qu'après approbation du
:class:`~trading_bot.risk.engine.RiskEngine`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import Any, Final, Protocol, TypeVar
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import BaseModel, ConfigDict

from trading_bot.backtest.runner import BacktestResult, BacktestRunner
from trading_bot.clock import clock_offset_seconds, get_online_now
from trading_bot.config import (
    PROJECT_ROOT,
    BacktestConfig,
    ConfigurationError,
    RiskConfig,
    Settings,
    configure_logging,
)
from trading_bot.data.loader import (
    BarFrame,
    CachedProvider,
    DataError,
    MarketDataProvider,
    YFinanceProvider,
    drop_incomplete_bar,
    momentum_zscores,
)
from trading_bot.data.news_sentiment import NewsAssessment, NewsSentimentAnalyzer
from trading_bot.execution.alpaca_client import (
    AccountSnapshot,
    AlpacaPaperClient,
    BrokerError,
    MarketStatus,
    OrderReceipt,
)
from trading_bot.history import HistoryStore
from trading_bot.notifications import (
    FanoutNotifier,
    GitHubStatsPublisher,
    Notifier,
)
from trading_bot.notifications.webhook import WebhookNotifier
from trading_bot.risk.engine import (
    AccountState,
    HaltEvent,
    HaltScope,
    OrderIntent,
    Position,
    RiskDecision,
    RiskEngine,
    RiskError,
    Side,
)
from trading_bot.strategies.base import Signal, SignalType, Strategy, StrategyError
from trading_bot.strategies.donchian_breakout import (
    DonchianBreakoutStrategy,
    DonchianParams,
)
from trading_bot.strategies.ema_cross import EmaCrossParams, EmaCrossStrategy
from trading_bot.strategies.relative_momentum import (
    RelativeMomentumParams,
    RelativeMomentumStrategy,
)
from trading_bot.strategies.shield_sword import (
    ShieldSwordParams,
    ShieldSwordStrategy,
)
from trading_bot.withdrawal import (
    WithdrawalResult,
    WithdrawalService,
    cancel_open_orders_verified,
)

LOGGER = logging.getLogger("trading_bot.main")
_ORDER_FILL_TIMEOUT_SECONDS: Final[float] = 30.0
STRATEGY_NAMES: Final[tuple[str, ...]] = (
    "ema_cross",
    "donchian_breakout",
    "relative_momentum",
    "shield_sword",
)
_JOURS_FR: Final[tuple[str, ...]] = (
    "lundi",
    "mardi",
    "mercredi",
    "jeudi",
    "vendredi",
    "samedi",
    "dimanche",
)


def market_closed_message(status: MarketStatus, market_timezone: str) -> str:
    """Explique en clair pourquoi un cycle a été ignoré.

    Le motif et le prochain instant d'ouverture viennent d'Alpaca : le message
    ne suppose donc jamais qu'une fermeture est due aux seuls horaires (un
    week-end ou un jour férié donnent respectivement « week-end » et
    « jour férié »).

    Args:
        status: État de marché renvoyé par le courtier.
        market_timezone: Fuseau de la place de cotation, pour l'heure annoncée.

    Returns:
        Une phrase en français, sans préfixe « cycle ignoré ».
    """
    cause = status.closure.value if status.closure is not None else "séance non ouverte"
    message = f"marché fermé ({cause})"
    if status.next_open is not None:
        local = status.next_open.astimezone(ZoneInfo(market_timezone))
        suffix = local.tzname() or market_timezone
        message += (
            f" — réouverture {_JOURS_FR[local.weekday()]} {local:%d/%m à %H:%M} ({suffix})"
        )
    return message


_T = TypeVar("_T")


def _cli_or(value: _T | None, fallback: _T) -> _T:
    """Valeur CLI si elle est fournie, sinon valeur de configuration.

    Les options du backtest sont déclarées avec ``default=None`` : cela permet
    de distinguer « non fourni » (on prend le .env) de « fourni à la valeur par
    défaut ». Sans cette distinction, un ``.env`` renseigné serait ignoré.
    """
    return fallback if value is None else value


def build_strategy(
    name: str,
    args: argparse.Namespace | None = None,
    *,
    settings: Settings | None = None,
) -> Strategy:
    """Construit la stratégie nommée, avec les surcharges CLI éventuelles.

    Args:
        name: Nom de la stratégie.
        args: Espace de noms argparse (les attributs absents prennent la valeur
            par défaut du code ou de la configuration).
        settings: Configuration effective. Fournie, elle alimente les
            paramètres de la stratégie (Bouclier + Épée) : sans elle, les
            réglages du ``.env`` seraient sans effet.

    Raises:
        ConfigurationError: stratégie inconnue.
        ValueError: paramètres de stratégie invalides.
    """
    namespace = args or argparse.Namespace()
    risk = settings.risk if settings is not None else RiskConfig()
    if name == "ema_cross":
        return EmaCrossStrategy(
            EmaCrossParams(
                fast=getattr(namespace, "fast", 20),
                slow=getattr(namespace, "slow", 50),
            )
        )
    if name == "donchian_breakout":
        return DonchianBreakoutStrategy(
            DonchianParams(
                entry_length=getattr(namespace, "entry", 20),
                exit_length=getattr(namespace, "exit_channel", 10),
                sma_length=getattr(namespace, "sma_long", 200),
            )
        )
    if name == "relative_momentum":
        return RelativeMomentumStrategy(
            RelativeMomentumParams(
                fast_mom=getattr(namespace, "mom_fast", 60),
                slow_mom=getattr(namespace, "mom_slow", 120),
            )
        )
    if name == "shield_sword":
        shield_symbol = risk.shield_symbol
        shield_symbols = tuple(dict.fromkeys([shield_symbol, *risk.safe_haven_symbols]))
        return ShieldSwordStrategy(
            ShieldSwordParams(
                shield_allocation_pct=float(
                    _cli_or(
                        getattr(namespace, "shield_pct", None),
                        settings.shield_allocation_pct if settings is not None else 0.70,
                    )
                ),
                sword_allocation_pct=float(
                    _cli_or(
                        getattr(namespace, "sword_pct", None),
                        settings.sword_allocation_pct if settings is not None else 0.30,
                    )
                ),
                sword_top_n=int(
                    _cli_or(
                        getattr(namespace, "sword_top_n", None),
                        settings.sword_top_n if settings is not None else 3,
                    )
                ),
                sword_rebalance_days=int(
                    _cli_or(
                        getattr(namespace, "sword_rebalance_days", None),
                        settings.sword_rebalance_days if settings is not None else 20,
                    )
                ),
                sword_trend_sma=int(
                    _cli_or(
                        getattr(namespace, "sword_trend_sma", None),
                        settings.sword_trend_sma if settings is not None else 100,
                    )
                ),
                sword_stop_atr_multiplier=(
                    settings.sword_stop_atr_multiplier if settings is not None else 5.0
                ),
                sword_max_loss_pct=(
                    settings.sword_max_loss_pct if settings is not None else 0.12
                ),
                shield_symbol=shield_symbol,
                shield_symbols=shield_symbols,
            )
        )
    raise ConfigurationError(
        f"stratégie inconnue: {name!r} (choix: {', '.join(STRATEGY_NAMES)})"
    )


def _inject_benchmark(
    strategy: Strategy,
    bars: pd.DataFrame,
    symbol: str | None = None,
) -> None:
    """Transmet le benchmark à une stratégie qui en déclare un (no-op sinon).

    Args:
        strategy: Stratégie cible.
        bars: Barres du benchmark (contrat identique aux barres symbole).
        symbol: Nom du benchmark tel que configuré ; il devient l'étiquette
            employée dans les messages. Sans lui, la stratégie conserve sa
            valeur par défaut.
    """
    setter = getattr(strategy, "set_benchmark", None)
    if not callable(setter):
        return
    if symbol is not None:
        strategy.benchmark_symbol = symbol
    setter(bars)


def _build_stats_publisher(settings: Settings) -> GitHubStatsPublisher | None:
    """Publication GitHub configurée, ou ``None`` si elle est absente.

    Une configuration incomplète (dépôt sans jeton) est signalée puis ignorée :
    échouer bruyamment à chaque cycle serait plus pénible qu'utile.

    Args:
        settings: Configuration effective.

    Returns:
        L'objet de publication, ou ``None`` s'il n'est pas exploitable.
    """
    repo = (settings.stats_github_repo or "").strip()
    if not repo:
        return None
    token = (
        settings.stats_github_token.get_secret_value().strip()
        if settings.stats_github_token is not None
        else ""
    )
    if not token:
        LOGGER.warning(
            "STATS_GITHUB_REPO est défini mais STATS_GITHUB_TOKEN est absent : "
            "publication des statistiques ignorée"
        )
        return None
    try:
        return GitHubStatsPublisher(
            repo=repo,
            token=token,
            path=settings.stats_github_path,
            branch=settings.stats_github_branch,
        )
    except ValueError as exc:
        LOGGER.warning("publication des statistiques désactivée: %s", exc)
        return None


def _build_news_analyzer(settings: Settings) -> NewsSentimentAnalyzer | None:
    """Analyseur d'actualités configuré, ou ``None`` si le filtre est inactif.

    Nuance importante par rapport à la publication des statistiques : une clé
    OpenRouter absente ne désactive pas le module. Le filtre existe alors,
    mais reste neutre — le bot continue de trader normalement. On ne retourne
    ``None`` que si le filtre est explicitement coupé, ou si les identifiants
    Alpaca manquent (sans eux, aucun article n'est lisible).

    Args:
        settings: Configuration effective.

    Returns:
        L'analyseur, ou ``None`` si le filtre ne doit pas tourner.
    """
    if not settings.news_filter_enabled:
        return None

    api_key = (
        settings.alpaca_api_key.get_secret_value().strip()
        if settings.alpaca_api_key is not None
        else ""
    )
    secret = (
        settings.alpaca_secret_key.get_secret_value().strip()
        if settings.alpaca_secret_key is not None
        else ""
    )
    if not api_key or not secret:
        LOGGER.warning(
            "filtre d'actualités demandé mais identifiants Alpaca absents : filtre ignoré"
        )
        return None

    openrouter = (
        settings.openrouter_api_key.get_secret_value()
        if settings.openrouter_api_key is not None
        else None
    )
    if not (openrouter or "").strip():
        LOGGER.warning(
            "OPENROUTER_API_KEY absent : le filtre d'actualités reste neutre "
            "(aucun achat ne sera refusé)"
        )

    return NewsSentimentAnalyzer(
        state_dir=settings.state_dir,
        alpaca_api_key=api_key,
        alpaca_secret_key=secret,
        openrouter_api_key=openrouter,
        model=settings.openrouter_model,
        lookback_hours=settings.news_lookback_hours,
        max_articles=settings.news_max_articles,
        timeout_seconds=settings.news_timeout_seconds,
    )


def _assess_news(
    settings: Settings,
    symbols: Sequence[str],
    *,
    errors: list[str] | None = None,
) -> dict[str, NewsAssessment]:
    """Note les actualités d'un ensemble de symboles, sans jamais lever.

    Renvoie un dictionnaire vide si le filtre est inactif ou en panne. Le
    diagnostic doit rester disponible même sans actualités : c'est un outil de
    lecture, il ne doit pas dépendre d'un service tiers pour fonctionner.

    Args:
        settings: Configuration effective.
        symbols: Symboles à noter.
        errors: Liste à enrichir en cas d'incident (diagnostic web).

    Returns:
        Un verdict par symbole, éventuellement vide.
    """
    analyzer = _build_news_analyzer(settings)
    if analyzer is None:
        return {}
    try:
        return analyzer.assess(symbols)
    except Exception as exc:  # noqa: BLE001 - jamais bloquant par construction
        LOGGER.warning("filtre d'actualités ignoré: %s", exc)
        if errors is not None:
            errors.append(f"filtre d'actualités ignoré: {exc}")
        return {}


def _build_notifier(settings: Settings) -> Notifier | None:
    """Construit le diffuseur configuré (webhook et/ou publication GitHub).

    Args:
        settings: Configuration effective.

    Returns:
        Le diffuseur, ou ``None`` si aucun canal n'est configuré — auquel cas
        aucune notification n'est tentée.
    """
    targets: list[Notifier] = []

    url = (
        settings.webhook_url.get_secret_value().strip()
        if settings.webhook_url is not None
        else None
    )
    if url:
        token = (
            settings.webhook_token.get_secret_value().strip()
            if settings.webhook_token is not None
            else None
        )
        targets.append(
            WebhookNotifier(
                url=url,
                format=settings.webhook_format,
                chat_id=settings.webhook_chat_id,
                token=token or None,
                timeout_seconds=settings.webhook_timeout_seconds,
            )
        )

    publisher = _build_stats_publisher(settings)
    if publisher is not None:
        targets.append(publisher)

    if not targets:
        return None
    if len(targets) == 1:
        return targets[0]
    return FanoutNotifier(targets)


class BarSource(Protocol):
    """Source de barres utilisée par la boucle live (courtier ou fake de test)."""

    def fetch_bars(
        self,
        symbol: str,
        *,
        timeframe: str = "1Day",
        lookback_days: int = 400,
        now: datetime | None = None,
    ) -> BarFrame:
        """Retourne les barres normalisées du symbole."""
        ...


def _utc_now() -> datetime:
    """Horloge par défaut : maintenant en UTC, aligné sur l'heure en ligne."""
    return get_online_now()


def _symbol_message(symbol: str, exc: Exception) -> str:
    """Message d'erreur attribué à un symbole, sans préfixe redondant.

    Les exceptions du chargeur de données commencent déjà par « SYMBOLE: » :
    les préfixer à nouveau afficherait « AAPL: AAPL: aucune barre » dans le
    rapport du bot comme dans le diagnostic de la page.

    Args:
        symbol: Symbole concerné.
        exc: Exception levée pendant son analyse.

    Returns:
        Le message, préfixé du symbole seulement si ce n'est pas déjà le cas.
    """
    text = str(exc).strip() or exc.__class__.__name__
    return text if text.startswith(f"{symbol}:") else f"{symbol}: {text}"


def _detect_history_source() -> str:
    """Origine d'exécution consignée dans l'historique des cycles.

    Le service systemd pose ``INVOCATION_ID`` dans l'environnement : c'est le
    seul moyen de distinguer le réveil automatique d'un lancement manuel.

    Returns:
        ``"timer"`` si le cycle vient du service systemd, ``"cli"`` sinon.
    """
    return "timer" if os.environ.get("INVOCATION_ID") else "cli"


class PersistedState(BaseModel):
    """État de risque minimal survivant aux redémarrages.

    Les stops sont la seule information que le courtier ne connaît pas : ils
    doivent être restaurés à l'identique pour éviter qu'une position reste
    « nue » après un redémarrage.
    """

    model_config = ConfigDict(extra="ignore")

    trading_day: str | None = None
    day_start_equity: float | None = None
    peak_equity: float | None = None
    halted: bool = False
    halt_scope: str | None = None
    halt_reason: str | None = None
    stops: dict[str, dict[str, Any]] = {}


class StateStore:
    """Persistance atomique de l'état de risque (JSON)."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """Chemin du fichier d'état."""
        return self._path

    def load(self) -> PersistedState:
        """Charge l'état, ou retourne un état vierge s'il n'existe pas.

        Raises:
            ConfigurationError: si le fichier existe mais est illisible.
        """
        if not self._path.exists():
            return PersistedState()
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            return PersistedState.model_validate(payload)
        except (OSError, ValueError) as exc:
            raise ConfigurationError(
                f"état de risque illisible ({self._path}): {exc}. "
                "Corrigez ou déplacez le fichier avant de redémarrer."
            ) from exc

    def save(self, state: PersistedState) -> None:
        """Écrit l'état de manière atomique (fichier temporaire + ``os.replace``)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = state.model_dump(mode="json")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self._path)
        directory = os.open(self._path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


class AuditLog:
    """Journal d'audit append-only au format JSON Lines."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def record(self, event: str, payload: Mapping[str, Any]) -> None:
        """Ajoute un événement horodaté et force l'écriture sur disque."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": _utc_now().isoformat(),
            "event": event,
            "payload": payload,
        }
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


@dataclass(slots=True)
class CycleReport:
    """Résultat observable d'un cycle de paper trading."""

    timestamp: datetime
    session_date: date
    mode: str = "full"
    skipped: bool = False
    skip_reason: str | None = None
    equity: float = 0.0
    cash: float = 0.0
    positions_count: int = 0
    unrealized_pnl: float = 0.0
    positions: list[dict[str, Any]] = field(default_factory=list)
    signals: dict[str, str] = field(default_factory=dict)
    decisions: list[RiskDecision] = field(default_factory=list)
    orders: list[OrderReceipt] = field(default_factory=list)
    halts: list[HaltEvent] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Représentation JSON-compatible du rapport."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "session_date": self.session_date.isoformat(),
            "mode": self.mode,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "equity": self.equity,
            "cash": self.cash,
            "positions_count": self.positions_count,
            "unrealized_pnl": self.unrealized_pnl,
            "positions": list(self.positions),
            "signals": dict(self.signals),
            "decisions": [
                {
                    "id": decision.decision_id,
                    "approved": decision.approved,
                    "code": decision.code.value,
                    "reason": decision.reason,
                    "symbol": decision.intent.symbol if decision.intent else None,
                    "side": decision.intent.side.value if decision.intent else None,
                    "qty": decision.intent.qty if decision.intent else None,
                    "stop_price": decision.intent.stop_price if decision.intent else None,
                }
                for decision in self.decisions
            ],
            "orders": [
                {
                    "order_id": order.order_id,
                    "client_order_id": order.client_order_id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "qty": order.qty,
                    "status": order.status,
                    "filled_qty": order.filled_qty,
                    "avg_fill_price": order.avg_fill_price,
                }
                for order in self.orders
            ],
            "halts": [
                {
                    "scope": halt.scope.value,
                    "reason": halt.reason,
                    "equity": halt.equity,
                }
                for halt in self.halts
            ],
            "errors": list(self.errors),
        }


class TradingBot:
    """Orchestrateur : données → stratégie → risque → exécution.

    Args:
        settings: Configuration runtime.
        strategy: Stratégie d'analyse (même instance qu'en backtest).
        broker: Passerelle Alpaca Paper.
        bars: Source de barres (par défaut : le courtier).
        state_store: Persistance de l'état de risque.
        audit: Journal d'audit JSONL.
        history: Historique des cycles, affiché par l'interface web.
        notifier: Canal(x) de notification de fin de cycle (webhook, GitHub).
        clock: Horloge injectable (tests).
        sleep: Fonction d'attente injectable (tests).
    """

    def __init__(
        self,
        settings: Settings,
        strategy: Strategy,
        broker: AlpacaPaperClient,
        *,
        bars: BarSource | None = None,
        state_store: StateStore | None = None,
        audit: AuditLog | None = None,
        history: HistoryStore | None = None,
        notifier: Notifier | None = None,
        news_analyzer: NewsSentimentAnalyzer | None = None,
        clock: Callable[[], datetime] = _utc_now,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._strategy = strategy
        self._broker = broker
        self._bars: BarSource = bars or broker
        self._state_store = state_store or StateStore(settings.state_dir / "state.json")
        self._audit = audit or AuditLog(settings.state_dir / "audit.jsonl")
        self._history = history or HistoryStore(settings.state_dir / "history.jsonl")
        self._notifier = notifier
        self._news = news_analyzer
        self._clock = clock
        self._sleep = sleep
        self._risk = RiskEngine(settings.risk)
        self._state = self._state_store.load()
        self._logger = LOGGER

    @property
    def risk(self) -> RiskEngine:
        """Moteur de risque utilisé par le bot."""
        return self._risk

    def run_forever(self) -> None:
        """Boucle de polling jusqu'à SIGINT/SIGTERM."""
        stop_event = threading.Event()

        def _handle_signal(signum: int, _frame: object) -> None:
            """Demande un arrêt propre entre deux cycles."""
            self._logger.info("signal %s reçu : arrêt demandé", signum)
            stop_event.set()

        previous_int = signal.signal(signal.SIGINT, _handle_signal)
        previous_term = signal.signal(signal.SIGTERM, _handle_signal)
        self._logger.info(
            "démarrage paper trading | %s", json.dumps(self._settings.redacted_summary())
        )
        try:
            while not stop_event.is_set():
                try:
                    report = self.run_once()
                    self._log_report(report)
                except (BrokerError, DataError, ConfigurationError, StrategyError) as exc:
                    self._logger.exception("cycle en échec: %s", exc)
                except Exception:
                    self._logger.exception("erreur inattendue pendant le cycle")
                stop_event.wait(self._settings.poll_interval_seconds)
        finally:
            signal.signal(signal.SIGINT, previous_int)
            signal.signal(signal.SIGTERM, previous_term)
            self._logger.info("boucle arrêtée")

    def run_once(self, *, source: str | None = None) -> CycleReport:
        """Exécute un cycle et l'inscrit dans l'historique.

        Args:
            source: Origine consignée dans l'historique (``timer``, ``gui`` ou
                ``cli``) ; déduite de l'environnement si omise.

        Returns:
            Le rapport du cycle.
        """
        report = self._execute_cycle()
        entry = report.to_dict()
        entry["dry_run"] = self._settings.dry_run
        self._history.record(entry, source=source or _detect_history_source())
        return report

    def run_watch_only(self, *, source: str | None = None) -> CycleReport:
        """Exécute une passe de vigie horaire (stops et news) et l'inscrit dans l'historique.

        Args:
            source: Origine consignée (par défaut 'watch_only').

        Returns:
            Le rapport de vigie.
        """
        report = self._execute_watch_only()
        entry = report.to_dict()
        entry["dry_run"] = self._settings.dry_run
        self._history.record(entry, source=source or "watch_only")
        return report

    def _execute_watch_only(self) -> CycleReport:
        """Mode vigie horaire : surveillance des stops 1Day et panic exits news sans nouveaux achats."""
        now = self._clock()
        if now.tzinfo is None:
            raise ConfigurationError("l'horloge du bot doit être timezone-aware")
        offset = clock_offset_seconds()
        if abs(offset) >= 2.0:
            self._logger.info(
                "heure en ligne utilisée : décalage de %+.1f s par rapport au PC",
                offset,
            )
        local_now = now.astimezone(ZoneInfo(self._settings.market_timezone))
        session_date = local_now.date()
        report = CycleReport(timestamp=now, session_date=session_date, mode="watch_only")

        try:
            status = self._broker.market_status()
        except BrokerError as exc:
            report.errors.append(str(exc))
            status = MarketStatus(is_open=False)
        if not status.is_open:
            report.skipped = True
            report.skip_reason = market_closed_message(
                status, self._settings.market_timezone
            )
            return report

        snapshot = self._broker.get_account()
        positions = self._broker.get_positions()
        account = self._build_account(
            snapshot.equity, snapshot.cash, snapshot.buying_power, positions
        )
        self._risk.start_session(account, session_date=session_date)
        halt = self._risk.update_risk_state(account, session_date=session_date)
        if halt is not None:
            report.halts.append(halt)
            self._audit_halt(halt)
            self._logger.critical("CIRCUIT BREAKER (%s): %s", halt.scope.value, halt.reason)
            self._flatten(account, {}, now, report)
            self._persist(account)
            report.equity = account.equity
            report.cash = account.cash
            report.positions_count = len(account.positions)
            report.positions = [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "avg_entry_price": p.avg_entry_price,
                    "current_price": p.current_price,
                    "stop_price": p.stop_price,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                }
                for p in account.positions.values()
            ]
            report.unrealized_pnl = sum(
                p.unrealized_pnl for p in account.positions.values()
            )
            self._notify(report)
            return report

        if not account.positions:
            self._logger.info(
                "vigie horaire : aucune position en portefeuille à surveiller"
            )
            self._persist(account)
            report.equity = account.equity
            report.cash = account.cash
            report.positions_count = 0
            return report

        open_symbols = sorted(account.positions.keys())
        try:
            live_prices = self._broker.get_latest_prices(open_symbols)
        except BrokerError as exc:
            self._logger.warning("vigie: lecture des prix en direct impossible: %s", exc)
            report.errors.append(f"prix en direct: {exc}")
            live_prices = {}

        marks: dict[str, float] = {}
        for sym, pos in account.positions.items():
            marks[sym] = live_prices.get(sym, pos.current_price)

        self._recompute_equity(account, marks)

        assessments = self._news_assessments(open_symbols, now=now, report=report)

        for symbol in open_symbols:
            position = account.positions.get(symbol)
            if position is None:
                continue
            price = marks.get(symbol, position.current_price)

            # 1. Sortie d'urgence sur alerte news critique
            decision = self._risk.evaluate_news_panic_exit(
                position,
                news=assessments.get(symbol),
                price=price,
                now=now,
            )
            if decision is not None:
                self._logger.warning(
                    "vigie horaire: sortie d'urgence %s: %s", symbol, decision.reason
                )

            # 2. Sortie sur stop-loss journalier
            if decision is None:
                decision = self._risk.evaluate_stop_exit(position, price=price, now=now)

            # 3. Take-profit partiel (+20% PnL)
            if decision is None:
                decision = self._risk.evaluate_partial_take_profit(
                    position, price=price, now=now
                )
                if decision is not None:
                    self._logger.info(
                        "vigie horaire: take-profit partiel %s: %s", symbol, decision.reason
                    )

            # 4. Écrémage surpondération (>25% équité)
            if decision is None:
                decision = self._risk.evaluate_overweight_trim(
                    position, price=price, account=account, now=now
                )
                if decision is not None:
                    self._logger.info(
                        "vigie horaire: écrémage surpondération %s: %s",
                        symbol,
                        decision.reason,
                    )

            if decision is None:
                continue

            report.decisions.append(decision)
            self._audit_decision(decision)
            if self._execute_decision(decision, now, account, report):
                self._logger.info(
                    "vigie horaire: position %s coupée ou allégée (%s)",
                    symbol,
                    decision.reason,
                )

        # 5. Reconstitution du plancher de cash via Safe Haven si nécessaire
        reconstitution_decisions = self._risk.evaluate_safe_haven_cash_floor_reconstitution(
            account, marks, now=now
        )
        for r_dec in reconstitution_decisions:
            report.decisions.append(r_dec)
            self._audit_decision(r_dec)
            if self._execute_decision(r_dec, now, account, report):
                self._logger.info(
                    "vigie horaire: reconstitution plancher cash via safe haven %s: %s",
                    r_dec.intent.symbol if r_dec.intent else "",
                    r_dec.reason,
                )

        self._recompute_equity(account, marks)
        self._persist(account)
        report.equity = account.equity
        report.cash = account.cash
        report.positions_count = len(account.positions)
        report.positions = [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_entry_price": p.avg_entry_price,
                "current_price": p.current_price,
                "stop_price": p.stop_price,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pnl,
            }
            for p in account.positions.values()
        ]
        report.unrealized_pnl = sum(p.unrealized_pnl for p in account.positions.values())
        if report.orders:
            self._notify(report)
        return report

    def _execute_cycle(self) -> CycleReport:
        """Exécute un cycle complet de décision et d'exécution."""
        now = self._clock()
        if now.tzinfo is None:
            raise ConfigurationError("l'horloge du bot doit être timezone-aware")
        offset = clock_offset_seconds()
        if abs(offset) >= 2.0:
            self._logger.info(
                "heure en ligne utilisée : décalage de %+.1f s par rapport au PC",
                offset,
            )
        local_now = now.astimezone(ZoneInfo(self._settings.market_timezone))
        session_date = local_now.date()
        report = CycleReport(timestamp=now, session_date=session_date)

        try:
            status = self._broker.market_status()
        except BrokerError as exc:
            report.errors.append(str(exc))
            status = MarketStatus(is_open=False)
        if not status.is_open:
            report.skipped = True
            report.skip_reason = market_closed_message(
                status, self._settings.market_timezone
            )
            return report

        snapshot = self._broker.get_account()
        positions = self._broker.get_positions()
        account = self._build_account(
            snapshot.equity, snapshot.cash, snapshot.buying_power, positions
        )
        self._risk.start_session(account, session_date=session_date)
        halt = self._risk.update_risk_state(account, session_date=session_date)
        if halt is not None:
            report.halts.append(halt)
            self._audit_halt(halt)
            self._logger.critical("CIRCUIT BREAKER (%s): %s", halt.scope.value, halt.reason)

        benchmark_symbol = self._settings.benchmark_symbol
        bull_market = False
        bear_market = False
        try:
            benchmark_bars = self._fetch_complete_bars(benchmark_symbol, now)
            _inject_benchmark(self._strategy, benchmark_bars, benchmark_symbol)
            close = benchmark_bars["close"].astype(float)
            sma_period = self._risk.config.regime_sma_period
            if len(close) >= sma_period:
                sma = float(close.rolling(sma_period).mean().iloc[-1])
                last_close = float(close.iloc[-1])
                bull_market = last_close > sma
                bear_market = last_close <= sma
            else:
                bull_market = False
                bear_market = False
        except (DataError, BrokerError, StrategyError) as exc:
            report.errors.append(f"{benchmark_symbol}: {exc}")
            self._logger.warning(
                "benchmark %s indisponible, régime prudent: %s", benchmark_symbol, exc
            )
        self._risk.set_market_regime(bull_market=bull_market, bear_market=bear_market)
        if bull_market:
            self._logger.info(
                "régime haussier: %s > SMA%d, plafonds desserrés",
                benchmark_symbol,
                self._risk.config.regime_sma_period,
            )
        elif bear_market:
            self._logger.warning(
                "régime baissier: %s <= SMA%d ([BEAR_REGIME_VETO] actif sur offensifs)",
                benchmark_symbol,
                self._risk.config.regime_sma_period,
            )

        safe_havens = (
            set(self._settings.safe_haven_symbols)
            if self._settings.safe_haven_enabled
            else set()
        )
        # Le Bouclier est un actif comme les autres pour la couche données : il
        # faut ses barres pour le coter, le classer hors du panier offensif et
        # l'acheter. Sans cette ligne, SHIELD_SYMBOL resterait une intention.
        sword = self._shield_sword_params()
        shield_sleeve = {sword.shield_symbol} if sword is not None else set()
        universe = sorted(
            set(self._settings.symbols)
            | safe_havens
            | shield_sleeve
            | set(account.positions)
        )
        bars_by_symbol: dict[str, BarFrame] = {}
        for symbol in universe:
            try:
                bars_by_symbol[symbol] = self._fetch_complete_bars(symbol, now)
            except (DataError, BrokerError) as exc:
                report.errors.append(_symbol_message(symbol, exc))
                self._logger.warning("téléchargement %s ignoré: %s", symbol, exc)
                continue

        if hasattr(self._strategy, "set_universe"):
            self._strategy.set_universe(bars_by_symbol)

        signals: dict[str, Signal] = {}
        for symbol, bars in bars_by_symbol.items():
            try:
                signal = self._strategy.latest_signal(bars, symbol=symbol)
            except StrategyError as exc:
                report.errors.append(_symbol_message(symbol, exc))
                self._logger.warning("analyse %s ignorée: %s", symbol, exc)
                continue
            signals[symbol] = signal
            report.signals[symbol] = signal.action.value

        marks = self._resolve_marks(universe, bars_by_symbol, account, report)
        self._adopt_unprotected_positions(account, signals, marks, report, now=now)
        assessments = self._news_assessments(universe, now=now, report=report)

        if halt is not None:
            self._flatten(account, marks, now, report)
            self._persist(account)
            report.equity = account.equity
            report.cash = account.cash
            report.positions_count = len(account.positions)
            report.positions = [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "avg_entry_price": p.avg_entry_price,
                    "current_price": p.current_price,
                    "stop_price": p.stop_price,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                }
                for p in account.positions.values()
            ]
            report.unrealized_pnl = sum(
                p.unrealized_pnl for p in account.positions.values()
            )
            self._notify(report)
            return report

        exited_symbols: set[str] = set()
        for symbol in sorted(list(account.positions)):
            position = account.positions.get(symbol)
            if position is None:
                continue
            price = marks.get(symbol, position.current_price)
            # La vigie passe avant le stop : sur un krach, le stop glisse et se
            # remplit bien plus bas que son seuil. Une menace critique justifie
            # de sortir au marché sans l'attendre.
            decision = self._risk.evaluate_news_panic_exit(
                position,
                news=assessments.get(symbol),
                price=price,
                now=now,
            )
            if decision is not None:
                self._logger.warning("sortie d'urgence %s: %s", symbol, decision.reason)
            if decision is None:
                decision = self._risk.evaluate_stop_exit(position, price=price, now=now)
            if decision is None and symbol not in safe_havens:
                exit_signal = signals.get(symbol)
                if exit_signal is not None and exit_signal.action is SignalType.SELL:
                    decision = self._risk.evaluate_exit(
                        position=position,
                        price=price,
                        now=now,
                        reason="strategy_exit",
                    )
            # Take-profit partiel (+20% PnL)
            if decision is None:
                decision = self._risk.evaluate_partial_take_profit(
                    position, price=price, now=now
                )
                if decision is not None:
                    self._logger.info("take-profit partiel %s: %s", symbol, decision.reason)
            # Écrémage surpondération (>25% équité)
            if decision is None:
                decision = self._risk.evaluate_overweight_trim(
                    position, price=price, account=account, now=now
                )
                if decision is not None:
                    self._logger.info(
                        "écrémage surpondération %s: %s", symbol, decision.reason
                    )

            if decision is None:
                continue
            report.decisions.append(decision)
            self._audit_decision(decision)
            if self._execute_decision(decision, now, account, report):
                if symbol not in account.positions:
                    exited_symbols.add(symbol)
                    self._logger.info("position %s liquidée (%s)", symbol, decision.reason)
                else:
                    self._logger.info("position %s allégée (%s)", symbol, decision.reason)

        # Reconstitution du plancher de cash via Safe Haven si nécessaire
        reconstitution_decisions = self._risk.evaluate_safe_haven_cash_floor_reconstitution(
            account, marks, now=now
        )
        for r_dec in reconstitution_decisions:
            report.decisions.append(r_dec)
            self._audit_decision(r_dec)
            if self._execute_decision(r_dec, now, account, report):
                self._logger.info(
                    "reconstitution plancher cash via safe haven %s: %s",
                    r_dec.intent.symbol if r_dec.intent else "",
                    r_dec.reason,
                )

        for symbol, position in sorted(account.positions.items()):
            trailing_signal = signals.get(symbol)
            if trailing_signal is None:
                continue
            atr = trailing_signal.indicators.get("atr")
            price = marks.get(symbol, position.current_price)
            if atr is None or not math.isfinite(atr) or atr <= 0:
                report.errors.append(f"{symbol}: ATR indisponible pour le trailing")
                continue
            update = self._risk.update_trailing_stop(position, price=price, atr=atr)
            if update is not None and self._risk.apply_trailing_stop(account, update):
                self._logger.info(
                    "trailing %s: stop %.2f -> %.2f (R=%.2f)",
                    symbol,
                    update.previous_stop,
                    update.new_stop,
                    update.r_multiple,
                )
                self._sync_broker_stop(position, price=price, now=now, report=report)

        candidates: dict[str, float] = {}
        for symbol in self._settings.symbols:
            if symbol in account.positions or symbol in exited_symbols:
                continue
            entry_signal = signals.get(symbol)
            if entry_signal is None or entry_signal.action is not SignalType.BUY:
                continue
            atr = entry_signal.indicators.get("atr")
            if atr is None or not math.isfinite(atr) or atr <= 0:
                report.errors.append(f"{symbol}: ATR indisponible, entrée refusée")
                continue
            candidates[symbol] = self._entry_score(bars_by_symbol.get(symbol))

        # Le budget de liquidités va d'abord aux actifs les plus vigoureux.
        for symbol in self._risk.rank_candidates(candidates):
            entry_signal = signals[symbol]
            atr = entry_signal.indicators["atr"]
            price = marks.get(symbol, entry_signal.price)

            # Auto-Unwind de repli si cash insuffisant pour l'ordre offensif validé
            if self._settings.safe_haven_enabled and not self._risk.bear_market:
                news_item = assessments.get(symbol)
                if not (news_item and self._risk.news_vetoes_entry(news_item)):
                    stop_distance = atr * self._risk.config.atr_stop_multiple
                    risk_budget = account.equity * self._risk.config.risk_per_trade_pct
                    qty_by_risk = risk_budget / stop_distance
                    gross_rem = max(
                        0.0,
                        account.equity * self._risk.effective_gross_cap_pct()
                        - account.gross_exposure,
                    )
                    notional_cap = min(
                        account.equity * self._risk.effective_position_cap_pct(),
                        gross_rem,
                    )
                    target_notional = min(qty_by_risk * price, notional_cap)
                    cash_buffer = account.equity * self._risk.config.cash_buffer_pct
                    usable_cash = max(
                        0.0, min(account.cash - cash_buffer, account.buying_power)
                    )
                    if target_notional > usable_cash:
                        needed = target_notional - usable_cash
                        unwind_decisions = self._risk.evaluate_safe_haven_unwind(
                            account, needed, marks, now=now
                        )
                        for u_dec in unwind_decisions:
                            report.decisions.append(u_dec)
                            self._audit_decision(u_dec)
                            if self._execute_decision(u_dec, now, account, report):
                                self._logger.info(
                                    "auto-unwind safe haven %s: %s",
                                    u_dec.intent.symbol if u_dec.intent else "",
                                    u_dec.reason,
                                )

            is_shield_asset = sword is not None and symbol in sword.shield_symbols
            target_weight: float | None = None
            atr_multiple: float | None = None
            max_loss: float | None = None
            if sword is not None:
                if is_shield_asset:
                    target_weight = sword.shield_allocation_pct
                else:
                    target_weight = sword.sword_allocation_pct / max(1, sword.sword_top_n)
                    atr_multiple = sword.sword_stop_atr_multiplier
                    max_loss = sword.sword_max_loss_pct
            decision = self._risk.evaluate_entry(
                symbol=symbol,
                price=price,
                atr=atr,
                account=account,
                now=now,
                session_date=session_date,
                news=assessments.get(symbol),
                target_weight=target_weight,
                atr_stop_multiple=atr_multiple,
                max_loss_pct=max_loss,
                is_shield=is_shield_asset,
            )
            report.decisions.append(decision)
            self._audit_decision(decision)
            if not decision.approved:
                self._logger.info(
                    "entrée %s refusée (%s): %s",
                    symbol,
                    decision.code.value,
                    decision.reason,
                )
                continue
            self._execute_decision(decision, now, account, report)

        # Le Bouclier se finance après les entrées offensives : il récupère le
        # cash que la sélection n'a pas engagé, jusqu'à son poids cible.
        if sword is not None and sword.shield_allocation_pct > 0.0:
            shield_price = marks.get(sword.shield_symbol)
            if shield_price is not None:
                shield_decision = self._risk.evaluate_shield_allocation(
                    account,
                    symbol=sword.shield_symbol,
                    price=shield_price,
                    target_weight=sword.shield_allocation_pct,
                    now=now,
                )
                if shield_decision is not None:
                    report.decisions.append(shield_decision)
                    self._audit_decision(shield_decision)
                    if self._execute_decision(shield_decision, now, account, report):
                        self._logger.info(
                            "bouclier %s financé à %.0f%%: %s",
                            sword.shield_symbol,
                            sword.shield_allocation_pct * 100,
                            shield_decision.reason,
                        )

        # Cash Sweep en fin de cycle : réinvestissement du cash excédentaire vers les safe havens
        if self._settings.safe_haven_enabled and not account.halted:
            atrs = {
                sym: sig.indicators.get("atr", 0.0)
                for sym, sig in signals.items()
                if sig is not None and "atr" in sig.indicators
            }
            sweep_decisions = self._risk.evaluate_cash_sweep(
                account,
                marks,
                now=now,
                news=assessments,
                atrs=atrs,
            )
            for s_dec in sweep_decisions:
                report.decisions.append(s_dec)
                self._audit_decision(s_dec)
                if self._execute_decision(s_dec, now, account, report):
                    self._logger.info(
                        "cash sweep %s: %s",
                        s_dec.intent.symbol if s_dec.intent else "",
                        s_dec.reason,
                    )

        self._recompute_equity(account, marks)
        self._persist(account)
        report.equity = account.equity
        report.cash = account.cash
        report.positions_count = len(account.positions)
        report.positions = [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_entry_price": p.avg_entry_price,
                "current_price": p.current_price,
                "stop_price": p.stop_price,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pnl,
            }
            for p in account.positions.values()
        ]
        report.unrealized_pnl = sum(p.unrealized_pnl for p in account.positions.values())
        self._notify(report)
        return report

    def _news_assessments(
        self, universe: Sequence[str], *, now: datetime, report: CycleReport
    ) -> dict[str, NewsAssessment]:
        """Note les actualités de l'univers, sans jamais faire échouer le cycle.

        L'analyseur est déjà *fail-safe* ; cette enveloppe couvre le reste
        (disque plein, état corrompu, imprévu). Un filtre d'actualités
        défaillant ne doit jamais empêcher la gestion des positions déjà
        ouvertes — c'est-à-dire le seul geste réellement urgent.

        Args:
            universe: Symboles à noter.
            now: Horodatage de référence du cycle.
            report: Rapport du cycle, enrichi en cas d'incident.

        Returns:
            Un verdict par symbole, éventuellement vide.
        """
        if self._news is None:
            return {}
        try:
            return self._news.assess(universe, now=now)
        except Exception as exc:  # noqa: BLE001 - jamais bloquant par construction
            report.errors.append(f"filtre d'actualités ignoré: {exc}")
            self._logger.warning("filtre d'actualités ignoré: %s", exc)
            return {}

    def _build_account(
        self,
        equity: float,
        cash: float,
        buying_power: float,
        positions: Mapping[str, Position],
    ) -> AccountState:
        """Reconstruit l'état de compte en restaurant les stops persistés."""
        restored: dict[str, Position] = {}
        for symbol, position in positions.items():
            stored = self._state.stops.get(symbol)
            if stored:
                position.apply_stop_state(stored)
            restored[symbol] = position
        trading_day: date | None = None
        if self._state.trading_day:
            try:
                trading_day = date.fromisoformat(self._state.trading_day)
            except ValueError as exc:
                raise ConfigurationError(
                    f"trading_day invalide dans l'état: {self._state.trading_day!r}"
                ) from exc
        halt_scope: HaltScope | None = None
        if self._state.halt_scope:
            try:
                halt_scope = HaltScope(self._state.halt_scope)
            except ValueError as exc:
                raise ConfigurationError(
                    f"halt_scope invalide dans l'état: {self._state.halt_scope!r}"
                ) from exc
        return AccountState(
            equity=equity,
            cash=cash,
            buying_power=buying_power,
            positions=restored,
            day_start_equity=self._state.day_start_equity,
            peak_equity=self._state.peak_equity,
            trading_day=trading_day,
            halted=self._state.halted,
            halt_scope=halt_scope,
            halt_reason=self._state.halt_reason,
        )

    def _fetch_complete_bars(self, symbol: str, now: datetime) -> BarFrame:
        """Récupère les barres et retire la barre en cours de formation."""
        bars = self._bars.fetch_bars(
            symbol,
            timeframe=self._settings.timeframe,
            lookback_days=self._settings.lookback_days,
            now=now,
        )
        complete = drop_incomplete_bar(
            bars,
            self._settings.timeframe,
            now,
            market_timezone=self._settings.market_timezone,
        )
        if len(complete) <= self._strategy.warmup_bars:
            raise DataError(
                f"{symbol}: {len(complete)} barres complètes, "
                f"{self._strategy.warmup_bars + 1} requises"
            )
        return complete

    def _resolve_marks(
        self,
        universe: Sequence[str],
        bars_by_symbol: Mapping[str, BarFrame],
        account: AccountState,
        report: CycleReport,
    ) -> dict[str, float]:
        """Derniers prix : flux temps réel, repli sur la dernière clôture."""
        marks: dict[str, float] = {}
        try:
            marks.update(self._broker.get_latest_prices(universe))
        except BrokerError as exc:
            report.errors.append(str(exc))
            self._logger.warning("prix temps réel indisponibles: %s", exc)
        for symbol in universe:
            if symbol in marks:
                continue
            bars = bars_by_symbol.get(symbol)
            if bars is not None and not bars.empty:
                marks[symbol] = float(bars["close"].iloc[-1])
                continue
            position = account.positions.get(symbol)
            if position is not None:
                marks[symbol] = position.current_price
        return marks

    def _notify(self, report: CycleReport) -> None:
        """Envoie le récapitulatif du cycle (équité, signaux, ordres, halts).

        Les cycles ignorés (marché fermé) ne notifient pas : en mode polling,
        ils se répètent toutes les ``poll_interval_seconds`` et noieraient le
        canal. Un échec d'envoi est journalisé, jamais propagé.
        """
        if self._notifier is None or report.skipped:
            return
        self._notifier.send("cycle_trading", report.to_dict())

    def _shield_sword_params(self) -> ShieldSwordParams | None:
        """Paramètres Bouclier + Épée de la stratégie courante, ou ``None``.

        Source unique des poids, du multiple d'ATR et de la perte maximale : le
        live ne doit pas recoder en dur ce que la stratégie (et donc le
        backtest) tient de la configuration.
        """
        if self._strategy.name != "shield_sword":
            return None
        params = getattr(self._strategy, "params", None)
        return params if isinstance(params, ShieldSwordParams) else None

    @staticmethod
    def _entry_score(bars: BarFrame | None) -> float:
        """Z-score de momentum (60/120 j) de la dernière barre achevée.

        Même mesure que le classement du backtest : le ranking en live ne peut
        pas différer de celui observé en simulation. Renvoie ``nan`` si les
        données sont insuffisantes (le classement relègue alors le symbole).
        """
        if bars is None or bars.empty:
            return float("nan")
        z_fast, z_slow = momentum_zscores(bars["close"])
        score = float(z_fast.iloc[-1]) + float(z_slow.iloc[-1])
        return score if math.isfinite(score) else float("nan")

    def _adopt_unprotected_positions(
        self,
        account: AccountState,
        signals: Mapping[str, Signal],
        marks: Mapping[str, float],
        report: CycleReport,
        *,
        now: datetime,
    ) -> None:
        """Attache un stop ATR aux positions ouvertes hors du bot.

        Le stop est posé localement **et** matérialisé chez le courtier : une
        position adoptée ne doit pas rester nue si le process s'arrête.
        """
        for symbol, position in account.positions.items():
            if position.stop_price is not None:
                continue
            if self._risk.is_defensive_symbol(symbol):
                # Le Bouclier est l'actif sans risque du montage : lui coller un
                # stop ATR le transformerait en action à protéger, et le ferait
                # liquider sur une simple respiration du marché obligataire.
                continue
            signal = signals.get(symbol)
            atr = signal.indicators.get("atr") if signal else None
            price = marks.get(symbol, position.current_price)
            if atr is None or not math.isfinite(atr) or atr <= 0:
                report.errors.append(
                    f"{symbol}: position sans stop et sans ATR — protégez-la manuellement"
                )
                self._logger.error(
                    "position %s non protégée : aucun ATR disponible", symbol
                )
                continue
            stop = round(price - atr * self._settings.risk.atr_stop_multiple, 2)
            if stop <= 0:
                report.errors.append(f"{symbol}: stop ATR non positif")
                continue
            position.stop_price = stop
            position.initial_stop_price = stop
            if position.highest_price <= 0:
                position.highest_price = price
            self._logger.warning(
                "position %s adoptée (hors état persisté) avec stop ATR %.2f",
                symbol,
                stop,
            )
            self._sync_broker_stop(position, price=price, now=now, report=report)

    def _execute_decision(
        self,
        decision: RiskDecision,
        now: datetime,
        account: AccountState,
        report: CycleReport,
    ) -> bool:
        """Transmet un ordre approuvé et met à jour l'état local après fill.

        Returns:
            ``True`` uniquement si un fill a été appliqué à l'état local.
            ``False`` en cas de refus, de dry-run ou d'échec de transmission.
        """
        if not decision.approved or decision.intent is None:
            return False
        intent = decision.intent
        if self._settings.dry_run:
            self._logger.info(
                "dry-run: %s %s %.4f @ ~%.2f (stop %.2f, %s)",
                intent.side.value,
                intent.symbol,
                intent.qty,
                decision.metrics.get("price", 0.0),
                intent.stop_price or 0.0,
                decision.reason,
            )
            return False
        if intent.side is Side.SELL and not self._cancel_open_orders(intent.symbol, report):
            # Une jambe de stop ouverte chez le courtier pourrait vendre à
            # découvert : attendre une annulation confirmée avant la liquidation.
            return False
        try:
            receipt = self._broker.submit(intent, now=now)
            report.orders.append(receipt)
            filled = self._broker.wait_for_fill(
                receipt.order_id, timeout_seconds=_ORDER_FILL_TIMEOUT_SECONDS
            )
        except BrokerError as exc:
            report.errors.append(f"{intent.symbol}: {exc}")
            self._logger.error("échec d'exécution %s: %s", intent.symbol, exc)
            return False
        self._apply_fill(account, intent, filled)
        if intent.side is Side.SELL and not self._cancel_open_orders(intent.symbol, report):
            self._logger.critical(
                "vente exécutée pour %s mais des ordres résiduels n'ont pas pu être réconciliés",
                intent.symbol,
            )
        self._audit.record(
            "order",
            {
                "order_id": filled.order_id,
                "client_order_id": filled.client_order_id,
                "symbol": filled.symbol,
                "side": filled.side.value,
                "qty": filled.qty,
                "filled_qty": filled.filled_qty,
                "avg_fill_price": filled.avg_fill_price,
                "status": filled.status,
            },
        )
        if intent.side is Side.BUY and intent.stop_price is not None:
            # Le stop ne doit pas exister seulement dans state.json : on
            # matérialise la jambe de protection chez le courtier.
            opened = account.positions.get(intent.symbol)
            if opened is not None:
                fill_price = (
                    filled.avg_fill_price
                    if filled.avg_fill_price is not None
                    else opened.current_price
                )
                self._sync_broker_stop(opened, price=fill_price, now=now, report=report)
        elif intent.side is Side.SELL:
            # Si la position existe encore (vente partielle), repositionner le stop
            remaining = account.positions.get(intent.symbol)
            if remaining is not None and remaining.stop_price is not None:
                fill_price = (
                    filled.avg_fill_price
                    if filled.avg_fill_price is not None
                    else remaining.current_price
                )
                self._sync_broker_stop(remaining, price=fill_price, now=now, report=report)
        return True

    def _cancel_open_orders(self, symbol: str, report: CycleReport) -> bool:
        """Annule les ordres ouverts du symbole, et échoue en mode fermé.

        Délègue au garde-fou partagé avec les retraits manuels : une seule
        implémentation, donc une seule règle de sécurité.
        """
        return cancel_open_orders_verified(
            self._broker, symbol, audit=self._audit, errors=report.errors
        )

    def _sync_broker_stop(
        self,
        position: Position,
        *,
        price: float,
        now: datetime,
        report: CycleReport,
    ) -> None:
        """Matérialise le stop de la position chez le courtier (ordre stop scellé).

        L'ordre stop remplace toute jambe précédente (annulée au préalable) :
        la protection survit à un crash du process. En dry-run, l'intention
        est uniquement journalisée.
        """
        decision = self._risk.evaluate_stop_replace(position, price=price, now=now)
        if decision is None or not decision.approved or decision.intent is None:
            return
        intent = decision.intent
        if self._settings.dry_run:
            self._logger.info(
                "dry-run: stop %s %.4f @ %.2f (%s)",
                intent.symbol,
                intent.qty,
                intent.stop_price or 0.0,
                intent.reason,
            )
            return
        if not self._cancel_open_orders(intent.symbol, report):
            return
        try:
            receipt = self._broker.submit(intent, now=now)
        except BrokerError as exc:
            report.errors.append(f"{intent.symbol}: {exc}")
            self._logger.error("échec du placement du stop %s: %s", intent.symbol, exc)
            return
        report.orders.append(receipt)
        self._audit_decision(decision)
        self._audit.record(
            "stop_replace",
            {
                "symbol": intent.symbol,
                "stop_price": intent.stop_price,
                "qty": intent.qty,
                "client_order_id": receipt.client_order_id,
            },
        )
        self._logger.info(
            "stop courtier %s %.4f @ %.2f (client_order_id=%s)",
            intent.symbol,
            intent.qty,
            intent.stop_price or 0.0,
            receipt.client_order_id,
        )

    def _apply_fill(
        self,
        account: AccountState,
        intent: OrderIntent,
        filled: OrderReceipt,
    ) -> None:
        """Répercute un fill sur l'état local (cash et positions)."""
        qty = filled.filled_qty if filled.filled_qty > 0 else filled.qty
        price = filled.avg_fill_price
        if price is None or not math.isfinite(price) or price <= 0:
            price = intent.stop_price or 0.0
            if price <= 0:
                raise BrokerError(
                    f"fill {filled.order_id} sans prix exploitable (statut={filled.status})"
                )
            self._logger.warning(
                "fill %s sans avg_fill_price, repli sur %.2f", filled.order_id, price
            )
        if intent.side is Side.BUY:
            account.cash -= qty * price
            existing = account.positions.get(intent.symbol)
            if existing is not None:
                new_qty = existing.qty + qty
                new_avg = (
                    (existing.avg_entry_price * existing.qty) + (price * qty)
                ) / new_qty
                existing.qty = new_qty
                existing.avg_entry_price = new_avg
                existing.current_price = price
                existing.highest_price = max(existing.highest_price, price)
                if intent.stop_price is not None:
                    existing.stop_price = intent.stop_price
            else:
                account.positions[intent.symbol] = Position(
                    symbol=intent.symbol,
                    qty=qty,
                    avg_entry_price=price,
                    current_price=price,
                    stop_price=intent.stop_price,
                    initial_stop_price=intent.stop_price,
                    highest_price=price,
                    opened_at=filled.submitted_at,
                )
            self._logger.info(
                "achat %s %.4f @ %.2f (stop %.2f)",
                intent.symbol,
                qty,
                price,
                intent.stop_price or 0.0,
            )
        else:
            account.cash += qty * price
            existing = account.positions.get(intent.symbol)
            if existing is not None:
                if qty >= existing.qty - 1e-6:
                    account.positions.pop(intent.symbol, None)
                else:
                    existing.qty -= qty
                    if intent.reason == "partial_take_profit":
                        existing.partial_tp_taken = True
                        existing.stop_price = max(
                            existing.stop_price or 0.0, existing.avg_entry_price
                        )
            else:
                account.positions.pop(intent.symbol, None)
        account.buying_power = account.cash

    def _flatten(
        self,
        account: AccountState,
        marks: Mapping[str, float],
        now: datetime,
        report: CycleReport,
    ) -> None:
        """Liquide toutes les positions après un circuit breaker."""
        if not account.positions:
            return
        decisions = self._risk.flatten_decisions(
            account,
            prices=marks,
            now=now,
            reason="circuit_breaker",
        )
        for decision in decisions:
            report.decisions.append(decision)
            self._audit_decision(decision)
            if self._execute_decision(decision, now, account, report):
                self._logger.critical(
                    "liquidation de %s demandée (circuit breaker)",
                    decision.intent.symbol if decision.intent else "?",
                )

    def _recompute_equity(self, account: AccountState, marks: Mapping[str, float]) -> None:
        """Marque les positions au dernier prix et recalcule l'équity."""
        self._risk.mark_to_market(account, marks)
        account.equity = account.cash + sum(
            position.market_value for position in account.positions.values()
        )
        account.buying_power = account.cash

    def _persist(self, account: AccountState) -> None:
        """Sauvegarde l'état de risque et les stops des positions ouvertes."""
        self._state.trading_day = (
            account.trading_day.isoformat() if account.trading_day else None
        )
        self._state.day_start_equity = account.day_start_equity
        self._state.peak_equity = account.peak_equity
        self._state.halted = account.halted
        self._state.halt_scope = account.halt_scope.value if account.halt_scope else None
        self._state.halt_reason = account.halt_reason
        self._state.stops = {
            symbol: {
                key: value
                for key, value in position.stop_state().items()
                if value is not None
            }
            for symbol, position in account.positions.items()
            if position.stop_price is not None
        }
        self._state_store.save(self._state)

    def _audit_decision(self, decision: RiskDecision) -> None:
        """Journalise une décision de risque."""
        self._audit.record(
            "risk_decision",
            {
                "decision_id": decision.decision_id,
                "approved": decision.approved,
                "code": decision.code.value,
                "reason": decision.reason,
                "symbol": decision.intent.symbol if decision.intent else None,
                "side": decision.intent.side.value if decision.intent else None,
                "qty": decision.intent.qty if decision.intent else None,
                "stop_price": decision.intent.stop_price if decision.intent else None,
                "metrics": dict(decision.metrics),
            },
        )

    def _audit_halt(self, halt: HaltEvent) -> None:
        """Journalise un circuit breaker."""
        self._audit.record(
            "circuit_breaker",
            {
                "scope": halt.scope.value,
                "reason": halt.reason,
                "session_date": halt.session_date.isoformat(),
                "equity": halt.equity,
                "daily_pnl_pct": halt.daily_pnl_pct,
                "drawdown_pct": halt.drawdown_pct,
            },
        )

    def _log_report(self, report: CycleReport) -> None:
        """Résumé compact d'un cycle dans les logs."""
        if report.skipped:
            self._logger.info("cycle ignoré: %s", report.skip_reason)
            return
        self._logger.info(
            "cycle %s | équité %.2f | signaux %s | décisions %d | ordres %d | erreurs %d",
            report.session_date.isoformat(),
            report.equity,
            report.signals,
            len(report.decisions),
            len(report.orders),
            len(report.errors),
        )


def _format_pct(value: float | int | None) -> str:
    """Formate un ratio en pourcentage lisible."""
    if value is None:
        return "n/a"
    return f"{float(value):.2%}"


def _format_number(value: float | int | None, digits: int = 2) -> str:
    """Formate un nombre avec un nombre fixe de décimales."""
    if value is None:
        return "n/a"
    return f"{float(value):,.{digits}f}"


def print_backtest_summary(result: BacktestResult) -> None:
    """Affiche les métriques principales d'un backtest."""
    metrics = result.metrics
    print("")
    print(f"Backtest {result.strategy_name} | {', '.join(result.symbols)}")
    print(f"  équity initiale      : {_format_number(metrics['initial_equity'])}")
    print(f"  équity finale        : {_format_number(metrics['final_equity'])}")
    print(f"  rendement total      : {_format_pct(metrics['total_return'])}")
    print(f"  CAGR                 : {_format_pct(metrics['cagr'])}")
    print(f"  Sharpe               : {_format_number(metrics['sharpe'])}")
    print(f"  Sortino              : {_format_number(metrics['sortino'])}")
    print(f"  max drawdown         : {_format_pct(metrics['max_drawdown'])}")
    print(f"  Calmar               : {_format_number(metrics['calmar'])}")
    print(f"  trades               : {metrics['trades']}")
    print(f"  taux de réussite     : {_format_pct(metrics['win_rate'])}")
    print(f"  profit factor        : {_format_number(metrics['profit_factor'])}")
    print(f"  espérance / trade    : {_format_number(metrics['expectancy'])}")
    print(f"  frais totaux         : {_format_number(metrics['total_fees'])}")
    print(f"  exposition moyenne   : {_format_pct(metrics['exposure_pct'])}")
    print(f"  circuit breakers     : {metrics['halt_events']}")
    print(f"  entrées refusées     : {metrics['entry_rejections']}")
    print(f"  positions ouvertes   : {metrics['open_positions_at_end']}")
    print(f"  buy & hold indice    : {_format_pct(metrics['benchmark_buy_hold'])}")
    print(f"  buy & hold panier    : {_format_pct(metrics['basket_buy_hold'])}")
    if result.halts:
        print("  halts:")
        for halt in result.halts:
            print(
                f"    - {halt.session_date.isoformat()} [{halt.scope.value}] {halt.reason}"
            )
    print("")


def _settings_from_args(args: argparse.Namespace) -> Settings:
    """Construit les Settings en appliquant les surcharges CLI."""
    overrides: dict[str, Any] = {}
    if getattr(args, "symbols", None):
        overrides["symbols"] = args.symbols
    if getattr(args, "dry_run", False):
        overrides["dry_run"] = True
    if getattr(args, "log_level", None):
        overrides["log_level"] = args.log_level
    return Settings(**overrides)


def cmd_live(args: argparse.Namespace) -> int:
    """Commande ``live`` : boucle paper trading."""
    settings = _settings_from_args(args)
    logger = configure_logging(settings)
    try:
        strategy = build_strategy(args.strategy, settings=settings)
    except (ConfigurationError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    try:
        broker = AlpacaPaperClient(settings)
    except (ConfigurationError, BrokerError) as exc:
        logger.error("%s", exc)
        return 2
    bot = TradingBot(
        settings,
        strategy,
        broker,
        notifier=_build_notifier(settings),
        news_analyzer=_build_news_analyzer(settings),
    )
    if getattr(args, "watch_only", False):
        report = bot.run_watch_only()
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return 0 if not report.errors else 1
    if args.once:
        report = bot.run_once()
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return 0 if not report.errors else 1
    try:
        bot.run_forever()
    except KeyboardInterrupt:
        logger.info("interruption clavier")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Commande ``check`` : diagnostic lecture seule, sans ordre."""
    settings = _settings_from_args(args)
    logger = configure_logging(settings)
    try:
        strategy = build_strategy(args.strategy, settings=settings)
    except (ConfigurationError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    try:
        broker = AlpacaPaperClient(settings)
        account = broker.get_account()
        positions = broker.get_positions()
        market_open = broker.is_market_open()
    except (ConfigurationError, BrokerError) as exc:
        logger.error("%s", exc)
        return 2

    symbol = settings.symbols[0]
    benchmark_symbol = settings.benchmark_symbol
    print("")
    print("Diagnostic paper trading Alpaca")
    print(f"  endpoint        : {settings.alpaca_trading_url}")
    print(f"  paper           : {settings.is_paper}")
    print(f"  marché ouvert   : {market_open}")
    print(f"  équité          : {_format_number(account.equity)}")
    print(f"  cash            : {_format_number(account.cash)}")
    print(f"  buying power    : {_format_number(account.buying_power)}")
    print(f"  univers         : {', '.join(settings.symbols)}")
    print(f"  stratégie       : {strategy.name}")
    print(f"  benchmark       : {benchmark_symbol}")
    print(f"  timeframe       : {settings.timeframe}")

    benchmark_complete: BarFrame | None = None
    try:
        benchmark_raw = broker.fetch_bars(
            benchmark_symbol,
            timeframe=settings.timeframe,
            lookback_days=settings.lookback_days,
            now=_utc_now(),
        )
        benchmark_complete = drop_incomplete_bar(
            benchmark_raw,
            settings.timeframe,
            _utc_now(),
            market_timezone=settings.market_timezone,
        )
        if getattr(strategy, "benchmark_symbol", None) is not None:
            _inject_benchmark(strategy, benchmark_complete, benchmark_symbol)
    except (BrokerError, DataError, StrategyError) as exc:
        logger.warning("benchmark %s indisponible: %s", benchmark_symbol, exc)

    sma_period = settings.risk.regime_sma_period
    if benchmark_complete is not None and len(benchmark_complete) >= sma_period:
        b_close = benchmark_complete["close"].astype(float)
        b_sma = float(b_close.rolling(sma_period).mean().iloc[-1])
        b_last = float(b_close.iloc[-1])
        regime_label = "BULL" if b_last > b_sma else "BEAR"
        extra = "" if regime_label == "BULL" else " [BEAR_REGIME_VETO actif sur offensifs]"
        print(
            f"  régime {benchmark_symbol} (SMA {sma_period}) : {regime_label} "
            f"(cours {_format_number(b_last)} vs SMA {_format_number(b_sma)}){extra}"
        )
    elif benchmark_complete is not None:
        print(
            f"  régime {benchmark_symbol} (SMA {sma_period}) : INCONNU "
            f"({len(benchmark_complete)}/{sma_period} barres requises)"
        )
    else:
        print(f"  régime {benchmark_symbol} (SMA {sma_period}) : INDISPONIBLE")

    if settings.safe_haven_enabled:
        safe_symbols_str = ", ".join(settings.safe_haven_symbols)
        cash_buf = account.equity * settings.risk.cash_buffer_pct
        max_sh_notional = account.equity * settings.risk.max_safe_haven_total_pct
        sh_held = {
            sym: pos
            for sym, pos in positions.items()
            if sym in settings.safe_haven_symbols and pos.qty > 0
        }
        sh_total_val = sum(p.market_value for p in sh_held.values())
        sh_pct = (sh_total_val / account.equity) if account.equity > 0 else 0.0
        print(
            f"  poche de repli  : activée (symboles: {safe_symbols_str} | "
            f"buffer cash min: {_format_pct(settings.risk.cash_buffer_pct)} [{_format_number(cash_buf)}] | "
            f"plafond: {_format_pct(settings.risk.max_safe_haven_total_pct)} [{_format_number(max_sh_notional)}])"
        )
        if sh_held:
            sh_details = ", ".join(
                f"{sym}: {p.qty:.2f} parts ({_format_number(p.market_value)})"
                for sym, p in sh_held.items()
            )
            print(
                f"  repli détenu    : {_format_number(sh_total_val)} "
                f"({_format_pct(sh_pct)}) [{sh_details}]"
            )
        else:
            print(f"  repli détenu    : {_format_number(0.0)} (0.00%) [aucune position]")
    else:
        print("  poche de repli  : désactivée")

    if strategy.name == "shield_sword":
        shield_weight = getattr(
            getattr(strategy, "params", None), "shield_allocation_pct", 0.0
        )
        print(
            f"  bouclier        : {settings.risk.shield_symbol} "
            f"({_format_pct(shield_weight)} de l'équité, sans stop)"
        )

    try:
        bars = broker.fetch_bars(
            symbol,
            timeframe=settings.timeframe,
            lookback_days=settings.lookback_days,
            now=_utc_now(),
        )
        complete = drop_incomplete_bar(
            bars,
            settings.timeframe,
            _utc_now(),
            market_timezone=settings.market_timezone,
        )
        if hasattr(strategy, "set_universe"):
            strategy.set_universe({symbol: complete})
        signal = strategy.latest_signal(complete, symbol=symbol)
    except (BrokerError, DataError, StrategyError) as exc:
        logger.error("données indisponibles: %s", exc)
        return 1

    atr = signal.indicators.get("atr", 0.0)
    print(f"  barres          : {len(complete)} (dernière {complete.index[-1]})")
    print(f"  prix            : {_format_number(signal.price)}")
    print(f"  signal          : {signal.action.value} — {signal.reason}")
    print(f"  ATR             : {_format_number(atr)}")
    if atr > 0:
        stop = signal.price - atr * settings.risk.atr_stop_multiple
        print(f"  stop ATR proposé: {_format_number(stop)}")

    # Le diagnostic doit annoncer la même chose que le cycle : un signal BUY
    # peut être refusé la nuit venue pour cause d'actualité, et c'est ici qu'on
    # le voit. Sans ce bloc, le diagnostic promettrait un achat qui n'aura pas
    # lieu — précisément le genre d'écart qui rend un outil trompeur.
    engine = RiskEngine(settings.risk)
    panic = settings.risk.news_panic_exit_threshold
    assessment = _assess_news(settings, [symbol]).get(symbol)
    print("")
    if assessment is None:
        print("  actualités      : filtre inactif (aucune analyse)")
    else:
        print(
            f"  actualités      : score {assessment.sentiment_score:+.2f} "
            f"(confiance {assessment.confidence:.2f}, "
            f"{assessment.articles_count} article(s))"
        )
        print(
            f"  effet           : {news_effect_label(assessment, engine=engine, panic=panic)}"
        )
        if assessment.is_critical_threat:
            print("  menace critique : oui")
        print(f"  motif           : {assessment.rationale}")
        if signal.action is SignalType.BUY and engine.news_vetoes_entry(assessment):
            print(
                "  ATTENTION       : ce signal d'achat serait REFUSÉ ([NEWS_VETO]) "
                "lors du prochain cycle."
            )
    print("")
    return 0


def news_effect_label(
    assessment: NewsAssessment, *, engine: RiskEngine, panic: float
) -> str:
    """Qualifie l'effet d'une note d'actualités sur la décision d'achat.

    Source unique de vérité pour l'affichage : le radar, le diagnostic en
    ligne de commande et le diagnostic web doivent annoncer la même chose que
    ce que le moteur de risque appliquera réellement.

    Args:
        assessment: Note d'actualités du symbole.
        engine: Moteur portant les seuils configurés.
        panic: Seuil de sortie d'urgence.

    Returns:
        ``"SORTIE URGENCE"``, ``"VETO ENTRÉE"`` ou ``"—"``.
    """
    if assessment.is_critical_threat and assessment.sentiment_score <= panic:
        return "SORTIE URGENCE"
    if engine.news_vetoes_entry(assessment):
        return "VETO ENTRÉE"
    return "—"


def _format_news_row(
    symbol: str,
    assessment: NewsAssessment,
    *,
    panic: float,
    engine: RiskEngine,
) -> str:
    """Met en forme une ligne du radar d'actualités."""
    effect = news_effect_label(assessment, engine=engine, panic=panic)
    rationale = " ".join(assessment.rationale.split())
    return (
        f"{symbol:<9}{assessment.sentiment_score:>+8.2f}"
        f"{assessment.confidence:>8.2f}"
        f"{'oui' if assessment.is_critical_threat else 'non':>9}"
        f"{assessment.articles_count:>5}  {effect:<15}{rationale[:44]}"
    )


def cmd_news(args: argparse.Namespace) -> int:
    """Commande ``news`` : radar de sentiment sur les actualités récentes.

    Affiche, pour chaque symbole, la note du modèle et l'effet qu'elle
    produirait sur le moteur de risque. Utile pour comprendre *pourquoi* un
    achat a été refusé, sans fouiller ``audit.jsonl``.
    """
    settings = _settings_from_args(args)
    configure_logging(settings)
    analyzer = _build_news_analyzer(settings)
    if analyzer is None:
        if not settings.news_filter_enabled:
            print("Filtre d'actualités désactivé (NEWS_FILTER_ENABLED=false).")
        else:
            print("Identifiants Alpaca absents : actualités illisibles.")
        return 2

    symbols = list(settings.symbols)
    veto = settings.risk.news_veto_threshold
    panic = settings.risk.news_panic_exit_threshold
    engine = RiskEngine(settings.risk)

    # Tout est écrit avec flush : un appel au modèle prend plusieurs secondes
    # et l'absence de retour immédiat donne l'impression que rien ne se passe.
    print(
        f"Radar d'actualités | fenêtre {settings.news_lookback_hours} h | "
        f"modèle {settings.openrouter_model}",
        flush=True,
    )
    print(
        f"seuils : veto d'entrée ≤ {veto:+.2f} | sortie d'urgence ≤ {panic:+.2f}",
        flush=True,
    )
    print(
        f"{len(symbols)} symbole(s). Un appel au modèle par symbole ayant des "
        "articles inédits ; les autres sont servis par le cache.",
        flush=True,
    )

    header = (
        f"{'Symbole':<9}{'Score':>8}{'Conf.':>8}{'Critique':>9}"
        f"{'Art.':>5}  {'Effet':<15}Motif"
    )
    print(header, flush=True)
    print("-" * 78, flush=True)

    def emit(symbol: str, assessment: NewsAssessment) -> None:
        """Affiche une ligne sitôt calculée, sans attendre les suivantes."""
        print(
            _format_news_row(symbol, assessment, panic=panic, engine=engine),
            flush=True,
        )

    analyzer.assess(symbols, now=get_online_now(), on_progress=emit)

    if not analyzer.llm_enabled:
        print("\n(Attention : aucune clé OpenRouter, les scores sont neutres.)")
    return 0


def cmd_publish_stats(args: argparse.Namespace) -> int:
    """Commande ``publish-stats`` : republie l'historique local vers GitHub.

    Sert à rattraper les cycles antérieurs à la mise en place de la
    publication, ou ceux dont l'envoi a échoué (PC éteint, réseau coupé, jeton
    posé après coup). La publication normale, elle, a lieu à la fin de chaque
    cycle sans rien demander.
    """
    settings = _settings_from_args(args)
    logger = configure_logging(settings)
    publisher = _build_stats_publisher(settings)
    if publisher is None:
        logger.error(
            "publication non configurée : renseignez STATS_GITHUB_REPO et "
            "STATS_GITHUB_TOKEN (voir .env.example)"
        )
        return 2

    store = HistoryStore(settings.state_dir / "history.jsonl")
    rows = store.entries(limit=-1)
    if not rows:
        logger.error("aucun cycle enregistré dans %s", store.path)
        return 1

    published = 0
    # Du plus ancien au plus récent : le fichier distant se construit dans
    # l'ordre, et la règle « une entrée par séance » s'applique correctement.
    for row in reversed(rows):
        report = {
            key: value
            for key, value in row.items()
            if key not in {"recorded_at", "source", "recovered_from"}
        }
        session = str(report.get("session_date", "?"))
        if publisher.send("cycle_trading", report):
            published += 1
            print(f"  publié : {session}")
        else:
            logger.warning("cycle %s non publié", session)

    total = len(rows)
    print(f"\n{published}/{total} cycle(s) publié(s) dans {settings.stats_github_repo}")
    return 0 if published == total else 1


def cmd_backtest(args: argparse.Namespace) -> int:
    """Commande ``backtest`` : télécharge les données et exécute la simulation."""
    settings = Settings()
    logger = configure_logging(settings)
    provider: MarketDataProvider = YFinanceProvider()
    if not args.no_cache:
        provider = CachedProvider(
            provider,
            settings.state_dir / "cache",
            refresh=args.refresh,
        )

    try:
        start = datetime.combine(date.fromisoformat(args.start), dt_time.min, tzinfo=UTC)
        end = datetime.combine(
            date.fromisoformat(args.end), dt_time.min, tzinfo=UTC
        ) + timedelta(days=1)
    except ValueError as exc:
        logger.error("dates invalides: %s", exc)
        return 2
    if end <= start:
        logger.error("la date de fin doit être postérieure à la date de début")
        return 2

    try:
        strategy = build_strategy(args.strategy, args, settings=settings)
    except ValueError as exc:
        logger.error("paramètres de stratégie invalides: %s", exc)
        return 2
    except ConfigurationError as exc:
        logger.error("%s", exc)
        return 2

    # Période de warm-up (200 barres journalières ~ 320 jours calendaires) pour
    # que la SMA(200) et les calculs de momentum soient actifs dès le jour 1.
    warmup_days = 320 if args.interval in {"1d", "1Day"} else 30
    fetch_start = start - timedelta(days=warmup_days)

    data: dict[str, BarFrame] = {}
    for symbol in Settings(symbols=args.symbols).symbols:
        try:
            data[symbol] = provider.fetch(symbol, fetch_start, end, interval=args.interval)
        except DataError as exc:
            logger.error("%s", exc)
            return 1
        logger.info(
            "%s: %d barres (%s → %s)",
            symbol,
            len(data[symbol]),
            data[symbol].index[0].date(),
            data[symbol].index[-1].date(),
        )

    # Le benchmark n'est PAS ajouté à ``data`` : le moteur ne doit pas le
    # trader. Il pilote le régime de marché (SMA longue) et sert de référence
    # de force relative aux stratégies qui en demandent un.
    benchmark_symbol = settings.benchmark_symbol
    benchmark_bars: BarFrame | None = None
    try:
        benchmark_bars = provider.fetch(
            benchmark_symbol, fetch_start, end, interval=args.interval
        )
        _inject_benchmark(strategy, benchmark_bars, benchmark_symbol)
    except (DataError, StrategyError) as exc:
        logger.error("benchmark %s indisponible: %s", benchmark_symbol, exc)
        return 1
    logger.info(
        "%s (benchmark): %d barres (%s → %s)",
        benchmark_symbol,
        len(benchmark_bars),
        benchmark_bars.index[0].date(),
        benchmark_bars.index[-1].date(),
    )

    config = BacktestConfig(
        initial_equity=args.initial_equity,
        commission_pct=args.commission,
        slippage_pct=args.slippage,
    )
    runner = BacktestRunner(
        strategy,
        config=config,
        risk_config=settings.risk,
    )
    result = runner.run(data, benchmark=benchmark_bars, start_date=start)
    print_backtest_summary(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Résultats JSON : {args.output}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        result.equity_curve.to_csv(args.csv)
        print(f"Courbe d'équity : {args.csv}")
    if args.trades:
        args.trades.parent.mkdir(parents=True, exist_ok=True)
        result.trades_frame().to_csv(args.trades, index=False)
        print(f"Trades CSV : {args.trades}")
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    """Commande ``web`` : interface locale, pensée pour les non-financiers."""
    settings = _settings_from_args(args)
    configure_logging(settings)
    from trading_bot.web.app import serve

    try:
        serve(settings, host=args.host, port=args.port)
    except (ConfigurationError, ValueError) as exc:
        logging.getLogger("trading_bot").error("%s", exc)
        return 2
    return 0


def _withdrawal_payload(kind: str, result: WithdrawalResult) -> dict[str, Any]:
    """Résumé JSON d'un retrait, identique en CLI et dans l'interface."""
    return {
        "ok": not result.errors,
        "kind": kind,
        "fills": [
            {
                "symbol": fill.symbol,
                "qty": fill.qty,
                "price": fill.price,
                "cash_freed": round(fill.cash_freed, 2),
                "reason": fill.reason,
            }
            for fill in result.fills
        ],
        "cash": round(result.cash, 2),
        "floor": result.floor,
        "errors": list(result.errors),
    }


def cmd_withdraw(args: argparse.Namespace) -> int:
    """Commande ``withdraw`` : retrait manuel ou sanctuarisation de cash."""
    settings = _settings_from_args(args)
    logger = configure_logging(settings)
    try:
        broker = AlpacaPaperClient(settings)
    except (ConfigurationError, BrokerError) as exc:
        logger.error("%s", exc)
        return 2

    service = WithdrawalService(
        settings,
        broker,
        state_store=StateStore(settings.state_dir / "state.json"),
        audit=AuditLog(settings.state_dir / "audit.jsonl"),
        env_path=PROJECT_ROOT / ".env",
    )

    try:
        if getattr(args, "free_cash", None) is not None:
            result = service.withdraw_free_cash(float(args.free_cash))
            print(
                json.dumps(
                    _withdrawal_payload("free_cash", result), indent=2, ensure_ascii=False
                )
            )
            return 0 if not result.errors else 1

        symbol = getattr(args, "symbol", None)
        if not symbol:
            logger.error(
                "veuillez spécifier soit --symbol avec (--amount ou --pct), "
                "soit --free-cash"
            )
            return 2
        result = service.withdraw_symbol(
            str(symbol),
            amount=getattr(args, "amount", None),
            pct=getattr(args, "pct", None),
        )
        print(
            json.dumps(_withdrawal_payload("symbol", result), indent=2, ensure_ascii=False)
        )
        return 0 if not result.errors else 1
    except (BrokerError, ValueError) as exc:
        logger.error("%s", exc)
        return 2


def apply_equity_rebase(
    settings: Settings,
    snapshot: AccountSnapshot,
    *,
    reason: str,
    state_store: StateStore,
    audit: AuditLog,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Ré-ancre le pic d'équité sur la valeur courante et rend un résumé.

    Partagé par la commande ``rebase`` et l'interface web : le raisonnement
    (pourquoi, avec quelle justification, ce qui est levé) ne doit exister
    qu'une fois.

    Args:
        settings: Configuration effective.
        snapshot: Compte lu chez le courtier.
        reason: Justification en clair, recopiée dans l'audit.
        state_store: Persistance de l'état de risque.
        audit: Journal d'audit.
        dry_run: Si ``True``, n'écrit rien.

    Returns:
        ``{"previous_peak", "new_peak", "halt_lifted", "dry_run"}``.

    Raises:
        RiskError: justification vide ou équité invalide.
    """
    state = state_store.load()
    account = AccountState(
        equity=snapshot.equity,
        cash=snapshot.cash,
        buying_power=snapshot.buying_power,
        peak_equity=state.peak_equity,
        day_start_equity=state.day_start_equity,
        halted=state.halted,
        halt_scope=HaltScope(state.halt_scope) if state.halt_scope else None,
        halt_reason=state.halt_reason,
    )
    previous_peak, lifted = RiskEngine(settings.risk).rebase_equity_reference(
        account, reason=reason
    )
    payload: dict[str, Any] = {
        "previous_peak": previous_peak,
        "new_peak": account.equity,
        "halt_lifted": lifted,
        "dry_run": dry_run,
    }
    if dry_run:
        return payload

    state.peak_equity = account.peak_equity
    state.day_start_equity = account.day_start_equity
    state.halted = account.halted
    state.halt_scope = account.halt_scope.value if account.halt_scope else None
    state.halt_reason = account.halt_reason
    state_store.save(state)
    audit.record(
        "equity_rebase",
        {
            "reason": reason,
            "previous_peak": previous_peak,
            "new_peak": account.peak_equity,
            "halt_lifted": lifted,
            "equity": account.equity,
        },
    )
    return payload


def cmd_rebase(args: argparse.Namespace) -> int:
    """Commande ``rebase`` : ré-ancre le pic d'équité après un mouvement de cash.

    Un dépôt ou un retrait fait chez le courtier modifie l'équité sans qu'aucun
    trade n'ait eu lieu. Le drawdown mesuré depuis l'ancien pic devient alors
    fictif et déclenche l'arrêt permanent. Cette commande, explicitement
    demandée avec sa justification, recale la référence et lève l'arrêt.
    """
    settings = _settings_from_args(args)
    logger = configure_logging(settings)
    reason = str(getattr(args, "reason", "") or "").strip()
    if not reason:
        logger.error("une justification est requise : --reason « retrait de X $ »")
        return 2

    try:
        broker = AlpacaPaperClient(settings)
        snapshot = broker.get_account()
    except (ConfigurationError, BrokerError) as exc:
        logger.error("%s", exc)
        return 2

    try:
        payload = apply_equity_rebase(
            settings,
            snapshot,
            reason=reason,
            state_store=StateStore(settings.state_dir / "state.json"),
            audit=AuditLog(settings.state_dir / "audit.jsonl"),
            dry_run=bool(getattr(args, "dry_run", False)),
        )
    except RiskError as exc:
        logger.error("%s", exc)
        return 2

    logger.info(
        "pic d'équité recalé : %s -> %.2f (%s)",
        f"{payload['previous_peak']:.2f}"
        if payload["previous_peak"] is not None
        else "n/a",
        payload["new_peak"],
        reason,
    )
    print(json.dumps({"ok": True, **payload}, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construit le parseur de commandes."""
    parser = argparse.ArgumentParser(
        prog="trading-bot",
        description=(
            "Bot de trading systématique restreint au paper trading Alpaca. "
            "Aucun ordre réel ne peut être transmis."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    live = subparsers.add_parser("live", help="boucle paper trading")
    live.add_argument("--once", action="store_true", help="exécuter un seul cycle")
    live.add_argument(
        "--watch-only",
        action="store_true",
        help="mode vigie horaire : surveillance temps réel des stops et news sans nouveaux achats",
    )
    live.add_argument(
        "--dry-run",
        action="store_true",
        help="analyser et journaliser sans transmettre d'ordre",
    )
    live.add_argument("--symbols", help="univers CSV (ex: AAPL,MSFT)")
    live.add_argument(
        "--strategy",
        choices=STRATEGY_NAMES,
        default="ema_cross",
        help="moteur de signaux (défaut: ema_cross)",
    )
    live.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    live.set_defaults(func=cmd_live)

    check = subparsers.add_parser(
        "check", help="diagnostic lecture seule (compte, données, signal)"
    )
    check.add_argument("--symbols", help="univers CSV")
    check.add_argument(
        "--strategy",
        choices=STRATEGY_NAMES,
        default="ema_cross",
        help="moteur de signaux (défaut: ema_cross)",
    )
    check.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    check.set_defaults(func=cmd_check)

    web = subparsers.add_parser(
        "web",
        help="interface web locale (tableau de bord, backtests, pilotage)",
    )
    web.add_argument(
        "--host",
        default="127.0.0.1",
        help="hôte d'écoute (localhost uniquement, par sécurité)",
    )
    web.add_argument("--port", type=int, default=8484)
    web.add_argument("--symbols", help="univers CSV")
    web.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    web.set_defaults(func=cmd_web)

    news = subparsers.add_parser(
        "news",
        help="radar de sentiment des actualités récentes",
    )
    news.add_argument(
        "--symbols",
        help="tickers à noter, séparés par des virgules (défaut : l'univers)",
    )
    news.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    news.set_defaults(func=cmd_news)

    publish = subparsers.add_parser(
        "publish-stats",
        help="republier l'historique local vers GitHub (rattrapage)",
    )
    publish.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    publish.set_defaults(func=cmd_publish_stats)

    backtest = subparsers.add_parser(
        "backtest", help="backtest déterministe sur données yfinance"
    )
    backtest.add_argument("--symbols", default="AAPL,MSFT")
    backtest.add_argument("--start", default="2022-01-01")
    backtest.add_argument("--end", default=date.today().isoformat())
    backtest.add_argument("--interval", default="1d")
    backtest.add_argument("--initial-equity", type=float, default=100_000.0)
    backtest.add_argument("--commission", type=float, default=0.001)
    backtest.add_argument("--slippage", type=float, default=0.001)
    backtest.add_argument(
        "--strategy",
        choices=STRATEGY_NAMES,
        default="ema_cross",
        help="moteur de signaux (défaut: ema_cross)",
    )
    backtest.add_argument("--fast", type=int, default=20, help="EMA rapide (ema_cross)")
    backtest.add_argument("--slow", type=int, default=50, help="EMA lente (ema_cross)")
    backtest.add_argument(
        "--entry", type=int, default=20, help="canal d'entrée (donchian_breakout)"
    )
    backtest.add_argument(
        "--exit-channel",
        type=int,
        default=10,
        help="canal de sortie (donchian_breakout)",
    )
    backtest.add_argument(
        "--sma-long", type=int, default=200, help="filtre de tendance (donchian_breakout)"
    )
    backtest.add_argument(
        "--mom-fast", type=int, default=60, help="momentum rapide (relative_momentum)"
    )
    backtest.add_argument(
        "--mom-slow", type=int, default=120, help="momentum lent (relative_momentum)"
    )
    backtest.add_argument(
        "--shield-pct",
        type=float,
        default=None,
        help="allocation bouclier (shield_sword, défaut: SHIELD_ALLOCATION_PCT)",
    )
    backtest.add_argument(
        "--sword-pct",
        type=float,
        default=None,
        help="allocation épée (shield_sword, défaut: SWORD_ALLOCATION_PCT)",
    )
    backtest.add_argument(
        "--sword-top-n",
        type=int,
        default=None,
        help="nombre d'actions dans l'épée (shield_sword, défaut: SWORD_TOP_N)",
    )
    backtest.add_argument(
        "--sword-rebalance-days",
        type=int,
        default=None,
        help="jours entre rebalancements (shield_sword, défaut: SWORD_REBALANCE_DAYS)",
    )
    backtest.add_argument(
        "--sword-trend-sma",
        type=int,
        default=None,
        help="SMA de tendance (shield_sword, défaut: SWORD_TREND_SMA)",
    )
    backtest.add_argument("--output", type=Path, help="fichier JSON de résultats")
    backtest.add_argument("--csv", type=Path, help="fichier CSV de la courbe d'équity")
    backtest.add_argument("--trades", type=Path, help="fichier CSV des trades")
    backtest.add_argument(
        "--no-cache",
        action="store_true",
        help="ignorer le cache disque et retélécharger les barres",
    )
    backtest.add_argument(
        "--refresh",
        action="store_true",
        help="retélécharger les barres puis réécrire le cache",
    )
    backtest.set_defaults(func=cmd_backtest)

    withdraw = subparsers.add_parser(
        "withdraw", help="retrait de cash manuel ou sanctuarisation de plancher"
    )
    withdraw.add_argument("--symbol", help="symbole à vendre (ex: NVDA)")
    withdraw.add_argument("--amount", type=float, help="montant en dollars à liquider")
    withdraw.add_argument(
        "--pct", type=float, help="pourcentage de la ligne à liquider (0-100)"
    )
    withdraw.add_argument(
        "--free-cash", type=float, help="montant de cash cible à sanctuariser"
    )
    withdraw.add_argument("--dry-run", action="store_true", help="simuler sans transmettre")
    withdraw.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    withdraw.set_defaults(func=cmd_withdraw)

    rebase = subparsers.add_parser(
        "rebase",
        help="ré-ancrer le pic d'équité après un dépôt ou un retrait chez le courtier",
    )
    rebase.add_argument(
        "--reason",
        required=True,
        help="justification en clair, recopiée dans audit.jsonl",
    )
    rebase.add_argument(
        "--dry-run", action="store_true", help="afficher l'effet sans rien enregistrer"
    )
    rebase.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    rebase.set_defaults(func=cmd_rebase)

    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée CLI, retourne un code de sortie."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigurationError as exc:
        logging.getLogger("trading_bot").error("%s", exc)
        return 2
    except KeyboardInterrupt:
        return 130


def main() -> None:
    """Entrée console ``python -m trading_bot.main``."""
    raise SystemExit(cli(sys.argv[1:]))


if __name__ == "__main__":
    main()
