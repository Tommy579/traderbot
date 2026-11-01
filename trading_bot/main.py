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
from typing import Any, Final, Protocol
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import BaseModel, ConfigDict

from trading_bot.backtest.runner import BacktestResult, BacktestRunner
from trading_bot.config import (
    BacktestConfig,
    ConfigurationError,
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
    trend_regime,
)
from trading_bot.execution.alpaca_client import (
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

LOGGER = logging.getLogger("trading_bot.main")
_ORDER_FILL_TIMEOUT_SECONDS: Final[float] = 30.0
STRATEGY_NAMES: Final[tuple[str, ...]] = (
    "ema_cross",
    "donchian_breakout",
    "relative_momentum",
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
    cause = (
        status.closure.value if status.closure is not None else "séance non ouverte"
    )
    message = f"marché fermé ({cause})"
    if status.next_open is not None:
        local = status.next_open.astimezone(ZoneInfo(market_timezone))
        suffix = local.tzname() or market_timezone
        message += (
            f" — réouverture {_JOURS_FR[local.weekday()]} {local:%d/%m à %H:%M}"
            f" ({suffix})"
        )
    return message


def build_strategy(name: str, args: argparse.Namespace | None = None) -> Strategy:
    """Construit la stratégie nommée, avec les surcharges CLI éventuelles.

    Raises:
        ConfigurationError: stratégie inconnue.
        ValueError: paramètres de stratégie invalides.
    """
    namespace = args or argparse.Namespace()
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
    """Horloge par défaut : maintenant en UTC, timezone-aware."""
    return datetime.now(UTC)


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
    stops: dict[str, dict[str, float]] = {}


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
    skipped: bool = False
    skip_reason: str | None = None
    equity: float = 0.0
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
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "equity": self.equity,
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

    def _execute_cycle(self) -> CycleReport:
        """Exécute un cycle complet de décision et d'exécution."""
        now = self._clock()
        if now.tzinfo is None:
            raise ConfigurationError("l'horloge du bot doit être timezone-aware")
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
        try:
            benchmark_bars = self._fetch_complete_bars(benchmark_symbol, now)
            _inject_benchmark(self._strategy, benchmark_bars, benchmark_symbol)
            regime = trend_regime(benchmark_bars)
            if len(regime):
                bull_market = bool(regime.iloc[-1])
        except (DataError, BrokerError, StrategyError) as exc:
            report.errors.append(f"{benchmark_symbol}: {exc}")
            self._logger.warning(
                "benchmark %s indisponible, régime prudent: %s", benchmark_symbol, exc
            )
        # Régime prudent par défaut ; haussier uniquement si le benchmark
        # clôture au-dessus de sa SMA longue.
        self._risk.set_market_regime(bull_market=bull_market)
        if bull_market:
            self._logger.info(
                "régime haussier: %s > SMA200, plafonds desserrés",
                benchmark_symbol,
            )

        universe = sorted(set(self._settings.symbols) | set(account.positions))
        bars_by_symbol: dict[str, BarFrame] = {}
        signals: dict[str, Signal] = {}
        for symbol in universe:
            try:
                bars = self._fetch_complete_bars(symbol, now)
                signal = self._strategy.latest_signal(bars, symbol=symbol)
            except (DataError, BrokerError, StrategyError) as exc:
                report.errors.append(_symbol_message(symbol, exc))
                self._logger.warning("analyse %s ignorée: %s", symbol, exc)
                continue
            bars_by_symbol[symbol] = bars
            signals[symbol] = signal
            report.signals[symbol] = signal.action.value

        marks = self._resolve_marks(universe, bars_by_symbol, account, report)
        self._adopt_unprotected_positions(account, signals, marks, report, now=now)

        if halt is not None:
            self._flatten(account, marks, now, report)
            self._persist(account)
            report.equity = account.equity
            self._notify(report)
            return report

        exited_symbols: set[str] = set()
        for symbol in sorted(list(account.positions)):
            position = account.positions.get(symbol)
            if position is None:
                continue
            price = marks.get(symbol, position.current_price)
            decision = self._risk.evaluate_stop_exit(position, price=price, now=now)
            if decision is None:
                exit_signal = signals.get(symbol)
                if exit_signal is not None and exit_signal.action is SignalType.SELL:
                    decision = self._risk.evaluate_exit(
                        position=position,
                        price=price,
                        now=now,
                        reason="strategy_exit",
                    )
            if decision is None:
                continue
            report.decisions.append(decision)
            self._audit_decision(decision)
            if self._execute_decision(decision, now, account, report):
                exited_symbols.add(symbol)
                self._logger.info("position %s liquidée (%s)", symbol, decision.reason)

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
            decision = self._risk.evaluate_entry(
                symbol=symbol,
                price=price,
                atr=atr,
                account=account,
                now=now,
                session_date=session_date,
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

        self._recompute_equity(account, marks)
        self._persist(account)
        report.equity = account.equity
        self._notify(report)
        return report

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
        return True

    def _cancel_open_orders(self, symbol: str, report: CycleReport) -> bool:
        """Annule les ordres ouverts du symbole, et échoue en mode fermé."""
        cancel = getattr(self._broker, "cancel_open_orders", None)
        if cancel is None:
            error = (
                f"{symbol}: courtier incapable de confirmer l'annulation des ordres ouverts"
            )
            report.errors.append(error)
            self._audit.record(
                "cancel_open_orders_failed", {"symbol": symbol, "error": error}
            )
            self._logger.error("%s", error)
            return False
        try:
            cancelled = int(cancel(symbol))
        except BrokerError as exc:
            error = f"{symbol}: annulation des ordres ouverts impossible: {exc}"
            report.errors.append(error)
            self._audit.record(
                "cancel_open_orders_failed", {"symbol": symbol, "error": str(exc)}
            )
            self._logger.error("%s", error)
            return False
        self._audit.record("cancel_open_orders", {"symbol": symbol, "count": cancelled})
        return True

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
                "entrée %s %.4f @ %.2f (stop %.2f)",
                intent.symbol,
                qty,
                price,
                intent.stop_price or 0.0,
            )
        else:
            account.cash += qty * price
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
                key: float(value)
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
        strategy = build_strategy(args.strategy)
    except (ConfigurationError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    try:
        broker = AlpacaPaperClient(settings)
    except (ConfigurationError, BrokerError) as exc:
        logger.error("%s", exc)
        return 2
    bot = TradingBot(settings, strategy, broker, notifier=_build_notifier(settings))
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
        strategy = build_strategy(args.strategy)
    except (ConfigurationError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    try:
        broker = AlpacaPaperClient(settings)
        account = broker.get_account()
        market_open = broker.is_market_open()
    except (ConfigurationError, BrokerError) as exc:
        logger.error("%s", exc)
        return 2

    symbol = settings.symbols[0]
    benchmark_symbol: str | None = getattr(strategy, "benchmark_symbol", None)
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
    if benchmark_symbol:
        print(f"  benchmark       : {benchmark_symbol}")
    print(f"  timeframe       : {settings.timeframe}")

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
        if benchmark_symbol is not None:
            benchmark_raw = broker.fetch_bars(
                benchmark_symbol,
                timeframe=settings.timeframe,
                lookback_days=settings.lookback_days,
                now=_utc_now(),
            )
            _inject_benchmark(
                strategy,
                drop_incomplete_bar(
                    benchmark_raw,
                    settings.timeframe,
                    _utc_now(),
                    market_timezone=settings.market_timezone,
                ),
                benchmark_symbol,
            )
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
    print("")
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
        strategy = build_strategy(args.strategy, args)
    except ValueError as exc:
        logger.error("paramètres de stratégie invalides: %s", exc)
        return 2
    except ConfigurationError as exc:
        logger.error("%s", exc)
        return 2

    data: dict[str, BarFrame] = {}
    for symbol in Settings(symbols=args.symbols).symbols:
        try:
            data[symbol] = provider.fetch(symbol, start, end, interval=args.interval)
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
            benchmark_symbol, start, end, interval=args.interval
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
    result = runner.run(data, benchmark=benchmark_bars)
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
