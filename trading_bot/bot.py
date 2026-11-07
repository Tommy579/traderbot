"""Orchestrateur : données → stratégie → risque → exécution.

Ce module porte la boucle de trading elle-même (``TradingBot``), la
construction des stratégies et les helpers de cycle (benchmark, actualités,
notifications). La persistance vit dans ``trading_bot.state`` et la ligne de
commande dans ``trading_bot.main`` : l'interface web peut ainsi importer
l'orchestrateur sans traîner argparse ni formatage de terminal.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Final, Protocol, TypeVar
from zoneinfo import ZoneInfo

import pandas as pd

from trading_bot.clock import clock_offset_seconds
from trading_bot.config import ConfigurationError, RiskConfig, Settings
from trading_bot.data.loader import (
    BarFrame,
    DataError,
    drop_incomplete_bar,
    momentum_zscores,
)
from trading_bot.data.news_sentiment import NewsAssessment, NewsSentimentAnalyzer
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
from trading_bot.state import AuditLog, StateStore, utc_now
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
from trading_bot.withdrawal import cancel_open_orders_verified

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
        clock: Callable[[], datetime] = utc_now,
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
