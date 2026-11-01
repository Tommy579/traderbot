"""Tests d'intégration de l'orchestrateur live avec un courtier simulé."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import trading_bot.main as trading_main
from trading_bot.config import ConfigurationError, Settings
from trading_bot.data.news_sentiment import NewsAssessment
from trading_bot.execution.alpaca_client import (
    AccountSnapshot,
    BrokerError,
    MarketClosure,
    MarketStatus,
    OrderReceipt,
)
from trading_bot.history import HistoryStore
from trading_bot.main import (
    PersistedState,
    StateStore,
    TradingBot,
    _inject_benchmark,
    market_closed_message,
)
from trading_bot.risk.engine import Position, Side
from trading_bot.strategies.base import SignalType, Strategy
from trading_bot.strategies.relative_momentum import RelativeMomentumStrategy

NOW = datetime(2024, 1, 10, 15, 0, tzinfo=UTC)


class AlwaysBuyStrategy(Strategy):
    """Stratégie de test : signale BUY sur chaque barre, ATR constant."""

    name = "always_buy"

    @property
    def warmup_bars(self) -> int:
        return 2

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        out = bars.copy()
        out["ema_fast"] = out["close"]
        out["ema_slow"] = out["close"] * 0.99
        out["rsi"] = 55.0
        out["atr"] = 1.0
        return out

    def compute_signals(self, prepared: pd.DataFrame) -> pd.Series:
        return pd.Series(SignalType.BUY, index=prepared.index, dtype=object)


class AlwaysSellStrategy(AlwaysBuyStrategy):
    """Stratégie de test : signale SELL sur chaque barre."""

    name = "always_sell"

    def compute_signals(self, prepared: pd.DataFrame) -> pd.Series:
        return pd.Series(SignalType.SELL, index=prepared.index, dtype=object)


class FakeBroker:
    """Courtier simulé : compte, positions, prix et fills déterministes."""

    def __init__(
        self,
        *,
        equity: float = 100_000.0,
        positions: dict[str, Position] | None = None,
        prices: dict[str, float] | None = None,
        market_open: bool = True,
        market_closure: MarketClosure | None = None,
        fail_cancel: bool = False,
        fail_cancel_calls: set[int] | None = None,
        open_orders: dict[str, list[str]] | None = None,
    ) -> None:
        self.equity = equity
        self.cash = equity
        self.positions = positions or {}
        self.prices = prices or {"AAPL": 100.0}
        self.market_open = market_open
        self.market_closure = market_closure
        self.fail_cancel = fail_cancel
        self.fail_cancel_calls = fail_cancel_calls or set()
        self.cancel_calls = 0
        self.open_orders = open_orders or {}
        self.orders: list[object] = []
        self.receipts: list[OrderReceipt] = []

    def cancel_open_orders(self, symbol: str) -> int:
        self.cancel_calls += 1
        if self.fail_cancel or self.cancel_calls in self.fail_cancel_calls:
            raise BrokerError("annulation simulée impossible")
        orders = self.open_orders.pop(symbol, [])
        return len(orders)

    def is_market_open(self) -> bool:
        return self.market_open

    def market_status(self) -> MarketStatus:
        if self.market_open:
            return MarketStatus(is_open=True)
        return MarketStatus(is_open=False, closure=self.market_closure)

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(equity=self.equity, cash=self.cash, buying_power=self.cash)

    def get_positions(self) -> dict[str, Position]:
        return {
            symbol: replace(position, stop_price=None, initial_stop_price=None)
            for symbol, position in self.positions.items()
        }

    def get_latest_prices(self, symbols) -> dict[str, float]:
        return {symbol: self.prices[symbol] for symbol in symbols if symbol in self.prices}

    def submit(self, intent, *, now=None) -> OrderReceipt:
        self.orders.append(intent)
        receipt = OrderReceipt(
            order_id=f"order-{len(self.orders)}",
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            status="filled",
            filled_qty=intent.qty,
            avg_fill_price=self.prices.get(intent.symbol, 100.0),
            submitted_at=now,
        )
        self.receipts.append(receipt)
        return receipt

    def wait_for_fill(
        self, order_id: str, *, timeout_seconds: float = 30.0
    ) -> OrderReceipt:
        return self.receipts[-1]


class FakeBars:
    """Source de barres figée (une seule frame, ou une par symbole)."""

    def __init__(self, bars: pd.DataFrame | dict[str, pd.DataFrame]) -> None:
        self.bars = bars

    def fetch_bars(self, symbol, *, timeframe="1Day", lookback_days=400, now=None):
        if isinstance(self.bars, dict):
            return self.bars[symbol]
        return self.bars


class FakeNewsAnalyzer:
    """Analyseur d'actualités simulé, injecté dans le bot."""

    def __init__(self, assessments: dict[str, NewsAssessment]) -> None:
        self.assessments = assessments
        self.calls: list[list[str]] = []

    def assess(
        self, symbols, *, now=None
    ) -> dict[str, NewsAssessment]:
        self.calls.append(list(symbols))
        return {
            symbol: self.assessments[symbol]
            for symbol in symbols
            if symbol in self.assessments
        }


@pytest.fixture
def bot_bars(bars_factory) -> pd.DataFrame:
    """30 barres calmes se terminant la veille de la séance de test."""
    return bars_factory([100.0] * 30, start="2023-12-01")


def make_bot(
    settings: Settings,
    bars: pd.DataFrame,
    broker: FakeBroker,
    strategy: Strategy | None = None,
) -> TradingBot:
    return TradingBot(
        settings,
        strategy or AlwaysBuyStrategy(),
        broker,  # type: ignore[arg-type]
        bars=FakeBars(bars),  # type: ignore[arg-type]
        clock=lambda: NOW,
    )


def test_cycle_opens_position_and_persists_stop(settings, bot_bars) -> None:
    broker = FakeBroker()
    bot = make_bot(settings, bot_bars, broker)
    report = bot.run_once()
    assert not report.skipped
    assert report.errors == []
    assert len(broker.orders) == 2  # entrée + jambe de stop côté courtier
    intent = broker.orders[0]
    assert intent.side is Side.BUY
    assert intent.qty == pytest.approx(200.0)  # plafond 20 % de 100 000 à 100 $
    assert intent.stop_price == pytest.approx(98.0)
    stop_intent = broker.orders[1]
    assert stop_intent.side is Side.SELL
    assert stop_intent.order_type.value == "stop"
    assert stop_intent.stop_price == pytest.approx(98.0)
    assert stop_intent.qty == pytest.approx(200.0)

    state = StateStore(settings.state_dir / "state.json").load()
    assert state.stops["AAPL"]["stop_price"] == pytest.approx(98.0)
    assert state.trading_day == "2024-01-10"
    assert state.peak_equity == pytest.approx(100_000.0)

    audit_path = settings.state_dir / "audit.jsonl"
    assert audit_path.exists()
    events = [json.loads(line)["event"] for line in audit_path.read_text().splitlines()]
    assert "risk_decision" in events
    assert "order" in events


def test_cycle_is_recorded_in_history(settings, bot_bars) -> None:
    """Chaque cycle laisse une trace lisible par l'interface web."""
    broker = FakeBroker()
    bot = make_bot(settings, bot_bars, broker)
    bot.run_once(source="gui")

    entries = HistoryStore(settings.state_dir / "history.jsonl").entries()
    assert len(entries) == 1
    assert entries[0]["session_date"] == "2024-01-10"
    assert entries[0]["source"] == "gui"
    assert entries[0]["dry_run"] is False
    assert entries[0]["signals"] == {"AAPL": "BUY"}
    assert entries[0]["orders"][0]["side"] == "buy"
    assert entries[0]["errors"] == []
    assert entries[0]["cash"] == pytest.approx(100_000.0 - 200.0 * 100.0)
    assert entries[0]["positions_count"] == 1
    assert entries[0]["unrealized_pnl"] == pytest.approx(0.0)


def test_history_source_defaults_to_timer_under_systemd(
    settings, bot_bars, monkeypatch
) -> None:
    """Le service systemd pose INVOCATION_ID : le cycle est marqué « timer »."""
    monkeypatch.setenv("INVOCATION_ID", "abc123")
    make_bot(settings, bot_bars, FakeBroker()).run_once()

    entries = HistoryStore(settings.state_dir / "history.jsonl").entries()
    assert entries[0]["source"] == "timer"


def test_two_cycles_same_day_leave_one_history_entry(settings, bot_bars) -> None:
    """Répétition puis vraie passe le même jour : la dernière fait foi."""
    make_bot(settings, bot_bars, FakeBroker()).run_once(source="gui")
    make_bot(settings, bot_bars, FakeBroker()).run_once(source="gui")

    entries = HistoryStore(settings.state_dir / "history.jsonl").entries()
    assert len(entries) == 1


def test_no_position_reopened_after_restart(settings, bot_bars) -> None:
    broker = FakeBroker()
    make_bot(settings, bot_bars, broker).run_once()
    position = Position(
        symbol="AAPL", qty=200.0, avg_entry_price=100.0, current_price=100.0
    )
    restarted_broker = FakeBroker(positions={"AAPL": position})
    bot = make_bot(settings, bot_bars, restarted_broker)
    report = bot.run_once()
    assert report.errors == []
    assert restarted_broker.orders == []  # pas de doublon
    state = StateStore(settings.state_dir / "state.json").load()
    assert state.stops["AAPL"]["stop_price"] == pytest.approx(98.0)


def test_dry_run_never_transmits_order(settings, bot_bars) -> None:
    dry_settings = settings.model_copy(update={"dry_run": True})
    broker = FakeBroker()
    report = make_bot(dry_settings, bot_bars, broker).run_once()
    assert broker.orders == []
    assert any(decision.approved for decision in report.decisions)
    state = StateStore(dry_settings.state_dir / "state.json").load()
    assert "AAPL" not in state.stops


def make_news(score: float, *, critical: bool = False, symbol: str = "AAPL") -> NewsAssessment:
    """Verdict d'actualités prêt à l'emploi pour les tests de cycle."""
    return NewsAssessment(
        symbol=symbol,
        sentiment_score=score,
        confidence=0.8,
        is_critical_threat=critical,
        rationale="test",
        articles_count=2,
        updated_at=NOW,
    )


def test_dry_run_still_consults_the_news_filter(settings, bot_bars) -> None:
    """Une répétition ne transmet aucun ordre, mais consomme bien le filtre.

    Conséquence à connaître avant de cliquer « simulation » un jour de marché
    ouvert : le cycle appelle réellement OpenRouter et réécrit l'état partagé
    (``processed_news_ids.json`` et ``news_sentiment.json``). Ce n'est donc
    pas gratuit — mais cela ne touche ni aux positions ni aux ordres.
    """
    dry_settings = settings.model_copy(update={"dry_run": True})
    broker = FakeBroker()
    news = FakeNewsAnalyzer({"AAPL": make_news(0.10)})
    bot = TradingBot(
        dry_settings,
        AlwaysBuyStrategy(),
        broker,  # type: ignore[arg-type]
        bars=FakeBars(bot_bars),  # type: ignore[arg-type]
        clock=lambda: NOW,
        news_analyzer=news,  # type: ignore[arg-type]
    )
    report = bot.run_once()

    assert not report.skipped, "séance ouverte : le cycle va bien au bout"
    assert broker.orders == [], "une répétition ne transmet jamais d'ordre"
    assert news.calls, "mais elle interroge bien le filtre d'actualités"


def test_closed_market_stops_before_the_news_filter(settings, bot_bars) -> None:
    """Séance close : le cycle sort avant tout appel d'actualités.

    C'est ce qui rend une simulation inoffensive le week-end : ni appel
    facturé, ni réécriture de l'état des notes.
    """
    broker = FakeBroker(market_open=False)
    news = FakeNewsAnalyzer({"AAPL": make_news(-0.90, critical=True)})
    bot = TradingBot(
        settings,
        AlwaysBuyStrategy(),
        broker,  # type: ignore[arg-type]
        bars=FakeBars(bot_bars),  # type: ignore[arg-type]
        clock=lambda: NOW,
        news_analyzer=news,  # type: ignore[arg-type]
    )
    report = bot.run_once()

    assert report.skipped
    assert broker.orders == []
    assert news.calls == [], "aucun appel facturé quand la séance est close"


def test_market_closed_skips_cycle(settings, bot_bars) -> None:
    broker = FakeBroker(market_open=False, market_closure=MarketClosure.WEEKEND)
    report = make_bot(settings, bot_bars, broker).run_once()
    assert report.skipped
    assert report.skip_reason == "marché fermé (week-end)"
    assert broker.orders == []


def test_market_closed_message_names_reason_and_reopen(settings) -> None:
    """Le message nomme la cause réelle et le prochain instant d'ouverture."""
    status = MarketStatus(
        is_open=False,
        closure=MarketClosure.WEEKEND,
        next_open=datetime(2026, 11, 2, 14, 30, tzinfo=UTC),
    )
    assert market_closed_message(status, settings.market_timezone) == (
        "marché fermé (week-end) — réouverture lundi 02/11 à 09:30 (EST)"
    )


def test_market_closed_message_without_reopen(settings) -> None:
    """Sans horaire d'ouverture connu, le message reste sobre et honnête."""
    status = MarketStatus(is_open=False, closure=MarketClosure.HOLIDAY)
    assert (
        market_closed_message(status, settings.market_timezone)
        == "marché fermé (jour férié)"
    )


def test_daily_circuit_breaker_liquidates(settings, bot_bars) -> None:
    state = StateStore(settings.state_dir / "state.json")
    from trading_bot.main import PersistedState

    state.save(
        PersistedState(
            trading_day="2024-01-10",
            day_start_equity=100_000.0,
            peak_equity=100_000.0,
        )
    )
    position = Position(symbol="AAPL", qty=100.0, avg_entry_price=95.0, current_price=95.0)
    broker = FakeBroker(
        equity=95_000.0,
        positions={"AAPL": position},
        prices={"AAPL": 95.0},
    )
    report = make_bot(settings, bot_bars, broker).run_once()
    assert report.halts
    assert broker.orders
    assert broker.orders[0].side is Side.SELL
    persisted = state.load()
    assert persisted.halted is True
    assert persisted.halt_scope == "day"
    assert "AAPL" not in persisted.stops


def test_cancel_failure_blocks_market_exit(settings, bot_bars) -> None:
    position = Position(symbol="AAPL", qty=10.0, avg_entry_price=100.0, current_price=100.0)
    broker = FakeBroker(
        positions={"AAPL": position},
        fail_cancel=True,
        open_orders={"AAPL": ["protective-stop"]},
    )
    StateStore(settings.state_dir / "state.json").save(
        PersistedState(
            trading_day="2024-01-10",
            day_start_equity=100_000.0,
            peak_equity=100_000.0,
            stops={
                "AAPL": {
                    "stop_price": 90.0,
                    "initial_stop_price": 90.0,
                    "highest_price": 100.0,
                }
            },
        )
    )

    report = make_bot(settings, bot_bars, broker, AlwaysSellStrategy()).run_once()

    assert broker.orders == []
    assert "AAPL" in broker.positions
    assert report.errors
    assert any("annulation" in error for error in report.errors)


def test_successful_cancel_allows_exit_and_removes_stop(settings, bot_bars) -> None:
    position = Position(symbol="AAPL", qty=10.0, avg_entry_price=100.0, current_price=100.0)
    broker = FakeBroker(
        positions={"AAPL": position},
        open_orders={"AAPL": ["protective-stop"]},
    )
    StateStore(settings.state_dir / "state.json").save(
        PersistedState(
            trading_day="2024-01-10",
            day_start_equity=100_000.0,
            peak_equity=100_000.0,
            stops={
                "AAPL": {
                    "stop_price": 90.0,
                    "initial_stop_price": 90.0,
                    "highest_price": 100.0,
                }
            },
        )
    )

    report = make_bot(settings, bot_bars, broker, AlwaysSellStrategy()).run_once()

    assert len(broker.orders) == 1
    assert broker.orders[0].side is Side.SELL
    assert broker.open_orders.get("AAPL", []) == []
    assert report.errors == []


def test_successful_sell_with_residual_cancel_failure_is_reported(
    settings, bot_bars
) -> None:
    position = Position(symbol="AAPL", qty=10.0, avg_entry_price=100.0, current_price=100.0)
    broker = FakeBroker(
        positions={"AAPL": position},
        fail_cancel_calls={2},
        open_orders={"AAPL": ["protective-stop"]},
    )
    StateStore(settings.state_dir / "state.json").save(
        PersistedState(
            trading_day="2024-01-10",
            day_start_equity=100_000.0,
            peak_equity=100_000.0,
            stops={
                "AAPL": {
                    "stop_price": 90.0,
                    "initial_stop_price": 90.0,
                    "highest_price": 100.0,
                }
            },
        )
    )

    report = make_bot(settings, bot_bars, broker, AlwaysSellStrategy()).run_once()

    assert len(broker.orders) == 1
    assert broker.orders[0].side is Side.SELL
    assert any("annulation" in error for error in report.errors)
    audit = (settings.state_dir / "audit.jsonl").read_text(encoding="utf-8")
    assert "cancel_open_orders_failed" in audit


def test_stop_replacement_failure_does_not_submit_new_stop(settings, bot_bars) -> None:
    position = Position(symbol="AAPL", qty=10.0, avg_entry_price=100.0, current_price=100.0)
    broker = FakeBroker(
        positions={"AAPL": position},
        prices={"AAPL": 110.0},
        fail_cancel=True,
        open_orders={"AAPL": ["protective-stop"]},
    )
    StateStore(settings.state_dir / "state.json").save(
        PersistedState(
            trading_day="2024-01-10",
            day_start_equity=100_000.0,
            peak_equity=100_000.0,
            stops={
                "AAPL": {
                    "stop_price": 98.0,
                    "initial_stop_price": 98.0,
                    "highest_price": 100.0,
                }
            },
        )
    )

    report = make_bot(settings, bot_bars, broker).run_once()

    assert broker.orders == []
    assert any("annulation" in error for error in report.errors)
    assert broker.open_orders["AAPL"] == ["protective-stop"]


def test_stop_loss_exits_position(settings, bot_bars) -> None:
    from trading_bot.main import PersistedState

    StateStore(settings.state_dir / "state.json").save(
        PersistedState(
            trading_day="2024-01-10",
            day_start_equity=100_000.0,
            peak_equity=100_000.0,
            stops={"AAPL": {"stop_price": 98.0, "initial_stop_price": 98.0}},
        )
    )
    position = Position(
        symbol="AAPL",
        qty=100.0,
        avg_entry_price=100.0,
        current_price=95.0,
        stop_price=98.0,
        initial_stop_price=98.0,
    )
    broker = FakeBroker(positions={"AAPL": position}, prices={"AAPL": 95.0})
    report = make_bot(settings, bot_bars, broker).run_once()
    assert broker.orders
    assert broker.orders[0].side is Side.SELL
    assert any(decision.reason == "stop_loss" for decision in report.decisions)


def test_state_store_roundtrip_and_corruption(settings) -> None:
    store = StateStore(Path(settings.state_dir) / "state.json")
    from trading_bot.main import PersistedState

    store.save(
        PersistedState(
            trading_day="2024-01-10",
            day_start_equity=100_000.0,
            peak_equity=110_000.0,
            stops={"AAPL": {"stop_price": 98.0}},
        )
    )
    loaded = store.load()
    assert loaded.peak_equity == pytest.approx(110_000.0)
    assert loaded.stops["AAPL"]["stop_price"] == pytest.approx(98.0)
    assert not store.path.with_suffix(".json.tmp").exists()

    store.path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="illisible"):
        store.load()


def test_state_store_missing_file_returns_defaults(settings) -> None:
    store = StateStore(Path(settings.state_dir) / "absent.json")
    loaded = store.load()
    assert loaded.trading_day is None
    assert loaded.stops == {}


def test_unprotected_position_gets_atr_stop(settings, bot_bars) -> None:
    position = Position(symbol="AAPL", qty=10.0, avg_entry_price=100.0, current_price=100.0)
    broker = FakeBroker(positions={"AAPL": position})
    make_bot(settings, bot_bars, broker).run_once()
    state = StateStore(settings.state_dir / "state.json").load()
    assert state.stops["AAPL"]["stop_price"] == pytest.approx(98.0)


def test_build_strategy_selection() -> None:
    from trading_bot.main import STRATEGY_NAMES, build_strategy
    from trading_bot.strategies.donchian_breakout import DonchianBreakoutStrategy
    from trading_bot.strategies.ema_cross import EmaCrossStrategy
    from trading_bot.strategies.relative_momentum import RelativeMomentumStrategy

    assert isinstance(build_strategy("ema_cross"), EmaCrossStrategy)
    assert isinstance(build_strategy("donchian_breakout"), DonchianBreakoutStrategy)
    momentum = build_strategy("relative_momentum")
    assert isinstance(momentum, RelativeMomentumStrategy)
    assert momentum.benchmark_symbol == "SPY"
    assert set(STRATEGY_NAMES) == {"ema_cross", "donchian_breakout", "relative_momentum"}

    # Surcharges CLI réservées au backtest : ignorées ailleurs par défaut.
    namespace = argparse.Namespace(entry=30, exit_channel=15, sma_long=100)
    donchian = build_strategy("donchian_breakout", namespace)
    assert donchian.params.entry_length == 30
    assert donchian.params.exit_length == 15
    assert donchian.params.sma_length == 100

    with pytest.raises(Exception, match="stratégie inconnue"):
        build_strategy("unknown")


def test_ranking_favors_strongest_candidate_in_live(settings, bars_factory) -> None:
    def rising(n, rate=1.01):
        return 100.0 * np.array([rate**i * (1.0 + 0.002 * math.sin(i)) for i in range(n)])

    from trading_bot.config import RiskConfig

    ranked_settings = settings.model_copy(
        update={
            "symbols": ["AAPL", "MSFT"],
            "risk": RiskConfig(max_open_positions=1),
        }
    )
    bars = {
        "AAPL": bars_factory([100.0] * 130),  # momentum nul : score NaN
        "MSFT": bars_factory(list(rising(130))),  # momentum vigoureux
        # SPY au-dessus de sa SMA 200 : régime haussier, plafond 30 %.
        "SPY": bars_factory(list(rising(210)), start="2023-01-02"),
    }
    broker = FakeBroker(prices={"AAPL": 100.0, "MSFT": 350.0})
    bot = make_bot(ranked_settings, bars["AAPL"], broker)
    bot._bars = FakeBars(bars)
    report = bot.run_once()
    assert report.errors == []
    assert len(broker.orders) == 2  # entrée MSFT + jambe de stop, AAPL refusé
    entry = broker.orders[0]
    assert entry.symbol == "MSFT"
    # Plafond de position haussier : 30 000 $ / 350 $ (base : 20 000 $).
    assert entry.qty == pytest.approx(30_000.0 / 350.0, rel=1e-3)


def test_benchmark_label_follows_the_configuration(bot_bars) -> None:
    """L'étiquette du benchmark est injectée, pas figée dans la stratégie.

    Régression : « SPY » était codé en dur, si bien qu'un bot configuré sur
    une autre référence (QQQ) aurait continué d'annoncer SPY dans ses
    explications et son audit.
    """
    strategy = RelativeMomentumStrategy()
    assert strategy.benchmark_symbol == "SPY"  # valeur par défaut

    _inject_benchmark(strategy, bot_bars, "QQQ")

    assert strategy.benchmark_symbol == "QQQ"
    assert "QQQ" in strategy.explain(
        strategy.prepare(bot_bars), SignalType.HOLD
    )


def test_publish_stats_backfills_oldest_first(settings, monkeypatch) -> None:
    """La republication envoie les cycles du plus ancien au plus récent.

    L'ordre compte : le fichier distant se construit par ajouts successifs,
    donc publier le plus récent d'abord laisserait l'historique en désordre.
    """
    store = HistoryStore(settings.state_dir / "history.jsonl")
    store.record(
        {"session_date": "2024-01-08", "skipped": False, "equity": 100_000.0},
        source="timer",
    )
    store.record(
        {"session_date": "2024-01-09", "skipped": False, "equity": 101_000.0},
        source="gui",
    )

    sent: list[dict] = []

    class RecordingPublisher:
        def send(self, event: str, payload: dict) -> bool:
            sent.append({"event": event, "payload": payload})
            return True

    monkeypatch.setattr(trading_main, "_settings_from_args", lambda _args: settings)
    monkeypatch.setattr(
        trading_main, "_build_stats_publisher", lambda _settings: RecordingPublisher()
    )

    assert trading_main.cmd_publish_stats(argparse.Namespace()) == 0
    assert [call["payload"]["session_date"] for call in sent] == [
        "2024-01-08",
        "2024-01-09",
    ]
    assert sent[0]["event"] == "cycle_trading"
    # Les métadonnées de stockage local ne font pas partie du rapport publié.
    assert "recorded_at" not in sent[0]["payload"]
    assert "source" not in sent[0]["payload"]


def test_publish_stats_reports_missing_configuration(settings, monkeypatch) -> None:
    """Sans dépôt ni jeton, la commande échoue proprement plutôt que en silence."""
    monkeypatch.setattr(trading_main, "_settings_from_args", lambda _args: settings)
    monkeypatch.setattr(trading_main, "_build_stats_publisher", lambda _settings: None)
    assert trading_main.cmd_publish_stats(argparse.Namespace()) == 2


def test_cycle_refuses_a_buy_when_news_is_bad(settings, bot_bars) -> None:
    """Bout en bout : un veto d'actualités traverse tout le cycle.

    Le contrôle porte aussi sur ``audit.jsonl`` : un refus qui n'y figure pas
    est un refus invisible, donc indéfendable quand on relit l'historique.
    """
    broker = FakeBroker()
    news = FakeNewsAnalyzer(
        {
            "AAPL": NewsAssessment(
                symbol="AAPL",
                sentiment_score=-0.80,
                confidence=0.9,
                is_critical_threat=False,
                rationale="avertissement sur les résultats",
                articles_count=3,
                updated_at=NOW,
            )
        }
    )
    bot = TradingBot(
        settings,
        AlwaysBuyStrategy(),
        broker,  # type: ignore[arg-type]
        bars=FakeBars(bot_bars),  # type: ignore[arg-type]
        clock=lambda: NOW,
        news_analyzer=news,  # type: ignore[arg-type]
    )
    report = bot.run_once()

    assert news.calls, "le filtre d'actualités doit avoir été consulté"
    assert broker.orders == [], "aucun ordre ne doit être transmis"
    assert [decision.code.value for decision in report.decisions] == ["NEWS_VETO"]
    audit = (settings.state_dir / "audit.jsonl").read_text(encoding="utf-8")
    assert "NEWS_VETO" in audit


def test_cycle_liquidates_a_position_on_critical_news(settings, bot_bars) -> None:
    """Un krach signalé par les actualités sort la position sans attendre le stop."""
    position = Position(
        symbol="AAPL",
        qty=10.0,
        avg_entry_price=100.0,
        current_price=100.0,
        stop_price=80.0,
        initial_stop_price=80.0,
    )
    broker = FakeBroker(positions={"AAPL": position}, prices={"AAPL": 100.0})
    news = FakeNewsAnalyzer(
        {
            "AAPL": NewsAssessment(
                symbol="AAPL",
                sentiment_score=-0.95,
                confidence=0.95,
                is_critical_threat=True,
                rationale="fraude comptable révélée",
                articles_count=5,
                updated_at=NOW,
            )
        }
    )
    bot = TradingBot(
        settings,
        AlwaysBuyStrategy(),
        broker,  # type: ignore[arg-type]
        bars=FakeBars(bot_bars),  # type: ignore[arg-type]
        clock=lambda: NOW,
        news_analyzer=news,  # type: ignore[arg-type]
    )
    report = bot.run_once()

    assert "news_panic_exit" in [decision.reason for decision in report.decisions]
    # ``FakeBroker.orders`` est typé ``list[object]`` : on lit le sens sans
    # présumer la classe, comme le fait déjà le reste du fichier.
    sides = [getattr(order, "side", None) for order in broker.orders]
    assert Side.SELL in sides


def test_cycle_ignores_the_filter_when_it_is_absent(settings, bot_bars) -> None:
    """Sans analyseur fourni, le cycle se comporte exactement comme avant."""
    broker = FakeBroker()
    bot = make_bot(settings, bot_bars, broker)
    report = bot.run_once()
    assert not report.skipped
    assert [decision.code.value for decision in report.decisions] == ["APPROVED"]


def test_cycle_sends_webhook_report(settings, bot_bars) -> None:
    class RecordingNotifier:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict]] = []

        def send(self, event: str, payload: dict) -> bool:
            self.events.append((event, payload))
            return True

    notifier = RecordingNotifier()
    broker = FakeBroker()
    bot = make_bot(settings, bot_bars, broker)
    bot._notifier = notifier
    report = bot.run_once()
    assert not report.skipped
    assert len(notifier.events) == 1
    event, payload = notifier.events[0]
    assert event == "cycle_trading"
    assert payload["equity"] == pytest.approx(report.equity)
    assert payload["signals"] == {"AAPL": "BUY"}

    # Cycle ignoré (marché fermé) : aucune notification, pas de spam.
    closed_broker = FakeBroker(market_open=False)
    bot._broker = closed_broker
    bot.run_once()
    assert len(notifier.events) == 1
