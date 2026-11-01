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

from trading_bot.config import ConfigurationError, Settings
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
