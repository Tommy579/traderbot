"""Tests du contrat de stratégie et de la stratégie EMA/RSI."""

from __future__ import annotations

import math
from datetime import UTC

import numpy as np
import pandas as pd
import pytest

from trading_bot.strategies.base import Signal, SignalType, StrategyError
from trading_bot.strategies.donchian_breakout import (
    DonchianBreakoutStrategy,
    DonchianParams,
)
from trading_bot.strategies.ema_cross import EmaCrossParams, EmaCrossStrategy
from trading_bot.strategies.relative_momentum import (
    RelativeMomentumParams,
    RelativeMomentumStrategy,
)

WIDE_RSI = EmaCrossParams(rsi_entry_min=0.0, rsi_entry_max=100.0)


def test_signal_type_values() -> None:
    assert {member.value for member in SignalType} == {"BUY", "SELL", "HOLD"}


def test_signal_rejects_invalid_price() -> None:
    with pytest.raises(ValueError):
        Signal(
            symbol="AAPL",
            action=SignalType.BUY,
            as_of=pd.Timestamp("2024-01-02", tz="UTC"),
            price=0.0,
        )


def test_signal_rejects_invalid_strength_and_naive_timestamp() -> None:
    with pytest.raises(ValueError):
        Signal(
            symbol="AAPL",
            action=SignalType.BUY,
            as_of=pd.Timestamp("2024-01-02", tz="UTC"),
            price=100.0,
            strength=1.5,
        )
    with pytest.raises(ValueError):
        Signal(
            symbol="AAPL",
            action=SignalType.HOLD,
            as_of=pd.Timestamp("2024-01-02"),
            price=100.0,
        )


def test_warmup_bars_are_hold(wave_bars) -> None:
    strategy = EmaCrossStrategy(WIDE_RSI)
    enriched = strategy.generate_signals(wave_bars)
    assert (enriched["signal"].iloc[: strategy.warmup_bars] == SignalType.HOLD).all()


def test_strategy_produces_buy_and_sell(wave_bars) -> None:
    strategy = EmaCrossStrategy(WIDE_RSI)
    enriched = strategy.generate_signals(wave_bars, symbol="TEST")
    assert (enriched["signal"] == SignalType.BUY).any()
    assert (enriched["signal"] == SignalType.SELL).any()
    assert set(enriched["signal"].unique()) <= set(SignalType)


def test_default_rsi_filter_blocks_overbought_entries(wave_bars) -> None:
    strategy = EmaCrossStrategy()  # RSI 50-70
    enriched = strategy.generate_signals(wave_bars)
    assert not (enriched["signal"] == SignalType.BUY).any()


def test_latest_signal_matches_last_bar(wave_bars) -> None:
    strategy = EmaCrossStrategy(WIDE_RSI)
    signal = strategy.latest_signal(wave_bars, symbol="TEST")
    assert signal.symbol == "TEST"
    assert signal.as_of == wave_bars.index[-1]
    assert signal.price == pytest.approx(float(wave_bars["close"].iloc[-1]))
    assert 0.0 <= signal.strength <= 1.0
    assert "atr" in signal.indicators
    assert signal.indicators["atr"] > 0


def test_no_lookahead_in_signal_generation(wave_bars) -> None:
    strategy = EmaCrossStrategy(WIDE_RSI)
    full = strategy.generate_signals(wave_bars)
    truncated = strategy.generate_signals(wave_bars.iloc[:150])
    pd.testing.assert_series_equal(
        full["signal"].iloc[:150], truncated["signal"], check_names=False
    )


def test_insufficient_history_rejected(wave_bars) -> None:
    strategy = EmaCrossStrategy()
    with pytest.raises(StrategyError):
        strategy.generate_signals(wave_bars.iloc[: strategy.warmup_bars])


def test_missing_columns_rejected(wave_bars) -> None:
    strategy = EmaCrossStrategy()
    with pytest.raises(StrategyError, match="colonnes manquantes"):
        strategy.generate_signals(wave_bars.drop(columns=["volume"]))


def test_naive_index_rejected(wave_bars) -> None:
    strategy = EmaCrossStrategy()
    naive = wave_bars.copy()
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(StrategyError, match="timezone"):
        strategy.generate_signals(naive)


def test_unsorted_index_rejected(wave_bars) -> None:
    strategy = EmaCrossStrategy()
    with pytest.raises(StrategyError, match="triées"):
        strategy.generate_signals(wave_bars.iloc[::-1])


def test_params_reject_inverted_periods() -> None:
    with pytest.raises(ValueError):
        EmaCrossParams(fast=50, slow=20)
    with pytest.raises(ValueError):
        EmaCrossParams(rsi_entry_min=80, rsi_entry_max=20)


def test_describe_exposes_parameters() -> None:
    description = EmaCrossStrategy().describe()
    assert description["strategy"] == "ema_cross"
    assert int(description["warmup_bars"]) > 50


def test_conviction_is_bounded(wave_bars) -> None:
    strategy = EmaCrossStrategy(WIDE_RSI)
    enriched = strategy.generate_signals(wave_bars)
    actions = [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
    for action in actions:
        value = strategy.conviction(enriched, action)
        assert 0.0 <= value <= 1.0
    assert strategy.conviction(enriched, SignalType.HOLD) == 0.0


def test_as_of_is_utc(wave_bars) -> None:
    signal = EmaCrossStrategy(WIDE_RSI).latest_signal(wave_bars, symbol="TEST")
    assert signal.as_of.tzinfo is not None
    assert signal.as_of.tzinfo.utcoffset(signal.as_of) == UTC.utcoffset(None)


# ---------------------------------------------------------------------------
# Donchian Breakout
# ---------------------------------------------------------------------------


def test_donchian_buy_on_breakout_with_trend(bars_factory) -> None:
    closes = [100.0] * 209 + [105.0]
    bars = bars_factory(closes)
    strategy = DonchianBreakoutStrategy()
    signal = strategy.latest_signal(bars, symbol="TEST")
    assert signal.action is SignalType.BUY
    assert "atr" in signal.indicators and signal.indicators["atr"] > 0
    assert "sma_long" in signal.indicators
    assert 0.0 <= signal.strength <= 1.0


def test_donchian_blocks_breakout_below_sma(bars_factory) -> None:
    # Cassure du canal haut mais clôture sous la SMA 200 : pas d'achat.
    closes = [100.0] * 150 + [80.0] * 59 + [81.0]
    bars = bars_factory(closes)
    signal = DonchianBreakoutStrategy().latest_signal(bars, symbol="TEST")
    assert signal.action is SignalType.HOLD


def test_donchian_sell_on_lower_channel_break(bars_factory) -> None:
    # Chute plus raide que le spread : chaque clôture casse le canal bas.
    closes = [100.0] * 180 + list(np.linspace(100.0, 78.0, 30))
    bars = bars_factory(closes)
    signal = DonchianBreakoutStrategy().latest_signal(bars, symbol="TEST")
    assert signal.action is SignalType.SELL


def test_donchian_warmup_is_hold_then_signals(wave_bars) -> None:
    strategy = DonchianBreakoutStrategy()
    assert strategy.warmup_bars == 201
    with pytest.raises(StrategyError):
        strategy.generate_signals(wave_bars.iloc[: strategy.warmup_bars])
    enriched = strategy.generate_signals(wave_bars)
    assert (enriched["signal"].iloc[: strategy.warmup_bars] == SignalType.HOLD).all()


def test_donchian_params_validation() -> None:
    with pytest.raises(ValueError):
        DonchianParams(entry_length=10, exit_length=20)
    assert DonchianBreakoutStrategy().describe()["strategy"] == "donchian_breakout"


def test_donchian_no_lookahead(bars_factory) -> None:
    closes = list(np.linspace(100.0, 90.0, 100)) + list(np.linspace(90.0, 125.0, 150))
    closes += list(np.linspace(125.0, 95.0, 150))
    bars = bars_factory(closes)
    strategy = DonchianBreakoutStrategy()
    full = strategy.generate_signals(bars)
    truncated = strategy.generate_signals(bars.iloc[:300])
    pd.testing.assert_series_equal(
        full["signal"].iloc[:300], truncated["signal"], check_names=False
    )


# ---------------------------------------------------------------------------
# Momentum relatif (benchmark SPY injecté)
# ---------------------------------------------------------------------------


def _rising_closes(n: int, rate: float, base: float = 100.0) -> list[float]:
    """Série en croissance régulière avec un peu de bruit (vol > 0)."""
    return [base * (rate**i) * (1.0 + 0.002 * math.sin(i)) for i in range(n)]


def test_momentum_buy_when_stronger_than_benchmark(bars_factory) -> None:
    symbol_bars = bars_factory(_rising_closes(150, 1.01))
    benchmark_bars = bars_factory(_rising_closes(150, 1.001))
    strategy = RelativeMomentumStrategy(benchmark=benchmark_bars)
    signal = strategy.latest_signal(symbol_bars, symbol="TEST")
    assert signal.action is SignalType.BUY
    assert signal.indicators["mom_fast"] > signal.indicators["benchmark_mom_fast"]
    assert signal.indicators["mom_slow"] > signal.indicators["benchmark_mom_slow"]
    assert signal.indicators["atr"] > 0


def test_momentum_sell_when_weaker_than_benchmark(bars_factory) -> None:
    symbol_bars = bars_factory(_rising_closes(150, 1.001))
    benchmark_bars = bars_factory(_rising_closes(150, 1.01))
    strategy = RelativeMomentumStrategy(benchmark=benchmark_bars)
    signal = strategy.latest_signal(symbol_bars, symbol="TEST")
    assert signal.action is SignalType.SELL


def test_momentum_requires_benchmark(bars_factory) -> None:
    strategy = RelativeMomentumStrategy()
    with pytest.raises(StrategyError, match="benchmark"):
        strategy.generate_signals(bars_factory(_rising_closes(150, 1.01)))


def test_momentum_rejects_invalid_benchmark(bars_factory) -> None:
    strategy = RelativeMomentumStrategy()
    benchmark = bars_factory(_rising_closes(150, 1.001))
    benchmark.index = benchmark.index.tz_localize(None)
    with pytest.raises(StrategyError, match="timezone-aware"):
        strategy.set_benchmark(benchmark)
    with pytest.raises(StrategyError, match="vide"):
        strategy.set_benchmark(benchmark.iloc[:0])


def test_momentum_warmup_is_hold(bars_factory) -> None:
    strategy = RelativeMomentumStrategy(benchmark=bars_factory(_rising_closes(150, 1.001)))
    assert strategy.warmup_bars == 122
    bars = bars_factory(_rising_closes(150, 1.01))
    with pytest.raises(StrategyError):
        strategy.generate_signals(bars.iloc[: strategy.warmup_bars])
    enriched = strategy.generate_signals(bars)
    assert (enriched["signal"].iloc[: strategy.warmup_bars] == SignalType.HOLD).all()


def test_momentum_no_lookahead(bars_factory) -> None:
    strategy = RelativeMomentumStrategy(benchmark=bars_factory(_rising_closes(150, 1.001)))
    bars = bars_factory(_rising_closes(150, 1.01))
    full = strategy.generate_signals(bars)
    truncated = strategy.generate_signals(bars.iloc[:140])
    pd.testing.assert_series_equal(
        full["signal"].iloc[:140], truncated["signal"], check_names=False
    )


def test_momentum_params_validation() -> None:
    with pytest.raises(ValueError):
        RelativeMomentumParams(fast_mom=120, slow_mom=60)
    assert RelativeMomentumStrategy().benchmark_symbol == "SPY"
    assert RelativeMomentumStrategy().describe()["strategy"] == "relative_momentum"
