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
from trading_bot.strategies.shield_sword import (
    ShieldSwordParams,
    ShieldSwordStrategy,
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


# ---------------------------------------------------------------------------
# Bouclier + Épée (Shield + Sword / Core-Satellite CPPI)
# ---------------------------------------------------------------------------


def test_shield_sword_params_validation() -> None:
    with pytest.raises(ValueError, match="fast_mom"):
        ShieldSwordParams(fast_mom=120, slow_mom=60)
    with pytest.raises(ValueError, match="100 %"):
        ShieldSwordParams(shield_allocation_pct=0.90, sword_allocation_pct=0.20)

    params = ShieldSwordParams()
    assert params.shield_allocation_pct == 0.70
    assert params.sword_allocation_pct == 0.30
    assert params.sword_trend_sma == 100
    assert params.sword_stop_atr_multiplier == 5.0
    assert params.sword_max_loss_pct == 0.12
    assert params.shield_symbol == "SGOV"

    strategy = ShieldSwordStrategy()
    assert strategy.describe()["strategy"] == "shield_sword"
    assert strategy.warmup_bars == 201


def _month_start_indices(bars: pd.DataFrame) -> list[int]:
    """Indices des premiers jours de bourse de chaque mois présents dans les barres."""
    index = bars.index
    return [i for i in range(1, len(index)) if index[i].month != index[i - 1].month]


def _first_rebalance_index(strategy: ShieldSwordStrategy, bars: pd.DataFrame) -> int:
    """Premier rebalancement exploitable : première ouverture de mois après le warm-up."""
    for position in _month_start_indices(bars):
        if position >= strategy.warmup_bars:
            return position
    raise AssertionError("aucun début de mois après le warm-up dans ces barres")


def test_shield_sword_rebalances_on_first_session_of_month(bars_factory) -> None:
    """Le rebalancement tombe le premier jour de bourse du mois, jamais à un offset de fenêtre."""
    n = 420
    bars = bars_factory(_rising_closes(n, 1.005), start="2023-01-02")
    bench = bars_factory(_rising_closes(n, 1.002), start="2023-01-02")

    strategy = ShieldSwordStrategy(
        ShieldSwordParams(sword_rebalance_days=20, sword_trend_sma=100),
        benchmark=bench,
    )
    signals = strategy.generate_signals(bars, symbol="OFFENSIVE")["signal"]

    month_starts = set(_month_start_indices(bars))
    active = [i for i, value in enumerate(signals) if value is not SignalType.HOLD]

    assert active, "aucun rebalancement : la stratégie est inerte"
    assert set(active) <= month_starts, "un rebalancement tombe hors d'un début de mois"
    # Après le warm-up, chaque ouverture de mois est un rebalancement.
    expected = {i for i in month_starts if i >= strategy.warmup_bars}
    assert set(active) == expected


def test_shield_sword_signals_are_window_independent(bars_factory) -> None:
    """Fenêtre glissante : les dates de rebalancement ne se décalent pas d'un cycle à l'autre."""
    n = 500
    bars = bars_factory(_rising_closes(n, 1.005), start="2023-01-02")
    bench = bars_factory(_rising_closes(n, 1.002), start="2023-01-02")

    def active_dates(window_start: int, window_end: int) -> set[pd.Timestamp]:
        sliced = bars.iloc[window_start:window_end]
        sliced_bench = bench.iloc[window_start:window_end]
        strategy = ShieldSwordStrategy(
            ShieldSwordParams(sword_rebalance_days=20, sword_trend_sma=100),
            benchmark=sliced_bench,
        )
        signals = strategy.generate_signals(sliced, symbol="OFFENSIVE")["signal"]
        return {stamp for stamp, value in signals.items() if value is not SignalType.HOLD}

    wide = active_dates(0, 400)
    slid = active_dates(5, 400)  # même fin, la fenêtre a glissé de 5 séances
    assert wide, "aucun rebalancement détecté sur la fenêtre de référence"
    assert wide == slid, "les dates de rebalancement dépendent du début de la fenêtre"


def test_shield_sword_buys_on_every_rebalance_while_selected(bars_factory) -> None:
    """Aucun état interne « déjà détenu » : chaque rebalancement réémet BUY."""
    n = 420
    bars = bars_factory(_rising_closes(n, 1.006), start="2023-01-02")
    bench = bars_factory(_rising_closes(n, 1.002), start="2023-01-02")

    strategy = ShieldSwordStrategy(
        ShieldSwordParams(sword_rebalance_days=20, sword_trend_sma=100),
        benchmark=bench,
    )
    signals = strategy.generate_signals(bars, symbol="OFFENSIVE")["signal"]

    buys = [i for i, value in enumerate(signals) if value is SignalType.BUY]
    assert len(buys) >= 2, "un actif en tendance doit être réacheté à chaque rebalancement"
    assert set(buys) <= set(_month_start_indices(bars))


def test_shield_sword_shield_symbols_ignored(bars_factory) -> None:
    n = 250
    bars = bars_factory(_rising_closes(n, 1.01))
    bench = bars_factory(_rising_closes(n, 1.002))
    strategy = ShieldSwordStrategy(benchmark=bench)
    for sym in ("BRK.B", "WMT", "KO", "SGOV"):
        enriched = strategy.generate_signals(bars, symbol=sym)
        assert (enriched["signal"] == SignalType.HOLD).all()


def test_shield_sword_top_n_selection(bars_factory) -> None:
    n = 250
    bench = bars_factory(_rising_closes(n, 1.002))

    # 4 offensive stocks with different growth rates
    bars_fast1 = bars_factory(_rising_closes(n, 1.012))
    bars_fast2 = bars_factory(_rising_closes(n, 1.010))
    bars_mid = bars_factory(_rising_closes(n, 1.004))
    bars_slow = bars_factory(_rising_closes(n, 1.001))

    universe = {
        "FAST1": bars_fast1,
        "FAST2": bars_fast2,
        "MID": bars_mid,
        "SLOW": bars_slow,
    }

    strategy = ShieldSwordStrategy(
        ShieldSwordParams(sword_top_n=2, sword_rebalance_days=20, sword_trend_sma=100),
        benchmark=bench,
    )
    strategy.set_universe(universe)

    rebal_idx = _first_rebalance_index(strategy, bars_fast1)
    sig_fast1 = strategy.generate_signals(bars_fast1, symbol="FAST1")["signal"].iloc[
        rebal_idx
    ]
    sig_fast2 = strategy.generate_signals(bars_fast2, symbol="FAST2")["signal"].iloc[
        rebal_idx
    ]
    sig_mid = strategy.generate_signals(bars_mid, symbol="MID")["signal"].iloc[rebal_idx]
    sig_slow = strategy.generate_signals(bars_slow, symbol="SLOW")["signal"].iloc[rebal_idx]

    # Top 2 (FAST1 and FAST2) are bought
    assert sig_fast1 == SignalType.BUY
    assert sig_fast2 == SignalType.BUY
    # MID and SLOW were not in Top 2
    assert sig_mid is not SignalType.BUY
    assert sig_slow is not SignalType.BUY


def test_shield_sword_bear_market_100pct_shield(bars_factory) -> None:
    n = 250
    # SPY falling below SMA 200
    bench_closes = list(np.linspace(120.0, 100.0, 250))
    bench = bars_factory(bench_closes)

    bars_fast = bars_factory(_rising_closes(n, 1.010))
    universe = {"FAST": bars_fast}

    strategy = ShieldSwordStrategy(benchmark=bench)
    strategy.set_universe(universe)

    rebal_idx = _first_rebalance_index(strategy, bars_fast)
    sig = strategy.generate_signals(bars_fast, symbol="FAST")["signal"].iloc[rebal_idx]
    # Because SPY < SMA(200), no offensive buy
    assert sig is not SignalType.BUY


def test_shield_immunity_from_stops_and_take_profit() -> None:
    from datetime import date, datetime

    from trading_bot.risk.engine import AccountState, Position, RiskConfig, RiskEngine

    engine = RiskEngine(RiskConfig())
    now = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)
    session = date(2026, 1, 5)
    account = AccountState(equity=100_000.0, cash=100_000.0, buying_power=100_000.0)

    # Entry evaluation for SGOV as shield asset: no stop price required
    decision = engine.evaluate_entry(
        symbol="SGOV",
        price=100.0,
        atr=0.10,
        account=account,
        now=now,
        session_date=session,
        is_shield=True,
        target_weight=0.70,
    )
    assert decision.approved
    assert decision.intent is not None
    assert decision.intent.stop_price is None

    # Open position in SGOV: trailing stop and take profit must be None
    pos_sgov = Position(
        symbol="SGOV", qty=700.0, avg_entry_price=100.0, current_price=125.0
    )
    assert engine.update_trailing_stop(pos_sgov, price=125.0, atr=0.10) is None
    assert engine.evaluate_partial_take_profit(pos_sgov, price=125.0, now=now) is None


def test_shield_symbol_explanation_names_the_shield(bars_factory) -> None:
    """Le diagnostic doit dire « Bouclier », pas « conservation entre rebalancements »."""
    n = 260
    bars = bars_factory(_rising_closes(n, 1.001), start="2023-01-02")
    bench = bars_factory(_rising_closes(n, 1.002), start="2023-01-02")
    strategy = ShieldSwordStrategy(ShieldSwordParams(shield_symbol="SHV"), benchmark=bench)

    signal = strategy.latest_signal(bars, symbol="SHV")

    assert signal.action is SignalType.HOLD
    assert "actif défensif" in signal.reason
    assert "rebalancement" not in signal.reason
