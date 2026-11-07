"""Tests du moteur de backtest : déterminisme, frais, stops, breakers."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtest.runner import BacktestError, BacktestRunner, Trade
from trading_bot.config import BacktestConfig, RiskConfig
from trading_bot.strategies.base import SignalType, Strategy
from trading_bot.strategies.donchian_breakout import DonchianBreakoutStrategy
from trading_bot.strategies.ema_cross import EmaCrossParams, EmaCrossStrategy
from trading_bot.strategies.relative_momentum import RelativeMomentumStrategy

WIDE_RSI = EmaCrossParams(rsi_entry_min=0.0, rsi_entry_max=100.0)
FAST_RISK = RiskConfig(
    max_daily_loss_pct=0.10,
    max_drawdown_pct=0.50,
)


class BuyOnceStrategy(Strategy):
    """Stratégie déterministe : un seul BUY, puis HOLD (test des stops)."""

    name = "buy_once"

    def __init__(self, buy_index: int = 5, atr: float = 1.0) -> None:
        self._buy_index = buy_index
        self._atr = atr

    @property
    def warmup_bars(self) -> int:
        return 2

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        out = bars.copy()
        out["ema_fast"] = out["close"]
        out["ema_slow"] = out["close"]
        out["rsi"] = 50.0
        out["atr"] = self._atr
        return out

    def compute_signals(self, prepared: pd.DataFrame) -> pd.Series:
        signals = pd.Series(SignalType.HOLD, index=prepared.index, dtype=object)
        if self._buy_index < len(signals):
            signals.iloc[self._buy_index] = SignalType.BUY
        return signals


def _runner(
    *,
    initial_equity: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.001,
    risk: RiskConfig | None = None,
) -> BacktestRunner:
    return BacktestRunner(
        EmaCrossStrategy(WIDE_RSI),
        config=BacktestConfig(
            initial_equity=initial_equity,
            commission_pct=commission,
            slippage_pct=slippage,
        ),
        risk_config=risk or FAST_RISK,
    )


def test_backtest_produces_trades_and_metrics(wave_bars) -> None:
    result = _runner().run({"TEST": wave_bars})
    assert result.metrics["trades"] == len(result.trades)
    assert len(result.trades) > 0
    assert result.metrics["final_equity"] == pytest.approx(
        result.equity_curve["equity"].iloc[-1]
    )
    assert result.metrics["total_fees"] == pytest.approx(
        sum(trade.fees for trade in result.trades)
    )
    assert result.equity_curve["equity"].gt(0).all()
    assert (result.equity_curve["drawdown_pct"] <= 1e-12).all()
    for key in (
        "total_return",
        "cagr",
        "sharpe",
        "sortino",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "exposure_pct",
        "halt_events",
        "benchmark_buy_hold",
        "basket_buy_hold",
    ):
        assert key in result.metrics


def test_benchmark_buy_hold_is_the_index_not_the_basket(bars_factory, wave_bars) -> None:
    """« buy & hold (réf.) » doit être l'indice, pas le panier équipondéré.

    Régression : la métrique servait la moyenne des buy & hold des actifs
    tradés, ce qui n'a rien à voir avec la référence de marché et ne permet
    donc pas de dire si la stratégie a battu le marché.
    """
    benchmark = bars_factory([100.0 + i for i in range(80)])
    result = _runner().run({"TEST": wave_bars}, benchmark=benchmark)

    expected_index = float(benchmark["close"].iloc[-1] / benchmark["close"].iloc[0] - 1.0)
    expected_basket = float(wave_bars["close"].iloc[-1] / wave_bars["close"].iloc[0] - 1.0)

    assert result.metrics["benchmark_buy_hold"] == pytest.approx(expected_index)
    assert result.metrics["basket_buy_hold"] == pytest.approx(expected_basket)
    assert result.metrics["benchmark_buy_hold"] != pytest.approx(expected_basket)


def test_benchmark_buy_hold_is_none_without_benchmark(wave_bars) -> None:
    """Sans référence fournie, la métrique reste indéterminée (jamais inventée)."""
    result = _runner().run({"TEST": wave_bars})
    assert result.metrics["benchmark_buy_hold"] is None
    assert result.metrics["basket_buy_hold"] is not None


def test_trade_pnl_matches_execution_prices(wave_bars) -> None:
    result = _runner(commission=0.002, slippage=0.002).run({"TEST": wave_bars})
    assert result.trades
    for trade in result.trades:
        expected = (trade.exit_price - trade.entry_price) * trade.qty - trade.fees
        assert trade.pnl == pytest.approx(expected)
        assert trade.fees > 0
        assert trade.exit_time > trade.entry_time
        assert trade.bars_held >= 1


def test_backtest_is_deterministic(wave_bars) -> None:
    runner = _runner()
    first = runner.run({"TEST": wave_bars})
    second = runner.run({"TEST": wave_bars})
    pd.testing.assert_frame_equal(first.equity_curve, second.equity_curve)
    assert first.trades == second.trades
    assert dict(first.metrics) == dict(second.metrics)


def test_commissions_reduce_performance(wave_bars) -> None:
    without_fees = _runner(commission=0.0, slippage=0.0).run({"TEST": wave_bars})
    with_fees = _runner(commission=0.005, slippage=0.005).run({"TEST": wave_bars})
    assert without_fees.metrics["final_equity"] >= with_fees.metrics["final_equity"]
    assert with_fees.metrics["total_fees"] > 0


def test_stop_loss_exit_on_crash(bars_factory) -> None:
    bars = bars_factory([100.0] * 15 + [95.0, 90.0, 85.0, 80.0], spread=0.4)
    runner = BacktestRunner(
        BuyOnceStrategy(buy_index=5, atr=1.0),
        config=BacktestConfig(initial_equity=100_000.0),
        risk_config=FAST_RISK,
    )
    result = runner.run({"TEST": bars})
    assert result.trades
    trade = result.trades[0]
    assert trade.exit_reason in {"stop_loss", "stop_gap"}
    assert trade.pnl < 0


def test_daily_circuit_breaker_liquidates_at_market(bars_factory) -> None:
    bars = bars_factory([100.0] * 15 + [90.0, 85.0], spread=0.2)
    risk = RiskConfig(
        risk_per_trade_pct=0.05,
        max_position_pct=0.50,
        max_gross_exposure_pct=1.0,
        atr_stop_multiple=10.0,
        trailing_atr_multiple=0.0,
        max_daily_loss_pct=0.005,
        max_drawdown_pct=0.50,
    )
    runner = BacktestRunner(
        BuyOnceStrategy(buy_index=5, atr=8.0),
        config=BacktestConfig(initial_equity=100_000.0),
        risk_config=risk,
    )
    result = runner.run({"TEST": bars})
    assert result.trades
    assert result.metrics["halt_events"] == 1
    assert result.trades[0].exit_reason == "circuit_breaker"
    assert bool(result.equity_curve["halted"].any())


def test_signals_are_executed_on_next_bar(wave_bars) -> None:
    strategy = EmaCrossStrategy(WIDE_RSI)
    enriched = strategy.generate_signals(wave_bars, symbol="TEST")
    buy_times = enriched.index[enriched["signal"] == SignalType.BUY]
    assert len(buy_times) > 0
    result = _runner().run({"TEST": wave_bars})
    assert result.trades
    first_entry = min(trade.entry_time for trade in result.trades)
    first_signal = buy_times[0]
    assert first_entry > first_signal
    assert first_entry == wave_bars.index[wave_bars.index.get_loc(first_signal) + 1]


def test_no_buy_on_entry_bar(wave_bars) -> None:
    strategy = EmaCrossStrategy(WIDE_RSI)
    enriched = strategy.generate_signals(wave_bars, symbol="TEST")
    result = _runner().run({"TEST": wave_bars})
    for trade in result.trades:
        assert enriched.loc[trade.entry_time, "signal"] is not SignalType.BUY


def test_multi_symbol_portfolio(wave_bars, bars_factory) -> None:
    other = bars_factory(wave_bars["close"].to_numpy() * 1.2)
    result = _runner().run({"AAA": wave_bars, "BBB": other})
    assert set(result.symbols) == {"AAA", "BBB"}
    assert result.metrics["trades"] > 0
    assert all(trade.symbol in {"AAA", "BBB"} for trade in result.trades)


def test_empty_data_rejected() -> None:
    with pytest.raises(BacktestError):
        _runner().run({})


def test_too_few_bars_rejected(bars_factory) -> None:
    with pytest.raises(BacktestError):
        _runner().run({"TEST": bars_factory([100.0, 101.0])})


def test_naive_index_rejected(wave_bars) -> None:
    naive = wave_bars.copy()
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(BacktestError):
        _runner().run({"TEST": naive})


def test_result_is_json_serializable(wave_bars) -> None:
    result = _runner().run({"TEST": wave_bars})
    payload = json.dumps(result.to_dict(), ensure_ascii=False)
    assert "metrics" in payload
    assert "equity_curve" in payload
    parsed = json.loads(payload)
    assert parsed["strategy"] == "ema_cross"


def test_trades_frame_shape(wave_bars) -> None:
    result = _runner().run({"TEST": wave_bars})
    frame = result.trades_frame()
    assert len(frame) == len(result.trades)
    assert list(frame.columns) == [
        "symbol",
        "qty",
        "entry_time",
        "entry_price",
        "exit_time",
        "exit_price",
        "fees",
        "pnl",
        "return_pct",
        "exit_reason",
        "bars_held",
    ]


def test_empty_trades_frame_has_columns(bars_factory) -> None:
    flat = bars_factory([100.0] * 120)
    result = _runner().run({"TEST": flat})
    assert result.trades_frame().empty
    assert "pnl" in result.trades_frame().columns


def test_trade_is_win_property() -> None:
    timestamp = pd.Timestamp("2024-01-02", tz="UTC")
    winner = Trade(
        symbol="TEST",
        qty=1,
        entry_time=timestamp,
        entry_price=100.0,
        exit_time=timestamp,
        exit_price=110.0,
        fees=1.0,
        pnl=9.0,
        return_pct=0.09,
        exit_reason="strategy_exit",
        bars_held=3,
    )
    assert winner.is_win


def test_donchian_backtest_opens_position_on_trend(bars_factory) -> None:
    closes = 100.0 * np.array([1.005**i for i in range(230)])
    runner = BacktestRunner(DonchianBreakoutStrategy(), config=BacktestConfig())
    result = runner.run({"TREND": bars_factory(closes)})
    assert result.strategy_name == "donchian_breakout"
    assert result.metrics["open_positions_at_end"] >= 1
    assert result.metrics["final_equity"] > result.metrics["initial_equity"]


def test_relative_momentum_backtest_with_benchmark(bars_factory) -> None:
    strong = 100.0 * np.array([1.01**i for i in range(160)])
    weak = 100.0 * np.array([1.001**i for i in range(160)])
    strategy = RelativeMomentumStrategy(benchmark=bars_factory(weak))
    runner = BacktestRunner(strategy, config=BacktestConfig())
    result = runner.run({"STRONG": bars_factory(strong)})
    assert result.metrics["open_positions_at_end"] >= 1


class BuyFromStrategy(Strategy):
    """BUY à partir de ``buy_from``, SELL à partir de ``sell_from`` (ATR fixe)."""

    name = "buy_from"

    def __init__(self, buy_from: int, sell_from: int | None = None) -> None:
        self._buy_from = buy_from
        self._sell_from = sell_from

    @property
    def warmup_bars(self) -> int:
        return 2

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        out = bars.copy()
        out["atr"] = 1.0
        return out

    def compute_signals(self, prepared: pd.DataFrame) -> pd.Series:
        positions = pd.Series(np.arange(len(prepared)), index=prepared.index)
        signals = pd.Series(SignalType.HOLD, index=prepared.index, dtype=object)
        signals.loc[positions >= self._buy_from] = SignalType.BUY
        if self._sell_from is not None:
            signals.loc[positions >= self._sell_from] = SignalType.SELL
        return signals


def _rising(n: int, rate: float = 1.01) -> np.ndarray:
    return 100.0 * np.array([rate**i * (1.0 + 0.002 * math.sin(i)) for i in range(n)])


def test_bull_regime_raises_gross_exposure(bars_factory) -> None:
    flat = bars_factory([100.0] * 80, start="2024-01-02")
    data = {"AAA": flat, "BBB": flat, "CCC": flat}
    strategy = BuyFromStrategy(buy_from=3)
    runner = BacktestRunner(strategy, config=BacktestConfig())

    base = runner.run(data)
    base_max_gross = float(base.equity_curve["gross_exposure"].max())
    assert base_max_gross < 0.79 * 100_000.0  # 3 x 20 % en régime prudent

    # Benchmark en tendance depuis assez longtemps pour armer la SMA 200.
    spy = bars_factory(_rising(330), start="2023-01-02")
    bull = runner.run(data, benchmark=spy)
    bull_max_gross = float(bull.equity_curve["gross_exposure"].max())
    assert bull_max_gross > 0.85 * 100_000.0  # 3 x 30 % en régime haussier


def test_ranking_favors_strongest_momentum(bars_factory) -> None:
    strong = bars_factory(_rising(135), start="2023-01-02")
    weak = bars_factory([100.0] * 135, start="2023-01-02")
    data = {"AAA": weak, "ZULU": strong}
    strategy = BuyFromStrategy(buy_from=125, sell_from=130)
    runner = BacktestRunner(
        strategy,
        config=BacktestConfig(),
        risk_config=RiskConfig(max_open_positions=1),
    )
    result = runner.run(data)
    # Une seule position ouverte à la fois : c'est ZULU (momentum vigoureux)
    # qui a consommé le budget, pas AAA (score NaN, relegué).
    assert len(result.trades) == 1
    assert result.trades[0].symbol == "ZULU"


class BuySymbolOnce(Strategy):
    """Stratégie déterministe : un BUY sur un seul symbole, puis HOLD."""

    name = "buy_symbol_once"

    def __init__(self, target: str = "AAPL", buy_index: int = 5, atr: float = 1.0) -> None:
        self._target = target
        self._buy_index = buy_index
        self._atr = atr

    @property
    def warmup_bars(self) -> int:
        return 2

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        out = bars.copy()
        out["ema_fast"] = out["close"]
        out["ema_slow"] = out["close"]
        out["rsi"] = 50.0
        out["atr"] = self._atr
        return out

    def compute_signals(self, prepared: pd.DataFrame) -> pd.Series:
        signals = pd.Series(SignalType.HOLD, index=prepared.index, dtype=object)
        if prepared.attrs.get("symbol") == self._target and self._buy_index < len(signals):
            signals.iloc[self._buy_index] = SignalType.BUY
        return signals


def test_backtest_applies_partial_take_profit(bars_factory) -> None:
    """La prise de profit partielle du live existe aussi dans le backtest."""
    closes = [100.0] * 8 + [100.0 * 1.04**i for i in range(1, 30)]
    bars = bars_factory(closes, spread=0.2)
    risk = RiskConfig(
        max_single_position_weight=0.99,
        max_position_pct=0.99,
        max_gross_exposure_pct=1.0,
    )
    runner = BacktestRunner(
        BuyOnceStrategy(buy_index=5, atr=1.0),
        config=BacktestConfig(initial_equity=100_000.0),
        risk_config=risk,
    )

    result = runner.run({"AAPL": bars})

    assert "partial_take_profit" in {trade.exit_reason for trade in result.trades}


def test_backtest_applies_overweight_trim(bars_factory) -> None:
    """L'écrémage du live (position > poids max) existe aussi dans le backtest."""
    closes = [100.0] * 8 + [100.0 * 1.05**i for i in range(1, 30)]
    bars = bars_factory(closes, spread=0.2)
    risk = RiskConfig(
        max_single_position_weight=0.10,
        max_position_pct=0.10,
        max_gross_exposure_pct=0.90,
        partial_take_profit_pct=0.0,
    )
    runner = BacktestRunner(
        BuyOnceStrategy(buy_index=5, atr=1.0),
        config=BacktestConfig(initial_equity=100_000.0),
        risk_config=risk,
    )

    result = runner.run({"AAPL": bars})

    assert "position_trim_overweight" in {trade.exit_reason for trade in result.trades}


def test_backtest_sweeps_excess_cash_into_safe_havens(bars_factory) -> None:
    """Le cash excédentaire part dans la poche de repli, comme en live."""
    flat = bars_factory([100.0] * 60, spread=0.1)
    risk = RiskConfig(
        safe_haven_enabled=True,
        safe_haven_symbols=["SHV"],
        cash_buffer_pct=0.10,
        max_safe_haven_total_pct=0.50,
        max_position_pct=0.10,
        max_gross_exposure_pct=0.90,
    )
    runner = BacktestRunner(
        BuySymbolOnce(target="AAPL", buy_index=5, atr=1.0),
        config=BacktestConfig(initial_equity=100_000.0),
        risk_config=risk,
    )

    result = runner.run({"AAPL": flat, "SHV": flat})

    assert result.final_positions.get("SHV", 0.0) > 0.0
    assert "SHV" not in {trade.symbol for trade in result.trades}


def test_shield_sword_backtest_pays_no_synthetic_cash_interest(bars_factory) -> None:
    """Le cash ne rapporte rien chez le courtier paper : le backtest non plus."""
    from trading_bot.strategies.shield_sword import ShieldSwordStrategy

    rising = bars_factory([100.0 * 1.002**i for i in range(260)], spread=0.1)
    falling = bars_factory([200.0 * 0.999**i for i in range(260)], spread=0.1)
    strategy = ShieldSwordStrategy(benchmark=falling)

    runner = BacktestRunner(
        strategy,
        config=BacktestConfig(initial_equity=100_000.0),
        risk_config=RiskConfig(),
    )
    result = runner.run({"AAPL": rising}, benchmark=falling)

    # Régime baissier : aucune entrée offensive, donc l'équité reste intacte.
    assert result.metrics["final_equity"] == pytest.approx(100_000.0)


def test_shield_topup_keeps_the_existing_sleeve(bars_factory) -> None:
    """Un complément de Bouclier s'ajoute à la poche, il ne la remplace pas."""
    from trading_bot.strategies.shield_sword import ShieldSwordParams, ShieldSwordStrategy

    # SGOV recule de 7,5 % puis se stabilise : le Bouclier est réajusté en cours de route.
    closes = (
        [100.0] * 205
        + [100.0 - 1.5 * i for i in range(1, 6)]
        + [92.5 + 0.2 * i for i in range(40)]
    )
    sgov = bars_factory(closes, spread=0.05)
    falling = bars_factory([200.0 * 0.999**i for i in range(len(closes))], spread=0.05)

    strategy = ShieldSwordStrategy(
        ShieldSwordParams(shield_symbol="SGOV", shield_symbols=("SGOV",)),
        benchmark=falling,
    )
    runner = BacktestRunner(
        strategy,
        config=BacktestConfig(initial_equity=100_000.0),
        risk_config=RiskConfig(shield_symbol="SGOV", safe_haven_symbols=["SHV"]),
    )

    result = runner.run({"SGOV": sgov}, benchmark=falling, start_date=sgov.index[0])

    assert not result.halts, "un réajustement du Bouclier a fait chuter l'équité"
    # 70 % de 100 000 ≈ 70 000 investis, à ~92-100 $ l'action
    assert result.final_positions["SGOV"] > 400.0
    assert result.metrics["final_equity"] > 90_000.0
