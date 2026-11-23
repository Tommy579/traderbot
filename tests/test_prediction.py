"""Tests du moteur de prédiction multi-horizons (purement informatif)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trading_bot.config import Settings
from trading_bot.data.news_sentiment import NewsAssessment
from trading_bot.prediction import HISTORICAL_CRASH_ANALOGUES, build_market_prediction
from trading_bot.risk.engine import Position

NOW = datetime(2026, 11, 20, 15, 0, tzinfo=UTC)


@pytest.fixture
def pred_universe(bars_factory):
    """Univers réaliste (> 250 barres) pour alimenter les 4 stratégies."""
    aapl = bars_factory([100.0 * (1.0012**i) for i in range(260)], start="2025-11-01")
    nvda = bars_factory([140.0 * (1.0018**i) for i in range(260)], start="2025-11-01")
    sgov = bars_factory([100.0 * (1.00018**i) for i in range(260)], spread=0.02, start="2025-11-01")
    spy = bars_factory([450.0 * (1.0007**i) for i in range(260)], start="2025-11-01")
    return {"AAPL": aapl, "NVDA": nvda, "SGOV": sgov}, spy


def test_prediction_multi_horizon_and_all_strategies(settings, pred_universe) -> None:
    bars, spy = pred_universe
    for horizon in (21, 63, 126):
        out = build_market_prediction(
            settings,
            bars_by_symbol=bars,
            benchmark_bars=spy,
            primary_strategy="shield_sword",
            horizon_days=horizon,
            initial_equity=100_000.0,
            now=NOW,
        )
        assert out["horizon_days"] == horizon
        assert len(out["timeline_points"]) == horizon
        assert len(out["milestones"]) >= 2
        assert len(out["strategies_comparison"]) == 4
        assert all(row["simulated"] is True for row in out["strategies_comparison"])
        assert {row["symbol"] for row in out["symbol_forecasts"]} == {"AAPL", "NVDA"}
        assert len(out["historical_context"]["crash_analogues"]) == len(
            HISTORICAL_CRASH_ANALOGUES
        )

        # La trajectoire haussière doit finir au-dessus du scénario krach sur les actions
        for sym_row in out["symbol_forecasts"]:
            assert sym_row["bull_price"] > sym_row["crash_price"]

        # Bouclier + Épée planifie des rebalancements mensuels sur l'horizon
        rebalances = [
            act for act in out["predicted_actions"] if act["kind"] == "rebalance"
        ]
        assert len(rebalances) >= 1


def test_prediction_incorporates_llm_news_sentiment_and_threats(
    settings, pred_universe
) -> None:
    bars, spy = pred_universe

    bull_news = {
        "NVDA": NewsAssessment(
            symbol="NVDA",
            sentiment_score=0.85,
            confidence=0.9,
            is_critical_threat=False,
            rationale="Résultats trimestriels très au-dessus des attentes.",
            articles_count=4,
            updated_at=NOW,
        )
    }
    bear_news = {
        "NVDA": NewsAssessment(
            symbol="NVDA",
            sentiment_score=-0.92,
            confidence=0.95,
            is_critical_threat=True,
            rationale="Enquête réglementaire majeure et avertissement sur résultats.",
            articles_count=5,
            updated_at=NOW,
        )
    }

    pos_pred = build_market_prediction(
        settings,
        bars_by_symbol=bars,
        benchmark_bars=spy,
        news_assessments=bull_news,
        horizon_days=63,
        now=NOW,
    )
    neg_pred = build_market_prediction(
        settings,
        bars_by_symbol=bars,
        benchmark_bars=spy,
        news_assessments=bear_news,
        current_positions={
            "NVDA": Position(
                symbol="NVDA",
                qty=10.0,
                avg_entry_price=200.0,
                current_price=float(bars["NVDA"]["close"].iloc[-1]),
                stop_price=190.0,
            )
        },
        horizon_days=63,
        now=NOW,
    )

    nvda_pos = next(r for r in pos_pred["symbol_forecasts"] if r["symbol"] == "NVDA")
    nvda_neg = next(r for r in neg_pred["symbol_forecasts"] if r["symbol"] == "NVDA")
    assert nvda_pos["central_price"] > nvda_neg["central_price"]

    assert neg_pred["market_outlook"]["crash_risk"]["score"] > pos_pred["market_outlook"]["crash_risk"]["score"]
    assert "NVDA" in neg_pred["news_outlook"]["critical_symbols"]
    assert "NVDA" in neg_pred["news_outlook"]["vetoed_symbols"]
    assert any(act["kind"] == "panic" and act["symbol"] == "NVDA" for act in neg_pred["predicted_actions"])


def test_prediction_detects_past_crash_and_raises_risk(settings, bars_factory) -> None:
    # Série calme vs série marquée par un krach passé de -30 % + volatilité récente
    calm_spy = bars_factory([400.0 * (1.0005**i) for i in range(260)], spread=0.01)
    calm_aapl = bars_factory([150.0 * (1.0006**i) for i in range(260)], spread=0.01)

    crash_closes = [400.0 * (1.0005**i) for i in range(200)]
    for _i in range(25):
        crash_closes.append(crash_closes[-1] * 0.985)
    for i in range(35):
        crash_closes.append(crash_closes[-1] * (1.022 if i % 2 == 0 else 0.975))
    crashed_spy = bars_factory(crash_closes, spread=0.04)

    calm_pred = build_market_prediction(
        settings,
        bars_by_symbol={"AAPL": calm_aapl},
        benchmark_bars=calm_spy,
        horizon_days=42,
        now=NOW,
    )
    crash_pred = build_market_prediction(
        settings,
        bars_by_symbol={"AAPL": calm_aapl},
        benchmark_bars=crashed_spy,
        horizon_days=42,
        now=NOW,
    )

    events = crash_pred["historical_context"]["observed_events"]
    assert events["max_drawdown_pct"] <= -0.25
    assert events["worst_20d_return_pct"] <= -0.15
    assert crash_pred["market_outlook"]["crash_risk"]["score"] > calm_pred["market_outlook"]["crash_risk"]["score"]


def test_prediction_is_read_only_and_touches_no_bot_state(
    settings: Settings, pred_universe
) -> None:
    bars, spy = pred_universe
    state_dir = settings.state_dir
    before = sorted(p.name for p in state_dir.iterdir()) if state_dir.exists() else []

    out = build_market_prediction(
        settings,
        bars_by_symbol=bars,
        benchmark_bars=spy,
        horizon_days=63,
        now=NOW,
    )
    assert out["disclaimer"]
    after = sorted(p.name for p in state_dir.iterdir()) if state_dir.exists() else []
    assert after == before


def test_prediction_monte_carlo_quantiles_and_live_portfolio(
    settings: Settings, pred_universe
) -> None:
    """Le bootstrap Monte Carlo est déterministe, ordonné P10<=P50<=P90 et utilise les positions."""
    bars, spy = pred_universe
    pos = {
        "NVDA": Position(
            symbol="NVDA",
            qty=100.0,
            avg_entry_price=150.0,
            current_price=float(bars["NVDA"]["close"].iloc[-1]),
            stop_price=140.0,
            initial_stop_price=140.0,
        )
    }
    first = build_market_prediction(
        settings,
        bars_by_symbol=bars,
        benchmark_bars=spy,
        current_positions=pos,
        initial_cash=50_000.0,
        horizon_days=63,
        now=NOW,
    )
    second = build_market_prediction(
        settings,
        bars_by_symbol=bars,
        benchmark_bars=spy,
        current_positions=pos,
        initial_cash=50_000.0,
        horizon_days=63,
        now=NOW,
    )
    assert first["uses_live_portfolio"] is True
    assert first["timeline_points"] == second["timeline_points"]
    mc = first["monte_carlo_summary"]
    assert mc["simulations"] == 200
    assert mc["benchmark_p10_return_pct"] <= mc["benchmark_p50_return_pct"] <= mc["benchmark_p90_return_pct"]


def test_prediction_detects_long_history_crash_episodes_and_macro_spy_news(
    settings: Settings, bars_factory, pred_universe
) -> None:
    """Un historique long SPY expose les épisodes de krach réels et les news macro SPY."""
    bars, spy = pred_universe
    # Construit 600 barres avec deux krachs distincts (-22 % puis -18 %).
    closes = [300.0]
    for _ in range(150):
        closes.append(closes[-1] * 1.001)
    for _ in range(25):
        closes.append(closes[-1] * 0.99)
    for _ in range(180):
        closes.append(closes[-1] * 1.002)
    for _ in range(25):
        closes.append(closes[-1] * 0.992)
    for _ in range(120):
        closes.append(closes[-1] * 1.0015)
    long_spy = bars_factory(closes, start="2020-01-02")

    macro_alert = {
        "SPY": NewsAssessment(
            symbol="SPY",
            sentiment_score=-0.85,
            confidence=0.9,
            is_critical_threat=True,
            rationale="Choc macroéconomique systémique sur les marchés américains.",
            articles_count=6,
            updated_at=NOW,
        )
    }
    out = build_market_prediction(
        settings,
        bars_by_symbol=bars,
        benchmark_bars=spy,
        benchmark_long_bars=long_spy,
        news_assessments=macro_alert,
        horizon_days=63,
        now=NOW,
    )
    episodes = out["historical_context"]["crash_episodes"]
    assert len(episodes) >= 2
    assert out["news_outlook"]["macro_score"] == pytest.approx(-0.85)
    assert "SPY" in out["news_outlook"]["critical_symbols"]
