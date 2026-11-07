"""Tests de la ligne de commande : parseur, formatage et codes de sortie.

Aucun réseau : les commandes testées s'arrêtent sur une validation d'arguments
ou n'utilisent que du formatage pur. Les cycles live et backtest complets sont
couverts ailleurs (``tests/test_main.py``, ``tests/test_backtest.py``).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from trading_bot.backtest.runner import BacktestResult
from trading_bot.data.news_sentiment import NewsAssessment
from trading_bot.main import (
    STRATEGY_NAMES,
    _format_news_row,
    _format_number,
    _format_pct,
    _settings_from_args,
    build_parser,
    cli,
    news_effect_label,
    print_backtest_summary,
)
from trading_bot.risk.engine import RiskConfig, RiskEngine

NOW = datetime(2024, 1, 10, 15, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Parseur
# ---------------------------------------------------------------------------
def test_parser_exposes_every_documented_command() -> None:
    """Chaque commande du README existe réellement dans le parseur."""
    parser = build_parser()
    actions = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    assert len(actions) == 1
    commands = set(actions[0].choices)
    assert commands == {
        "live",
        "check",
        "web",
        "news",
        "publish-stats",
        "backtest",
        "withdraw",
        "rebase",
    }


def test_backtest_sword_defaults_defer_to_configuration() -> None:
    """``--sword-*`` non fourni doit valoir ``None`` pour laisser parler le .env."""
    args = build_parser().parse_args(["backtest"])
    assert args.sword_top_n is None
    assert args.sword_pct is None
    assert args.shield_pct is None
    assert args.sword_rebalance_days is None
    assert args.sword_trend_sma is None
    assert args.strategy in STRATEGY_NAMES


def test_rebase_requires_a_reason() -> None:
    """La justification du ré-ancrage n'est pas optionnelle."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["rebase"])


def test_settings_from_args_applies_overrides(tmp_path) -> None:
    """Les options de la ligne de commande priment sur le fichier .env."""
    args = argparse.Namespace(symbols="aapl, msft", dry_run=True, log_level="DEBUG")
    settings = _settings_from_args(args)
    assert settings.symbols == ["AAPL", "MSFT"]
    assert settings.dry_run is True
    assert settings.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# Codes de sortie, sans réseau
# ---------------------------------------------------------------------------
def test_backtest_rejects_reversed_dates() -> None:
    """Un intervalle incohérent échoue avant tout téléchargement."""
    assert cli(["backtest", "--start", "2023-01-01", "--end", "2022-01-01"]) == 2


def test_backtest_rejects_unreadable_dates() -> None:
    """Une date illisible est une erreur d'usage, pas une exception."""
    assert cli(["backtest", "--start", "pas-une-date"]) == 2


def test_live_rejects_unknown_strategy() -> None:
    """Argparse refuse une stratégie inconnue au lieu de démarrer un bot muet."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["live", "--strategy", "magic"])


# ---------------------------------------------------------------------------
# Formatage
# ---------------------------------------------------------------------------
def test_percent_and_number_formatting() -> None:
    """Les formateurs affichent « n/a » plutôt que de lever sur une valeur absente."""
    assert _format_pct(None) == "n/a"
    assert _format_pct(0.1234) == "12.34%"
    assert _format_number(None) == "n/a"
    assert _format_number(1234.5678) == "1,234.57"


def test_print_backtest_summary_shows_the_expected_sections(capsys) -> None:
    """Le résumé d'un backtest reste lisible sans graphique."""
    curve = pd.DataFrame(
        {"equity": [100_000.0, 101_000.0], "drawdown_pct": [0.0, -0.01]},
        index=pd.DatetimeIndex(
            [pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-03", tz="UTC")]
        ),
    )
    metrics: dict[str, object] = {
        "initial_equity": 100_000.0,
        "final_equity": 101_000.0,
        "total_return": 0.01,
        "cagr": 0.01,
        "sharpe": 1.2,
        "sortino": 1.5,
        "max_drawdown": -0.01,
        "calmar": 1.0,
        "trades": 2,
        "win_rate": 0.5,
        "profit_factor": 1.4,
        "expectancy": 50.0,
        "total_fees": 20.0,
        "exposure_pct": 0.5,
        "halt_events": 0,
        "entry_rejections": 1,
        "open_positions_at_end": 1,
        "benchmark_buy_hold": 0.02,
        "basket_buy_hold": 0.03,
    }
    result = BacktestResult(
        equity_curve=curve,
        trades=(),
        metrics=metrics,  # type: ignore[arg-type]
        halts=(),
        initial_equity=100_000.0,
        symbols=("AAPL",),
        strategy_name="ema_cross",
    )

    print_backtest_summary(result)

    output = capsys.readouterr().out
    assert "Backtest ema_cross" in output
    assert "Sharpe" in output
    assert "1,000.00" in output
    assert "1.20" in output


# ---------------------------------------------------------------------------
# Filtre d'actualités : libellés partagés CLI / page
# ---------------------------------------------------------------------------
def _assessment(score: float, *, critical: bool = False) -> NewsAssessment:
    return NewsAssessment(
        symbol="NVDA",
        sentiment_score=score,
        confidence=0.8,
        is_critical_threat=critical,
        rationale="test",
        articles_count=1,
        updated_at=NOW,
    )


def test_news_effect_label_matches_the_risk_engine() -> None:
    """Le libellé affiché décrit exactement ce que le moteur appliquera."""
    engine = RiskEngine(RiskConfig())
    panic = engine.config.news_panic_exit_threshold

    assert news_effect_label(_assessment(-0.10), engine=engine, panic=panic) == "—"
    assert (
        news_effect_label(_assessment(-0.70), engine=engine, panic=panic) == "VETO ENTRÉE"
    )
    assert (
        news_effect_label(_assessment(-0.95, critical=True), engine=engine, panic=panic)
        == "SORTIE URGENCE"
    )


def test_format_news_row_is_aligned_and_short() -> None:
    """Chaque ligne du radar tient sur une ligne de terminal."""
    engine = RiskEngine(RiskConfig())
    row = _format_news_row(
        "NVDA",
        _assessment(-0.70),
        panic=engine.config.news_panic_exit_threshold,
        engine=engine,
    )
    assert row.startswith("NVDA")
    assert "-0.70" in row
    assert "VETO ENTRÉE" in row
    assert len(row) < 120


def test_withdraw_payload_is_json_ready() -> None:
    """Le résumé JSON d'un retrait reste sérialisable tel quel."""
    from trading_bot.main import _withdrawal_payload
    from trading_bot.withdrawal import WithdrawalFill, WithdrawalResult

    result = WithdrawalResult(
        fills=[WithdrawalFill("AAPL", 10.0, 100.0, "manual_withdrawal", "ord-1")],
        cash=1_000.0,
        floor=None,
    )
    payload = _withdrawal_payload("symbol", result)
    assert payload["ok"] is True
    assert payload["fills"][0] == {
        "symbol": "AAPL",
        "qty": 10.0,
        "price": 100.0,
        "cash_freed": 1000.0,
        "reason": "manual_withdrawal",
    }
    assert json.loads(json.dumps(payload))["cash"] == 1_000.0
