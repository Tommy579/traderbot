"""Tests de la configuration et des garde-fous paper-only."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trading_bot.config import (
    PAPER_ENDPOINT,
    ConfigurationError,
    RiskConfig,
    Settings,
    is_paper_endpoint,
)


def test_symbols_csv_parsed_and_deduplicated() -> None:
    settings = Settings(_env_file=None, symbols="aapl, msft;AAPL")
    assert settings.symbols == ["AAPL", "MSFT"]


def test_default_universe_is_the_configured_basket() -> None:
    """L'univers par défaut est le panier de 12 valeurs, SPY exclu."""
    settings = Settings(_env_file=None)
    assert settings.symbols == [
        "NVDA",
        "AMD",
        "AVGO",
        "PLTR",
        "MSFT",
        "AMZN",
        "META",
        "XOM",
        "NEE",
        "LLY",
        "JPM",
        "COST",
    ]
    assert settings.benchmark_symbol == "SPY"
    assert "SPY" not in settings.symbols


def test_benchmark_is_removed_from_the_tradable_universe() -> None:
    """SPY présent par erreur dans SYMBOLS ne doit jamais être tradé.

    Le benchmark sert de référence (régime de marché, force relative) :
    l'acheter reviendrait à doubler l'exposition au marché.
    """
    settings = Settings(_env_file=None, symbols=["NVDA", "SPY", "MSFT"])
    assert settings.symbols == ["NVDA", "MSFT"]
    assert settings.benchmark_symbol == "SPY"


def test_benchmark_filter_follows_the_configured_reference() -> None:
    """Le filtre suit BENCHMARK_SYMBOL, pas un nom figé dans le code."""
    settings = Settings(
        _env_file=None, symbols=["NVDA", "QQQ", "MSFT"], benchmark_symbol="QQQ"
    )
    assert settings.symbols == ["NVDA", "MSFT"]


def test_universe_with_only_the_benchmark_is_rejected() -> None:
    """Refuser plutôt que de lancer un bot sans aucun actif à trader."""
    with pytest.raises(ValidationError, match="benchmark"):
        Settings(_env_file=None, symbols=["SPY"])


@pytest.mark.parametrize(
    "url",
    [
        "https://api.alpaca.markets",
        "https://paper-api.alpaca.markets.evil.example",
        "http://paper-api.alpaca.markets",
        "https://example.com",
    ],
)
def test_non_paper_endpoints_rejected(url: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, alpaca_trading_url=url)


def test_paper_endpoint_accepted() -> None:
    settings = Settings(_env_file=None, alpaca_trading_url=PAPER_ENDPOINT + "/")
    assert settings.is_paper
    assert settings.alpaca_trading_url == PAPER_ENDPOINT


def test_is_paper_endpoint_function() -> None:
    assert is_paper_endpoint(PAPER_ENDPOINT)
    assert not is_paper_endpoint("https://api.alpaca.markets")


def test_risk_overrides_are_applied() -> None:
    settings = Settings(
        _env_file=None,
        risk_per_trade_pct=0.02,
        max_daily_loss_pct=0.05,
    )
    assert settings.risk.risk_per_trade_pct == 0.02
    assert settings.risk.max_daily_loss_pct == 0.05
    assert settings.risk.max_position_pct == RiskConfig().max_position_pct


def test_risk_config_rejects_incoherent_limits() -> None:
    with pytest.raises(ValidationError):
        RiskConfig(max_position_pct=0.9, max_gross_exposure_pct=0.5)
    with pytest.raises(ValidationError):
        RiskConfig(max_daily_loss_pct=0.20, max_drawdown_pct=0.10)


def test_risk_config_rejects_invalid_percentages() -> None:
    with pytest.raises(ValidationError):
        RiskConfig(risk_per_trade_pct=0.5)
    with pytest.raises(ValidationError):
        RiskConfig(atr_stop_multiple=0.0)


def test_fractional_disabled_requires_integer_quantities() -> None:
    with pytest.raises(ValidationError):
        RiskConfig(fractional_shares=False, qty_decimals=4)
    assert RiskConfig(fractional_shares=False, qty_decimals=0).qty_decimals == 0


def test_credentials_required_for_trading() -> None:
    settings = Settings(_env_file=None)
    with pytest.raises(ConfigurationError):
        settings.require_credentials()


def test_credentials_returned_without_leaking_secret() -> None:
    settings = Settings(
        _env_file=None,
        alpaca_api_key="key-123",
        alpaca_secret_key="secret-456",
    )
    assert settings.require_credentials() == ("key-123", "secret-456")
    summary = settings.redacted_summary()
    assert "secret-456" not in str(summary)
    assert "key-123" not in str(summary)


def test_empty_symbols_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, symbols=" , ")


def test_invalid_ticker_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, symbols="AAPL,$BAD")


def test_lookback_must_be_sufficient() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, lookback_days=10)


def test_explicit_risk_config_survives_default_flat_fields() -> None:
    """Un RiskConfig passé explicitement n'est pas écrasé par les défauts plats.

    ``safe_haven_enabled`` et ``safe_haven_symbols`` ont une valeur par défaut
    non nulle au niveau racine : sans contrôle des champs réellement fournis,
    ils écrasent silencieusement tout profil de risque construit à la main.
    """
    settings = Settings(
        _env_file=None,
        risk=RiskConfig(safe_haven_enabled=False, max_open_positions=2),
    )
    assert settings.risk.safe_haven_enabled is False
    assert settings.risk.max_open_positions == 2


def test_environment_flat_override_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Une variable d'environnement plate continue de piloter le profil de risque."""
    monkeypatch.setenv("SAFE_HAVEN_ENABLED", "false")
    monkeypatch.setenv("MAX_OPEN_POSITIONS", "7")
    settings = Settings(_env_file=None)
    assert settings.risk.safe_haven_enabled is False
    assert settings.risk.max_open_positions == 7
