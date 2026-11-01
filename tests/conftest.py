"""Fixtures partagées des tests (aucun accès réseau)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_bot.config import Settings


@pytest.fixture
def bars_factory():
    """Fabrique des barres OHLCV synthétiques cohérentes et déterministes."""

    def make(
        closes: Sequence[float],
        *,
        start: str = "2023-01-02",
        spread: float = 0.5,
        volume: float = 1_000_000.0,
    ) -> pd.DataFrame:
        index = pd.date_range(start, periods=len(closes), freq="B", tz="UTC")
        close = pd.Series(np.asarray(closes, dtype=float), index=index)
        open_ = close.shift(1).fillna(close.iloc[0])
        frame = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) + spread,
                "low": np.minimum(open_, close) - spread,
                "close": close,
                "volume": np.full(len(closes), volume, dtype=float),
            },
            index=index,
        )
        frame.index.name = "timestamp"
        return frame

    return make


@pytest.fixture
def wave_bars(bars_factory) -> pd.DataFrame:
    """Trois cycles d'oscillation : produit plusieurs croisements EMA."""
    closes = (
        100.0
        + 10.0 * np.sin(np.linspace(0.0, 6.0 * np.pi, 220))
        + np.linspace(0.0, 5.0, 220)
    )
    return bars_factory(closes)


@pytest.fixture
def crash_bars(bars_factory) -> pd.DataFrame:
    """Tendance baissière, retournement, puis chute brutale (test des stops)."""
    down = np.linspace(120.0, 90.0, 80)
    up = np.linspace(90.0, 115.0, 40)
    crash = np.linspace(115.0, 70.0, 12)
    return bars_factory(np.concatenate([down, up, crash]), spread=1.0)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Configuration paper de test, isolée du fichier .env réel."""
    return Settings(
        _env_file=None,
        alpaca_api_key="PK_TEST",
        alpaca_secret_key="SECRET_TEST",
        symbols=["AAPL"],
        state_dir=tmp_path / "state",
        log_level="DEBUG",
    )


@pytest.fixture
def fixed_now() -> datetime:
    """Instant de test timezone-aware (séance ouverte, 10:00 ET)."""
    return datetime(2024, 1, 10, 15, 0, tzinfo=UTC)
