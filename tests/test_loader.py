"""Tests du chargement, de la normalisation et des indicateurs."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from trading_bot.data.loader import (
    CachedProvider,
    DataError,
    DataQualityError,
    compute_indicators,
    drop_incomplete_bar,
    lookback_window,
    normalize_ohlcv,
    timeframe_interval,
)


def _raw_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    close = pd.Series(np.linspace(100.0, 110.0, len(index)), index=index)
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1_000.0,
        },
        index=index,
    )


def test_normalize_sorts_deduplicates_and_converts_utc() -> None:
    naive = pd.date_range("2024-01-03", periods=3, freq="D")
    raw = _raw_frame(naive)
    raw = pd.concat([raw, raw.iloc[[1]]])  # doublon volontaire
    normalized = normalize_ohlcv(raw.iloc[::-1], "TEST")
    assert isinstance(normalized.index, pd.DatetimeIndex)
    assert str(normalized.index.tz) == "UTC"
    assert normalized.index.is_monotonic_increasing
    assert not normalized.index.has_duplicates
    assert list(normalized.columns) == ["open", "high", "low", "close", "volume"]
    assert normalized.dtypes.eq(float).all()


def test_normalize_flattens_multiindex_columns() -> None:
    index = pd.date_range("2024-01-02", periods=4, freq="D", tz="UTC")
    raw = _raw_frame(index)
    raw.columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["AAPL"]]
    )
    normalized = normalize_ohlcv(raw, "AAPL")
    assert list(normalized.columns) == ["open", "high", "low", "close", "volume"]


def test_normalize_drops_incoherent_bars(bars_factory) -> None:
    bars = bars_factory(np.linspace(100, 105, 6))
    bars.loc[bars.index[2], "high"] = 1.0  # high < low : incohérent
    normalized = normalize_ohlcv(bars, "TEST")
    assert len(normalized) == len(bars) - 1


def test_normalize_strict_mode_raises(bars_factory) -> None:
    bars = bars_factory(np.linspace(100, 105, 6))
    bars.loc[bars.index[2], "high"] = 1.0
    with pytest.raises(DataQualityError):
        normalize_ohlcv(bars, "TEST", strict=True)


def test_normalize_rejects_missing_columns(bars_factory) -> None:
    bars = bars_factory(np.linspace(100, 105, 6)).drop(columns=["volume"])
    with pytest.raises(DataQualityError, match="colonnes manquantes"):
        normalize_ohlcv(bars, "TEST")


def test_normalize_rejects_empty_frame() -> None:
    with pytest.raises(DataQualityError):
        normalize_ohlcv(pd.DataFrame(), "TEST")


def test_compute_indicators_masks_warmup(bars_factory) -> None:
    bars = bars_factory(np.linspace(100, 140, 100))
    enriched = compute_indicators(bars, fast=10, slow=30, rsi_length=14, atr_length=14)
    warmup = max(30, 15, 15) - 1
    assert enriched["ema_slow"].isna().sum() == warmup
    assert enriched["rsi"].isna().sum() == warmup
    assert enriched["atr"].isna().sum() == warmup
    assert enriched["atr"].dropna().gt(0).all()


def test_pandas_ta_and_native_backends_are_identical(bars_factory) -> None:
    pytest.importorskip("pandas_ta_classic")
    rng = np.random.default_rng(11)
    closes = 100 + np.cumsum(rng.normal(0, 1.0, 300))
    bars = bars_factory(closes, spread=1.0)
    native = compute_indicators(bars, backend="native")
    accelerated = compute_indicators(bars, backend="pandas_ta")
    for column in ("ema_fast", "ema_slow", "rsi", "atr"):
        np.testing.assert_allclose(
            native[column].to_numpy(),
            accelerated[column].to_numpy(),
            rtol=0.0,
            atol=1e-9,
            equal_nan=True,
        )


def test_insufficient_bars_rejected(bars_factory) -> None:
    with pytest.raises(DataError):
        compute_indicators(bars_factory(np.linspace(100, 110, 20)))


def test_invalid_ema_periods_rejected(bars_factory) -> None:
    bars = bars_factory(np.linspace(100, 140, 100))
    with pytest.raises(DataError):
        compute_indicators(bars, fast=50, slow=20)


def test_unknown_backend_rejected(bars_factory) -> None:
    bars = bars_factory(np.linspace(100, 140, 100))
    with pytest.raises(DataError):
        compute_indicators(bars, backend="talib")  # type: ignore[arg-type]


def test_drop_incomplete_daily_bar_during_session() -> None:
    index = pd.date_range("2024-01-08 05:00", periods=3, freq="D", tz="UTC")
    bars = _raw_frame(index)
    bars.columns = [column.lower() for column in bars.columns]
    during = drop_incomplete_bar(
        bars,
        "1Day",
        datetime(2024, 1, 10, 20, 0, tzinfo=UTC),  # 15:00 ET
    )
    after = drop_incomplete_bar(
        bars,
        "1Day",
        datetime(2024, 1, 10, 21, 30, tzinfo=UTC),  # 16:30 ET
    )
    assert len(during) == 2
    assert len(after) == 3


def test_drop_incomplete_intraday_bar() -> None:
    index = pd.date_range("2024-01-10 14:50", periods=3, freq="5min", tz="UTC")
    bars = _raw_frame(index)
    bars.columns = [column.lower() for column in bars.columns]
    early = drop_incomplete_bar(bars, "5Min", datetime(2024, 1, 10, 14, 58, tzinfo=UTC))
    late = drop_incomplete_bar(bars, "5Min", datetime(2024, 1, 10, 15, 6, tzinfo=UTC))
    assert len(early) == 2
    assert len(late) == 3


def test_timeframe_conversion() -> None:
    assert timeframe_interval("1Day") == "1d"
    assert timeframe_interval("5Min") == "5m"
    with pytest.raises(DataError):
        timeframe_interval("2Weeks")


def test_lookback_window_scales_for_intraday() -> None:
    now = datetime(2024, 6, 3, tzinfo=UTC)
    start_daily, end_daily = lookback_window("1Day", now=now, lookback_days=60)
    start_intraday, _ = lookback_window("5Min", now=now, lookback_days=60)
    assert end_daily == now
    assert (now - start_daily).days == 60
    assert (now - start_intraday) > (now - start_daily)


START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 6, 1, tzinfo=UTC)


class _CountingProvider:
    """Fournisseur factice qui compte ses appels."""

    def __init__(self, bars: pd.DataFrame) -> None:
        self.bars = bars
        self.calls = 0

    def fetch(
        self, symbol: str, start: datetime, end: datetime, interval: str = "1d"
    ) -> pd.DataFrame:
        self.calls += 1
        return self.bars


def test_cached_provider_hits_disk_on_second_call(tmp_path, bars_factory) -> None:
    stub = _CountingProvider(bars_factory(np.linspace(100.0, 120.0, 40)))
    provider = CachedProvider(stub, tmp_path / "cache")
    first = provider.fetch("TEST", START, END, "1d")
    second = provider.fetch("TEST", START, END, "1d")
    assert stub.calls == 1
    assert provider.cache_path("TEST", START, END, "1d").exists()
    pd.testing.assert_frame_equal(first, second, check_freq=False)


def test_cached_provider_refresh_forces_download(tmp_path, bars_factory) -> None:
    stub = _CountingProvider(bars_factory(np.linspace(100.0, 120.0, 40)))
    provider = CachedProvider(stub, tmp_path / "cache")
    provider.fetch("TEST", START, END, "1d")
    refreshed = CachedProvider(stub, tmp_path / "cache", refresh=True)
    refreshed.fetch("TEST", START, END, "1d")
    assert stub.calls == 2


def test_cached_provider_rejects_corrupt_cache(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    provider = CachedProvider(_CountingProvider(pd.DataFrame()), cache_dir)
    provider.cache_path("TEST", START, END, "1d").write_text(
        "nimporte,quoi\n1,2\n", encoding="utf-8"
    )
    with pytest.raises(DataQualityError):
        provider.fetch("TEST", START, END, "1d")


def test_cached_provider_round_trip_is_lossless(tmp_path, bars_factory) -> None:
    """Régression : un cache CSV doit relire les flottants bit à bit.

    pandas écrit des flottants exacts avec ``float_format="%.17g"`` mais son
    parseur CSV par défaut introduit des erreurs de quelques ULP ; sans
    ``float_precision="round_trip"`` deux backtests identiques divergent.
    """
    closes = [191.81321143511212, 173.70339765267045, 1.1 + 0.2, 100.0 / 3.0]
    bars = bars_factory(closes, spread=0.01)
    provider = CachedProvider(_CountingProvider(bars), tmp_path / "cache")
    first = provider.fetch("TEST", START, END, "1d")
    second = provider.fetch("TEST", START, END, "1d")
    assert np.array_equal(first.to_numpy(dtype=float), second.to_numpy(dtype=float))
