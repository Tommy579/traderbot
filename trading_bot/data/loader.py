"""Chargement des données de marché et calcul des indicateurs.

Deux responsabilités, volontairement séparées de toute logique de trading :

* ``MarketDataProvider`` / :class:`YFinanceProvider` — acquisition et
  normalisation des barres OHLCV (index UTC, colonnes typées, qualité vérifiée).
* :func:`compute_indicators` — EMA, RSI (Wilder) et ATR via ``pandas-ta``
  lorsque disponible, avec une implémentation native équivalente en repli.

La normalisation produit toujours le même schéma, ce qui rend le backtest et le
live interchangeables : mêmes colonnes, même index temporel, mêmes indicateurs.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Final, Literal, Protocol, TypeAlias, cast

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

OHLCV_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume")
INDICATOR_COLUMNS: Final[tuple[str, ...]] = ("ema_fast", "ema_slow", "rsi", "atr")
BarFrame: TypeAlias = pd.DataFrame
IndicatorBackend = Literal["auto", "pandas_ta", "native"]

TIMEFRAME_TO_YFINANCE: Final[Mapping[str, str]] = {
    "1Min": "1m",
    "5Min": "5m",
    "15Min": "15m",
    "30Min": "30m",
    "1Hour": "1h",
    "1Day": "1d",
}

_BAR_DURATION: Final[Mapping[str, timedelta]] = {
    "1Min": timedelta(minutes=1),
    "5Min": timedelta(minutes=5),
    "15Min": timedelta(minutes=15),
    "30Min": timedelta(minutes=30),
    "1Hour": timedelta(hours=1),
    "1Day": timedelta(days=1),
}

_PANDAS_TA: ModuleType | None = None
_PANDAS_TA_PROBED = False


class DataError(RuntimeError):
    """Erreur d'acquisition ou de normalisation des données de marché."""


class DataQualityError(DataError):
    """Les données reçues ne respectent pas le contrat OHLCV."""


class MarketDataProvider(Protocol):
    """Contrat minimal d'un fournisseur de barres historiques.

    L'intervalle doit être exprimé au format yfinance (``1d``, ``1h``, ``5m``).
    L'index retourné est un ``DatetimeIndex`` UTC trié et unique.
    """

    def fetch(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> BarFrame:
        """Retourne les barres OHLCV de ``symbol`` sur ``[start, end)``."""
        ...


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Aplatit les colonnes MultiIndex renvoyées par yfinance."""
    result = frame.copy()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            str(level[0]).strip().lower() for level in result.columns.to_list()
        ]
    else:
        result.columns = [str(column).strip().lower() for column in result.columns]
    return result


def _to_utc_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Force un index ``DatetimeIndex`` UTC, trié et sans doublons."""
    result = frame.copy()
    if not isinstance(result.index, pd.DatetimeIndex):
        try:
            result.index = pd.DatetimeIndex(pd.to_datetime(result.index, utc=True))
        except (TypeError, ValueError) as exc:
            raise DataQualityError("index temporel illisible") from exc
    index = result.index
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    result.index = index
    result = result[~result.index.duplicated(keep="last")].sort_index()
    result.index.name = "timestamp"
    return result


def normalize_ohlcv(
    raw: pd.DataFrame,
    symbol: str,
    *,
    strict: bool = False,
) -> BarFrame:
    """Normalise un DataFrame brut au contrat OHLCV du projet.

    Étapes : aplatissement des colonnes, conversion numérique, index UTC trié,
    suppression des lignes inexploitables, contrôle de cohérence OHLC.

    Args:
        raw: Données brutes du fournisseur.
        symbol: Ticker associé (utilisé dans les messages d'erreur).
        strict: Si ``True``, toute ligne incohérente lève ``DataQualityError``
            au lieu d'être écartée avec un avertissement.

    Returns:
        DataFrame avec colonnes ``open, high, low, close, volume`` en ``float``.

    Raises:
        DataQualityError: schéma absent, index invalide ou aucune ligne valide.
    """
    if raw is None or raw.empty:
        raise DataQualityError(f"{symbol}: aucune donnée reçue")

    frame = _to_utc_index(_flatten_columns(raw))

    missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise DataQualityError(f"{symbol}: colonnes manquantes {missing}")

    frame = frame.loc[:, list(OHLCV_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame["volume"] = frame["volume"].fillna(0.0).clip(lower=0.0)

    price_ok = (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
    ohlc_ok = (frame["high"] >= frame[["open", "close", "low"]].max(axis=1)) & (
        frame["low"] <= frame[["open", "close", "high"]].min(axis=1)
    )
    valid = price_ok & ohlc_ok
    invalid_count = int((~valid).sum())
    if invalid_count:
        message = f"{symbol}: {invalid_count} barre(s) incohérente(s) écartée(s)"
        if strict:
            raise DataQualityError(message)
        LOGGER.warning(message)
        frame = frame.loc[valid]

    if frame.empty:
        raise DataQualityError(f"{symbol}: aucune barre valide après nettoyage")

    frame = frame.astype(float)
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise DataQualityError(f"{symbol}: valeurs non finies persistantes")

    frame.attrs["symbol"] = symbol
    return frame


class YFinanceProvider:
    """Fournisseur historique basé sur ``yfinance`` (ajusté des splits/dividendes).

    ``end`` est exclusif, comme dans l'API yfinance.
    """

    def __init__(self, *, auto_adjust: bool = True) -> None:
        self._auto_adjust = auto_adjust

    def fetch(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> BarFrame:
        """Télécharge les barres et les normalise.

        Raises:
            DataError: ticker inconnu, réseau indisponible ou réponse vide.
        """
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - dépendance obligatoire
            raise DataError("yfinance n'est pas installé") from exc

        try:
            raw = yf.Ticker(symbol).history(
                start=start,
                end=end,
                interval=interval,
                auto_adjust=self._auto_adjust,
                actions=False,
                raise_errors=False,
            )
        except Exception as exc:  # yfinance lève des types variés
            raise DataError(f"{symbol}: échec du téléchargement yfinance") from exc

        if raw is None or raw.empty:
            raise DataError(f"{symbol}: yfinance n'a retourné aucune barre")
        return normalize_ohlcv(raw, symbol)


class CachedProvider:
    """Cache disque d'un :class:`MarketDataProvider`, pour des backtests reproductibles.

    Un fournisseur réseau peut renvoyer des valeurs légèrement révisées entre
    deux appels (yfinance « répare » certains prix) ; le cache garantit qu'un
    backtest relancé sur la même fenêtre travaille exactement sur les mêmes
    barres. Utilisez ``refresh=True`` pour forcer un nouveau téléchargement.

    Args:
        provider: Fournisseur sous-jacent (yfinance, courtier, fake de test).
        cache_dir: Répertoire des fichiers CSV.
        refresh: Si ``True``, ignore le cache existant et le réécrit.
    """

    def __init__(
        self,
        provider: MarketDataProvider,
        cache_dir: Path,
        *,
        refresh: bool = False,
    ) -> None:
        self._provider = provider
        self._cache_dir = Path(cache_dir)
        self._refresh = refresh

    @property
    def provider(self) -> MarketDataProvider:
        """Fournisseur sous-jacent."""
        return self._provider

    @property
    def cache_dir(self) -> Path:
        """Répertoire de cache effectif."""
        return self._cache_dir

    def cache_path(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> Path:
        """Chemin du fichier de cache pour ce symbole et cette fenêtre."""
        key = f"{symbol}_{interval}_{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}.csv"
        return self._cache_dir / key

    def fetch(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> BarFrame:
        """Retourne les barres du cache, ou les télécharge puis les met en cache."""
        path = self.cache_path(symbol, start, end, interval)
        if path.exists() and not self._refresh:
            LOGGER.debug("cache utilisé pour %s (%s)", symbol, path.name)
            return _read_cached_bars(path, symbol)

        bars = self._provider.fetch(symbol, start, end, interval)
        path.parent.mkdir(parents=True, exist_ok=True)
        bars.to_csv(path, float_format="%.17g")
        LOGGER.debug("barres mises en cache pour %s (%s)", symbol, path.name)
        return bars


def _read_cached_bars(path: Path, symbol: str) -> BarFrame:
    """Relit des barres mises en cache et réapplique le contrat OHLCV.

    Raises:
        DataQualityError: fichier illisible ou contenu invalide.
    """
    try:
        raw = pd.read_csv(path, index_col=0, float_precision="round_trip")
        if raw.empty:
            raise DataQualityError(f"{symbol}: cache vide ({path.name})")
        raw.index = pd.DatetimeIndex(pd.to_datetime(raw.index, utc=True, format="ISO8601"))
    except DataQualityError:
        raise
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DataQualityError(f"{symbol}: cache illisible ({path.name})") from exc
    return normalize_ohlcv(raw, symbol)


def _probe_pandas_ta() -> ModuleType | None:
    """Importe ``pandas_ta`` (ou ``pandas_ta_classic``) une seule fois.

    Les distributions récentes de ``pandas-ta-classic`` exposent le module sous
    le nom ``pandas_ta_classic`` ; les anciennes sous ``pandas_ta``. Les deux
    APIs utilisées ici sont identiques.
    """
    global _PANDAS_TA, _PANDAS_TA_PROBED
    if _PANDAS_TA_PROBED:
        return _PANDAS_TA
    _PANDAS_TA_PROBED = True
    import importlib

    for module_name in ("pandas_ta", "pandas_ta_classic"):
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 - import optionnel, erreurs variées
            continue
        if all(hasattr(module, name) for name in ("ema", "rsi", "atr")):
            _PANDAS_TA = module
            LOGGER.debug("indicateurs calculés avec %s", module_name)
            return _PANDAS_TA
    LOGGER.warning("pandas-ta indisponible : repli sur les indicateurs natifs équivalents")
    _PANDAS_TA = None
    return None


def _sma_seeded_average(
    series: pd.Series,
    length: int,
    *,
    span: int | None = None,
    alpha: float | None = None,
) -> pd.Series:
    """Moyenne exponentielle amorcée par une SMA (convention TA-Lib).

    La première valeur est la moyenne des ``length`` premiers points valides,
    puis la récursion exponentielle démarre. Cette amorce est celle utilisée par
    ``pandas-ta`` (``sma=True`` pour l'EMA, RMA de Wilder pour RSI/ATR), ce qui
    garantit des indicateurs identiques entre les deux backends.
    """
    values = series.astype(float).copy()
    first_valid = values.first_valid_index()
    if first_valid is None:
        return pd.Series(np.nan, index=values.index, dtype=float)
    position = values.index.get_loc(first_valid)
    if position + length > len(values):
        return pd.Series(np.nan, index=values.index, dtype=float)
    seed = float(values.iloc[position : position + length].mean())
    values.iloc[: position + length - 1] = np.nan
    values.iloc[position + length - 1] = seed
    if span is not None:
        return values.ewm(span=span, adjust=False).mean()
    if alpha is not None:
        return values.ewm(alpha=alpha, adjust=False).mean()
    raise ValueError("span ou alpha requis")


def _ema_native(series: pd.Series, length: int) -> pd.Series:
    """EMA amorcée par SMA puis lissage exponentiel (``adjust=False``)."""
    return _sma_seeded_average(series, length, span=length)


def _rsi_native(series: pd.Series, length: int) -> pd.Series:
    """RSI de Wilder, identique à la formule TA-Lib/pandas-ta."""
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = _sma_seeded_average(gains, length, alpha=1.0 / length)
    average_loss = _sma_seeded_average(losses, length, alpha=1.0 / length)
    denominator = average_gain + average_loss.abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100.0 * average_gain / denominator
    rsi = rsi.mask(denominator == 0.0, 0.0)
    return rsi.clip(lower=0.0, upper=100.0)


def _atr_native(bars: pd.DataFrame, length: int) -> pd.Series:
    """ATR de Wilder (RMA amorcée par SMA) à partir du True Range."""
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    previous_close = bars["close"].astype(float).shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # La première barre n'a pas de clôture précédente : TR indéfini (TA-Lib).
    true_range.iloc[0] = np.nan
    return _sma_seeded_average(true_range, length, alpha=1.0 / length)


def compute_atr(bars: pd.DataFrame, *, length: int = 14) -> pd.Series:
    """ATR de Wilder (RMA amorcée par SMA) sur des barres normalisées.

    Exposé séparément de :func:`compute_indicators` pour les stratégies qui
    n'utilisent ni EMA ni RSI (Donchian, momentum) : elles obtiennent exactement
    le même ATR que le reste du système, donc le même dimensionnement de risque.

    Raises:
        DataError: paramètre ou profondeur de données insuffisante.
    """
    if length < 2:
        raise DataError(f"atr_length doit être >= 2 (reçu: {length})")
    missing = [column for column in OHLCV_COLUMNS if column not in bars.columns]
    if missing:
        raise DataQualityError(f"colonnes OHLCV manquantes: {missing}")
    if len(bars) < length + 1:
        raise DataError(f"{len(bars)} barres fournies, {length + 1} requises pour l'ATR")
    return _atr_native(bars, length)


def momentum_zscores(
    close: pd.Series,
    *,
    fast_length: int = 60,
    slow_length: int = 120,
    vol_length: int = 60,
) -> tuple[pd.Series, pd.Series]:
    """Z-scores de momentum (rendement / volatilité scalée par horizon).

    Source unique de la normalisation ``z(h) = (close/close_h - 1) /
    (sigma_journalier * sqrt(h))`` : utilisée par la stratégie de momentum
    relatif et par le classement multi-symboles du moteur de risque, ce qui
    garantit que le ranking en backtest et en live repose sur la même mesure.

    Raises:
        DataError: paramètres incohérents.
    """
    if fast_length < 2 or slow_length <= fast_length or vol_length < 2:
        raise DataError(
            f"paramètres de momentum invalides: fast={fast_length}, "
            f"slow={slow_length}, vol={vol_length}"
        )
    close = close.astype(float)
    returns = close.pct_change()
    vol = returns.rolling(vol_length).std()
    z_fast = (close / close.shift(fast_length) - 1.0) / (vol * math.sqrt(fast_length))
    z_slow = (close / close.shift(slow_length) - 1.0) / (vol * math.sqrt(slow_length))
    return (
        z_fast.replace([np.inf, -np.inf], np.nan),
        z_slow.replace([np.inf, -np.inf], np.nan),
    )


def trend_regime(bars: pd.DataFrame, *, sma_length: int = 200) -> pd.Series:
    """Filtre de régime de marché : clôture au-dessus de sa SMA longue.

    Série booléenne alignée sur les barres fournies (aucun décalage : la
    valeur à la date t n'utilise que les barres achevées jusqu'à t inclus).
    Appelée sur le benchmark (SPY), elle pilote l'ouverture du budget
    d'exposition du portefeuille. Tant que la SMA n'est pas définie (période
    de chauffe), le régime vaut ``False`` — jamais d'optimisme par défaut.

    Raises:
        DataError: paramètre invalide ou schéma incomplet.
    """
    if sma_length < 2:
        raise DataError(f"sma_length doit être >= 2 (reçu: {sma_length})")
    if "close" not in bars.columns:
        raise DataQualityError("colonnes OHLCV manquantes: ['close']")
    close = bars["close"].astype(float)
    sma = close.rolling(sma_length).mean()
    regime = close > sma
    return regime.where(sma.notna(), False).astype(bool)


def _resolve_backend(backend: IndicatorBackend) -> IndicatorBackend:
    """Résout ``auto`` vers ``pandas_ta`` si le paquet est opérationnel."""
    if backend not in ("auto", "pandas_ta", "native"):
        raise DataError(f"backend d'indicateurs inconnu: {backend!r}")
    if backend == "auto":
        return "pandas_ta" if _probe_pandas_ta() is not None else "native"
    if backend == "pandas_ta" and _probe_pandas_ta() is None:
        raise DataError("backend 'pandas_ta' demandé mais indisponible")
    return backend


def compute_indicators(
    bars: pd.DataFrame,
    *,
    fast: int = 20,
    slow: int = 50,
    rsi_length: int = 14,
    atr_length: int = 14,
    backend: IndicatorBackend = "auto",
) -> BarFrame:
    """Calcule EMA rapide/lente, RSI et ATR sur des barres normalisées.

    Les ``max(slow, rsi_length + 1, atr_length + 1) - 1`` premières lignes sont
    volontairement masquées (``NaN``) quel que soit le backend, afin que la
    période de chauffe — et donc les signaux — soient identiques entre
    ``pandas_ta`` et l'implémentation native. C'est une condition de
    reproductibilité du backtest.

    Args:
        bars: DataFrame OHLCV normalisé.
        fast: longueur de l'EMA rapide.
        slow: longueur de l'EMA lente.
        rsi_length: longueur du RSI.
        atr_length: longueur de l'ATR.
        backend: ``auto``, ``pandas_ta`` ou ``native``.

    Returns:
        Copie de ``bars`` enrichie de ``ema_fast, ema_slow, rsi, atr``.

    Raises:
        DataError: paramètres invalides, données insuffisantes ou backend
            explicitement demandé mais indisponible.
    """
    if fast < 2 or slow < 3 or fast >= slow:
        raise DataError(f"périodes EMA invalides: fast={fast}, slow={slow}")
    if rsi_length < 2 or atr_length < 2:
        raise DataError("rsi_length et atr_length doivent être >= 2")
    missing = [column for column in OHLCV_COLUMNS if column not in bars.columns]
    if missing:
        raise DataQualityError(f"colonnes OHLCV manquantes: {missing}")
    if len(bars) < slow + 1:
        raise DataError(
            f"{len(bars)} barres fournies, {slow + 1} requises pour les indicateurs"
        )

    resolved = _resolve_backend(backend)
    out = bars.copy()
    close = out["close"].astype(float)

    if resolved == "pandas_ta":
        pandas_ta = _probe_pandas_ta()
        assert pandas_ta is not None  # garanti par _resolve_backend
        ema_fast = pandas_ta.ema(close, length=fast)
        ema_slow = pandas_ta.ema(close, length=slow)
        rsi = pandas_ta.rsi(close, length=rsi_length)
        atr = pandas_ta.atr(
            out["high"].astype(float),
            out["low"].astype(float),
            close,
            length=atr_length,
        )
        if any(series is None for series in (ema_fast, ema_slow, rsi, atr)):
            raise DataError("pandas-ta n'a pas pu calculer tous les indicateurs")
        out["ema_fast"] = pd.Series(ema_fast, index=out.index, dtype=float)
        out["ema_slow"] = pd.Series(ema_slow, index=out.index, dtype=float)
        out["rsi"] = pd.Series(rsi, index=out.index, dtype=float)
        out["atr"] = pd.Series(atr, index=out.index, dtype=float)
    else:
        out["ema_fast"] = _ema_native(close, fast)
        out["ema_slow"] = _ema_native(close, slow)
        out["rsi"] = _rsi_native(close, rsi_length)
        out["atr"] = _atr_native(out, atr_length)

    warmup = max(slow, rsi_length + 1, atr_length + 1) - 1
    if warmup > 0:
        warmup_index = out.index[:warmup]
        out.loc[warmup_index, list(INDICATOR_COLUMNS)] = np.nan

    out.attrs["symbol"] = bars.attrs.get("symbol", "")
    out.attrs["indicator_backend"] = resolved
    return cast(BarFrame, out)


def timeframe_interval(timeframe: str) -> str:
    """Convertit un timeframe Alpaca (``1Day``) en intervalle yfinance (``1d``)."""
    try:
        return TIMEFRAME_TO_YFINANCE[timeframe]
    except KeyError as exc:
        raise DataError(f"timeframe non supporté: {timeframe!r}") from exc


def drop_incomplete_bar(
    bars: pd.DataFrame,
    timeframe: str,
    now: datetime,
    *,
    market_timezone: str = "America/New_York",
) -> BarFrame:
    """Retire la barre en cours de formation pour éviter tout lookahead.

    Pour les barres journalières, la barre du jour n'est conservée qu'après la
    clôture du marché (16:00 heure de New York). Pour les barres intraday, toute
    barre dont la fin est postérieure à ``now`` est retirée.

    Args:
        bars: Barres normalisées.
        timeframe: timeframe Alpaca de la barre.
        now: Instant courant, idéalement timezone-aware.
        market_timezone: Fuseau de la place de cotation.

    Returns:
        Barres complètes uniquement (copie).
    """
    if bars.empty:
        return bars
    if now.tzinfo is None:
        now = now.replace(tzinfo=bars.index.tz)

    if timeframe == "1Day":
        from zoneinfo import ZoneInfo

        local_now = now.astimezone(ZoneInfo(market_timezone))
        last_date = bars.index[-1].tz_convert(ZoneInfo(market_timezone)).date()
        if last_date == local_now.date() and local_now.hour < 16:
            return cast(BarFrame, bars.iloc[:-1].copy())
        return cast(BarFrame, bars.copy())

    duration = _BAR_DURATION.get(timeframe)
    if duration is None:
        raise DataError(f"timeframe non supporté: {timeframe!r}")
    bar_end = bars.index[-1] + duration
    return cast(BarFrame, (bars.iloc[:-1] if bar_end > now else bars).copy())


def lookback_window(
    timeframe: str,
    *,
    now: datetime,
    lookback_days: int,
) -> tuple[datetime, datetime]:
    """Calcule une fenêtre glissante adaptée au timeframe et aux indicateurs.

    Pour les timeframes intraday, la fenêtre est élargie afin de disposer d'au
    moins ``lookback_days`` séances de trading.
    """
    duration = _BAR_DURATION.get(timeframe)
    if duration is None:
        raise DataError(f"timeframe non supporté: {timeframe!r}")
    if timeframe == "1Day":
        span = timedelta(days=lookback_days)
    else:
        bars_per_day = max(1, math.floor(timedelta(days=1) / duration))
        # Marge de sécurité pour week-ends et jours fériés.
        span = max(
            timedelta(days=int(lookback_days * 1.6) + 5),
            timedelta(days=5),
        )
        LOGGER.debug("fenêtre intraday: %s jours, ~%s barres/jour", span.days, bars_per_day)
    return now - span, now


__all__ = [
    "INDICATOR_COLUMNS",
    "OHLCV_COLUMNS",
    "BarFrame",
    "CachedProvider",
    "DataError",
    "DataQualityError",
    "IndicatorBackend",
    "MarketDataProvider",
    "YFinanceProvider",
    "compute_atr",
    "compute_indicators",
    "drop_incomplete_bar",
    "lookback_window",
    "momentum_zscores",
    "normalize_ohlcv",
    "timeframe_interval",
    "trend_regime",
]
