"""Stratégie de momentum relatif face à un benchmark (SPY par défaut).

Règles (long-only, aucune vente à découvert) :

* **BUY** — le momentum de l'actif sur ``fast_mom`` et ``slow_mom`` jours,
  normalisé par la volatilité (z-score de type t-stat), est positif **et**
  supérieur à celui du benchmark sur les deux horizons : on n'achète que la
  force relative, pas simplement une hausse.
* **SELL** — le momentum court devient négatif ou perd sa supériorité sur le
  benchmark : la tendance relative est cassée.
* **HOLD** — tout le reste, y compris pendant la période de chauffe.

Le benchmark est **injecté** (:meth:`set_benchmark`) par l'orchestration :
la stratégie reste aveugle au compte et au courtier, mais exige ses données de
référence pour produire un signal. Sans benchmark, elle échoue bruyamment —
jamais de signal silencieusement différent entre backtest et live.

Normalisation : ``z(h) = (close / close_h - 1) / (sigma_journalier * sqrt(h))`` où
``sigma_journalier`` est l'écart-type des rendements sur ``vol_length`` barres.
C'est le momentum par unité de risque, comparable entre actifs et horizons.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trading_bot.data.loader import compute_atr, momentum_zscores
from trading_bot.strategies.base import SignalType, Strategy, StrategyError


class RelativeMomentumParams(BaseModel):
    """Paramètres figés de la stratégie de momentum relatif."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fast_mom: int = Field(60, ge=5, le=250)
    slow_mom: int = Field(120, ge=10, le=500)
    vol_length: int = Field(60, ge=10, le=250)
    atr_length: int = Field(14, ge=2, le=100)

    @model_validator(mode="after")
    def _validate(self) -> RelativeMomentumParams:
        """L'horizon rapide doit précéder l'horizon lent."""
        if self.fast_mom >= self.slow_mom:
            raise ValueError("fast_mom doit être strictement inférieur à slow_mom")
        return self


def _momentum_zscores(
    close: pd.Series,
    params: RelativeMomentumParams,
) -> tuple[pd.Series, pd.Series]:
    """Délègue au loader : même normalisation que le classement de risque."""
    return momentum_zscores(
        close,
        fast_length=params.fast_mom,
        slow_length=params.slow_mom,
        vol_length=params.vol_length,
    )


class RelativeMomentumStrategy(Strategy):
    """Momentum normalisé par la volatilité, comparé au benchmark injecté."""

    name = "relative_momentum"
    # Valeur par défaut : le nom effectif est injecté depuis BENCHMARK_SYMBOL.
    benchmark_symbol: str | None = "SPY"
    indicator_columns = (
        "mom_fast",
        "mom_slow",
        "benchmark_mom_fast",
        "benchmark_mom_slow",
        "atr",
    )

    def __init__(
        self,
        params: RelativeMomentumParams | None = None,
        *,
        benchmark: pd.DataFrame | None = None,
    ) -> None:
        self._params = params or RelativeMomentumParams()
        self._benchmark_close: pd.Series | None = None
        if benchmark is not None:
            self.set_benchmark(benchmark)

    @property
    def params(self) -> RelativeMomentumParams:
        """Paramètres effectifs de la stratégie."""
        return self._params

    @property
    def warmup_bars(self) -> int:
        """Barres nécessaires : momentum lent, volatilité, ATR, une barre de décision."""
        return (
            max(
                self._params.slow_mom + 1,
                self._params.vol_length + 2,
                self._params.atr_length + 1,
            )
            + 1
        )

    def set_benchmark(self, bars: pd.DataFrame) -> None:
        """Injecte les barres du benchmark (contrat identique aux barres symbole).

        Raises:
            StrategyError: données vides, schéma ou index invalide.
        """
        if bars is None or bars.empty:
            raise StrategyError("benchmark vide")
        if "close" not in bars.columns:
            raise StrategyError("benchmark sans colonne 'close'")
        index = bars.index
        if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
            raise StrategyError("l'index du benchmark doit être timezone-aware (UTC)")
        if not index.is_monotonic_increasing or index.has_duplicates:
            raise StrategyError("l'index du benchmark doit être trié et sans doublons")
        close = bars["close"].astype(float)
        if not np.isfinite(close.to_numpy()).all() or (close <= 0).any():
            raise StrategyError("clôtures du benchmark invalides")
        self._benchmark_close = close.copy()

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Calcule les z-scores de l'actif et du benchmark aligné sur ses dates."""
        if self._benchmark_close is None:
            raise StrategyError(
                f"benchmark {self.benchmark_symbol} requis : appelez set_benchmark "
                "avant de générer les signaux"
            )
        params = self._params
        out = bars.copy()
        z_fast, z_slow = _momentum_zscores(out["close"], params)
        out["mom_fast"] = z_fast
        out["mom_slow"] = z_slow

        bench_fast, bench_slow = _momentum_zscores(self._benchmark_close, params)
        # Alignement « au plus tard » : la valeur du benchmark est celle de sa
        # dernière barre achevée à la date de décision (aucun lookahead).
        out["benchmark_mom_fast"] = bench_fast.reindex(out.index, method="ffill")
        out["benchmark_mom_slow"] = bench_slow.reindex(out.index, method="ffill")
        out["atr"] = compute_atr(bars, length=params.atr_length)
        # Période de chauffe uniforme : aucun signal avant warmup_bars.
        warmup_index = out.index[: self.warmup_bars]
        out.loc[warmup_index, list(self.indicator_columns)] = np.nan
        return out

    def compute_signals(self, prepared: pd.DataFrame) -> pd.Series:
        """Traduit la force relative en signaux."""
        mom_fast = prepared["mom_fast"]
        mom_slow = prepared["mom_slow"]
        bench_fast = prepared["benchmark_mom_fast"]
        bench_slow = prepared["benchmark_mom_slow"]
        atr = prepared["atr"]

        valid = (
            mom_fast.notna()
            & mom_slow.notna()
            & bench_fast.notna()
            & bench_slow.notna()
            & atr.notna()
            & (atr > 0.0)
        )

        strong_fast = (mom_fast > 0.0) & (mom_fast > bench_fast)
        strong_slow = (mom_slow > 0.0) & (mom_slow > bench_slow)
        broken_fast = (mom_fast < 0.0) | (mom_fast < bench_fast)

        signals = pd.Series(SignalType.HOLD, index=prepared.index, dtype=object)
        signals.loc[strong_fast & strong_slow & valid] = SignalType.BUY
        signals.loc[broken_fast & valid] = SignalType.SELL
        signals.loc[~valid] = SignalType.HOLD
        return signals

    def conviction(self, prepared: pd.DataFrame, action: SignalType) -> float:
        """Supériorité du momentum court sur le benchmark, bornée à 1."""
        if action is SignalType.HOLD:
            return 0.0
        last = prepared.iloc[-1]
        spread = float(last["mom_fast"]) - float(last["benchmark_mom_fast"])
        if not math.isfinite(spread):
            return 0.0
        return max(0.0, min(spread, 1.0))

    def explain(self, prepared: pd.DataFrame, action: SignalType) -> str:
        """Message lisible décrivant la force relative."""
        last = prepared.iloc[-1]
        params = self._params
        label = self.benchmark_symbol or "benchmark"
        if action is SignalType.BUY:
            return (
                f"momentum {params.fast_mom}j {float(last['mom_fast']):+.2f} > "
                f"{label} {float(last['benchmark_mom_fast']):+.2f} et "
                f"{params.slow_mom}j {float(last['mom_slow']):+.2f} > "
                f"{label} {float(last['benchmark_mom_slow']):+.2f}"
            )
        if action is SignalType.SELL:
            return (
                f"momentum {params.fast_mom}j {float(last['mom_fast']):+.2f} cassé "
                f"(négatif ou sous {label} "
                f"{float(last['benchmark_mom_fast']):+.2f})"
            )
        return (
            f"momentum {params.fast_mom}j {float(last['mom_fast']):+.2f} vs "
            f"{label} {float(last['benchmark_mom_fast']):+.2f} : pas de force "
            "relative exploitable"
        )

    def describe(self) -> dict[str, str]:
        """Paramètres journalisables, utiles à l'audit d'un backtest."""
        return {
            "strategy": self.name,
            "benchmark": self.benchmark_symbol or "",
            "fast_mom": str(self._params.fast_mom),
            "slow_mom": str(self._params.slow_mom),
            "vol_length": str(self._params.vol_length),
            "atr_length": str(self._params.atr_length),
            "warmup_bars": str(self.warmup_bars),
        }


__all__ = ["RelativeMomentumParams", "RelativeMomentumStrategy"]
