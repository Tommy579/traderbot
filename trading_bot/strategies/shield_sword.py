"""Stratégie Bouclier + Épée (Core-Satellite CPPI).

Philosophie :
* 70 % du portefeuille dans l'actif du Bouclier (« Le Bouclier ») : un support
  défensif réel (SGOV, Bons du Trésor US court terme), acheté sur le marché et
  sans stop. Le cash nu ne rapporte rien chez le courtier paper : c'est le
  titre qui porte le rendement sans risque.
* 30 % répartis équitablement sur le Top 3 des actions en forte tendance (« L'Épée »),
  soit 10 % du portefeuille par action.
* Rebalancement lent (le premier jour de bourse du mois, soit ~20 séances) :
  - Sélection Top 3 sur SMA(100), rendement 60j > 0 et score momentum composite
    (0.6 * r60 + 0.4 * r120) / vol60.
  - Conservation stricte des 3 lignes en HOLD entre deux rebalancements.
  - Si une action reste dans le Top 3 au rebalancement : conservée en HOLD (0 frais).
  - Stop-Loss catastrophe ultra-large (5.0 * ATR ou perte max -12 % sur la ligne)
    pour ne jamais être éjecté par le bruit quotidien.
  - Si SPY < SMA(200) au rebalancement : 100 % sur le Bouclier sans risque.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import ClassVar, Final

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trading_bot.data.loader import compute_atr
from trading_bot.strategies.base import SignalType, Strategy, StrategyError

#: Séances de bourse par mois, pour convertir ``sword_rebalance_days`` en mois.
_SESSIONS_PER_MONTH: Final[float] = 21.0


class ShieldSwordParams(BaseModel):
    """Paramètres configurables de la stratégie Bouclier + Épée."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    shield_allocation_pct: float = Field(0.70, ge=0.0, le=1.0)
    sword_allocation_pct: float = Field(0.30, ge=0.0, le=1.0)
    sword_top_n: int = Field(3, ge=1, le=20)
    sword_rebalance_days: int = Field(20, ge=1, le=250)
    sword_trend_sma: int = Field(100, ge=10, le=500)
    benchmark_trend_sma: int = Field(200, ge=10, le=500)
    fast_mom: int = Field(60, ge=5, le=250)
    slow_mom: int = Field(120, ge=10, le=500)
    vol_length: int = Field(60, ge=10, le=250)
    atr_length: int = Field(14, ge=2, le=100)
    sword_stop_atr_multiplier: float = Field(5.0, ge=1.0, le=20.0)
    sword_max_loss_pct: float = Field(0.12, ge=0.01, le=0.50)
    shield_symbol: str = "SGOV"
    shield_symbols: tuple[str, ...] = ("SGOV", "BRK.B", "WMT", "KO")

    @model_validator(mode="after")
    def _validate(self) -> ShieldSwordParams:
        if self.fast_mom >= self.slow_mom:
            raise ValueError("fast_mom doit être strictement inférieur à slow_mom")
        if self.shield_allocation_pct + self.sword_allocation_pct > 1.0 + 1e-6:
            raise ValueError("l'allocation bouclier + épée ne peut pas dépasser 100 %")
        return self


class ShieldSwordStrategy(Strategy):
    """Stratégie Core-Satellite CPPI : 70 % Bouclier défensif + 30 % Épée momentum mensuelle."""

    name: ClassVar[str] = "shield_sword"
    benchmark_symbol: str | None = "SPY"
    indicator_columns: ClassVar[tuple[str, ...]] = (
        "sma_100",
        "composite_mom",
        "atr",
        "proposed_stop",
    )

    def __init__(
        self,
        params: ShieldSwordParams | None = None,
        *,
        benchmark: pd.DataFrame | None = None,
    ) -> None:
        self._params = params or ShieldSwordParams()
        self._benchmark_close: pd.Series | None = None
        self._universe: dict[str, pd.DataFrame] = {}
        self._top_by_date: dict[pd.Timestamp, set[str]] = {}
        self._rebalance_dates: set[pd.Timestamp] = set()
        self._start_date: pd.Timestamp | None = None
        self._current_symbol: str = ""
        if benchmark is not None:
            self.set_benchmark(benchmark)

    @property
    def params(self) -> ShieldSwordParams:
        """Paramètres effectifs de la stratégie."""
        return self._params

    @property
    def warmup_bars(self) -> int:
        """Barres nécessaires pour stabiliser SMA(100), momentum 120j et SPY SMA(200)."""
        return (
            max(
                self._params.benchmark_trend_sma,
                self._params.sword_trend_sma,
                self._params.slow_mom + 1,
                self._params.vol_length + 2,
                self._params.atr_length + 1,
            )
            + 1
        )

    @property
    def rebalance_period_months(self) -> int:
        """Nombre de mois entre deux rebalancements, dérivé de ``sword_rebalance_days``.

        Le rebalancement est ancré sur le calendrier (premier jour de bourse du
        mois) et non sur la fenêtre de barres fournie : c'est la seule façon
        qu'une même date soit un rebalancement en backtest **et** en live, où la
        fenêtre glisse à chaque cycle. ``sword_rebalance_days`` (20 séances par
        défaut, soit un mois) est converti en mois ; en dessous d'un mois, la
        périodicité retombe sur mensuel.
        """
        return max(1, round(self._params.sword_rebalance_days / _SESSIONS_PER_MONTH))

    def _rebalance_mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Vrai pour le premier jour de bourse d'un mois de rebalancement.

        Le premier point de l'index n'est jamais retenu : sans barre
        précédente, rien ne prouve qu'il ouvre un mois (une fenêtre glissante
        peut commencer en milieu de mois).
        """
        months = np.asarray(index.year) * 12 + np.asarray(index.month)
        mask = np.zeros(len(index), dtype=bool)
        if len(index) > 1:
            mask[1:] = months[1:] != months[:-1]
        period = self.rebalance_period_months
        if period > 1:
            mask &= months % period == 0
        return mask

    @staticmethod
    def _is_eligible(
        close: float,
        sma: float,
        benchmark_bull: bool,
        ret_fast: float,
        composite: float,
    ) -> bool:
        """Tendance haussière, momentum positif et marché porteur : critères d'entrée."""
        if not benchmark_bull:
            return False
        if not (
            pd.notna(close) and pd.notna(sma) and pd.notna(ret_fast) and pd.notna(composite)
        ):
            return False
        return bool(close > sma and ret_fast > 0.0 and composite > 0.0)

    def set_start_date(self, start_date: datetime | date | str | None) -> None:
        """Fixe la date de début de simulation pour aligner les rebalancements."""
        if start_date is not None:
            ts = pd.Timestamp(start_date)
            self._start_date = (
                ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            )
        else:
            self._start_date = None
        if self._universe:
            self._precompute_rankings()

    def set_benchmark(self, bars: pd.DataFrame) -> None:
        """Injecte les barres du benchmark (contrat identique aux barres symbole)."""
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
        if self._universe:
            self._precompute_rankings()

    def set_universe(self, universe: Mapping[str, pd.DataFrame]) -> None:
        """Enregistre l'ensemble des données de l'univers pour le classement transversal."""
        self._universe = {k: v.copy() for k, v in universe.items()}
        self._precompute_rankings()

    def _precompute_rankings(self) -> None:
        """Précalcule le classement transversal Top N sur chaque date de rebalancement."""
        if not self._universe:
            self._top_by_date = {}
            self._rebalance_dates = set()
            return

        scores: dict[str, pd.Series] = {}
        smas: dict[str, pd.Series] = {}
        closes: dict[str, pd.Series] = {}
        ret60s: dict[str, pd.Series] = {}

        shield_set = set(self._params.shield_symbols)
        for sym, df in self._universe.items():
            if sym in shield_set or sym == self.benchmark_symbol:
                continue
            if "close" not in df.columns or len(df) <= self.warmup_bars:
                continue
            c = df["close"].astype(float)
            closes[sym] = c
            smas[sym] = c.rolling(self._params.sword_trend_sma).mean()
            r60 = (c - c.shift(self._params.fast_mom)) / c.shift(self._params.fast_mom)
            r120 = (c - c.shift(self._params.slow_mom)) / c.shift(self._params.slow_mom)
            daily_ret = c.pct_change()
            v60 = daily_ret.rolling(self._params.vol_length).std()
            scores[sym] = (0.6 * r60 + 0.4 * r120) / v60
            ret60s[sym] = r60

        if not scores:
            self._top_by_date = {}
            self._rebalance_dates = set()
            return

        bench_bull: dict[pd.Timestamp, bool] = {}
        if (
            self._benchmark_close is not None
            and len(self._benchmark_close) >= self._params.benchmark_trend_sma
        ):
            bench_sma = self._benchmark_close.rolling(
                self._params.benchmark_trend_sma
            ).mean()
            b_bull = self._benchmark_close > bench_sma
            bench_bull = b_bull.to_dict()

        ref_index = pd.DatetimeIndex(
            self._benchmark_close.index
            if self._benchmark_close is not None
            else sorted(set().union(*(s.index for s in scores.values())))
        )

        if len(ref_index) <= self.warmup_bars:
            self._top_by_date = {}
            self._rebalance_dates = set()
            return

        # Dates de rebalancement : ancrées sur le calendrier, donc identiques
        # quelle que soit la profondeur de la fenêtre fournie (live compris).
        start_ts = self._start_date
        rebal_dates_list = [
            stamp
            for stamp, is_rebalance in zip(
                ref_index, self._rebalance_mask(ref_index), strict=True
            )
            if is_rebalance and (start_ts is None or stamp >= start_ts)
        ]

        rebal_dates = set(rebal_dates_list)
        self._top_by_date = {}

        for dt in rebal_dates_list:
            # Marché baissier sur SPY : zéro action offensive éligible (100 % Bouclier)
            if bench_bull and not bench_bull.get(dt, False):
                self._top_by_date[dt] = set()
                continue

            candidates: list[tuple[float, str]] = []
            for sym, sc in scores.items():
                if dt not in sc.index:
                    continue
                if self._is_eligible(
                    float(closes[sym].loc[dt]),
                    float(smas[sym].loc[dt]),
                    True,
                    float(ret60s[sym].loc[dt]),
                    float(sc.loc[dt]),
                ):
                    candidates.append((float(sc.loc[dt]), sym))

            candidates.sort(key=lambda item: item[0], reverse=True)
            top_syms = {item[1] for item in candidates[: self._params.sword_top_n]}
            self._top_by_date[dt] = top_syms

        self._rebalance_dates = rebal_dates

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Calcule SMA(100), score momentum composite, ATR et stop catastrophe."""
        out = bars.copy()
        close = out["close"].astype(float)
        out["sma_100"] = close.rolling(self._params.sword_trend_sma).mean()
        r60 = (close - close.shift(self._params.fast_mom)) / close.shift(
            self._params.fast_mom
        )
        r120 = (close - close.shift(self._params.slow_mom)) / close.shift(
            self._params.slow_mom
        )
        v60 = close.pct_change().rolling(self._params.vol_length).std()
        out["composite_mom"] = (0.6 * r60 + 0.4 * r120) / v60
        out["ret_60"] = r60
        atr = compute_atr(bars, length=self._params.atr_length)
        out["atr"] = atr

        stop_atr = close - self._params.sword_stop_atr_multiplier * atr
        stop_loss = close * (1.0 - self._params.sword_max_loss_pct)
        out["proposed_stop"] = np.maximum(stop_atr, stop_loss)

        if (
            self._benchmark_close is not None
            and len(self._benchmark_close) >= self._params.benchmark_trend_sma
        ):
            bench_c = self._benchmark_close.reindex(out.index, method="ffill")
            bench_sma = bench_c.rolling(self._params.benchmark_trend_sma).mean()
            out["benchmark_bull"] = (bench_c > bench_sma).fillna(False)
        else:
            out["benchmark_bull"] = True

        warmup_idx = out.index[: self.warmup_bars]
        out.loc[warmup_idx, list(self.indicator_columns)] = np.nan
        return out

    def generate_signals(self, bars: pd.DataFrame, *, symbol: str = "") -> pd.DataFrame:
        """Mémorise le symbole actif avant d'exécuter la génération vectorisée."""
        self._current_symbol = symbol or str(bars.attrs.get("symbol", ""))
        return super().generate_signals(bars, symbol=symbol)

    def compute_signals(self, prepared: pd.DataFrame) -> pd.Series:
        """Génère BUY / SELL / HOLD selon le calendrier de rebalancement.

        La règle est **sans état** : à chaque date de rebalancement, un actif
        retenu dans le Top N reçoit BUY, les autres SELL. Supprimer un BUY au
        motif qu'on détenait déjà la ligne ferait diverger le live du backtest :
        le portefeuille réel peut très bien ne pas la détenir (cycle manqué,
        position vendue par un stop, bot démarré plus tard). En backtest comme
        en live, le moteur de risque rejette un BUY sur une position déjà
        ouverte — c'est lui qui garantit l'absence de doublon, pas la stratégie.
        """
        sym = self._current_symbol
        signals = pd.Series(SignalType.HOLD, index=prepared.index, dtype=object)

        # Si le symbole fait partie du Bouclier (Safe Haven / SGOV), la stratégie offensive ne le trade pas
        if sym and (
            sym in self._params.shield_symbols or sym == self._params.shield_symbol
        ):
            return signals

        close = prepared["close"].astype(float)
        sma = prepared["sma_100"]
        bench_bull = prepared.get("benchmark_bull")
        if bench_bull is None:
            bench_bull = pd.Series(True, index=prepared.index)
        composite = prepared["composite_mom"]
        ret_fast = prepared.get("ret_60", pd.Series(1.0, index=prepared.index))

        for position in np.flatnonzero(self._rebalance_mask(prepared.index)):
            if position < self.warmup_bars:
                continue
            stamp = prepared.index[position]
            if self._top_by_date:
                selected = sym in self._top_by_date.get(stamp, set())
            else:
                # Mode isolé (un seul ticker, pas de classement transversal) :
                # mêmes critères d'éligibilité que le classement Top N.
                selected = self._is_eligible(
                    float(close.iloc[position]),
                    float(sma.iloc[position]),
                    bool(bench_bull.iloc[position]),
                    float(ret_fast.iloc[position]),
                    float(composite.iloc[position]),
                )
            signals.iloc[position] = SignalType.BUY if selected else SignalType.SELL

        return signals

    def conviction(self, prepared: pd.DataFrame, action: SignalType) -> float:
        """Conviction à 1 si BUY, 0 sinon."""
        return 1.0 if action is SignalType.BUY else 0.0

    def explain(self, prepared: pd.DataFrame, action: SignalType) -> str:
        """Explication lisible du signal produit."""
        sym = self._current_symbol
        top_n = self._params.sword_top_n
        sma_p = self._params.sword_trend_sma
        if action is SignalType.BUY:
            return (
                f"Bouclier+Épée : {sym} sélectionné dans le Top {top_n} "
                f"au rebalancement (tendance > SMA{sma_p}, r60j > 0 et SPY haussier)"
            )
        if action is SignalType.SELL:
            return (
                f"Bouclier+Épée : sortie de {sym} "
                f"(sorti du Top {top_n} lors du rebalancement mensuel)"
            )
        return f"Bouclier+Épée : {sym} maintenu (conservation stricte entre rebalancements)"

    def describe(self) -> dict[str, str]:
        """Description journalisable des paramètres de la stratégie."""
        return {
            "strategy": self.name,
            "shield_allocation": f"{self._params.shield_allocation_pct:.0%}",
            "sword_allocation": f"{self._params.sword_allocation_pct:.0%}",
            "sword_top_n": str(self._params.sword_top_n),
            "rebalance_days": str(self._params.sword_rebalance_days),
            "trend_sma": str(self._params.sword_trend_sma),
            "benchmark_trend_sma": str(self._params.benchmark_trend_sma),
            "stop_atr_multiplier": str(self._params.sword_stop_atr_multiplier),
            "max_loss_pct": f"{self._params.sword_max_loss_pct:.0%}",
            "warmup_bars": str(self.warmup_bars),
        }


__all__ = ["ShieldSwordParams", "ShieldSwordStrategy"]
