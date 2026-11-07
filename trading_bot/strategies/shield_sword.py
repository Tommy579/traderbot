"""Stratégie Bouclier + Épée (Core-Satellite CPPI).

Philosophie :
* 70 % du portefeuille en actifs ultra-défensifs sans risque (« Le Bouclier ») :
  Bons du Trésor US court terme (SGOV) ou rémunération au taux sans risque
  (4.5 % annuel) pour sanctuariser le capital avec 0 drawdown.
* 30 % répartis équitablement sur le Top 3 des actions en forte tendance (« L'Épée »),
  soit 10 % du portefeuille par action.
* Rebalancement lent (tous les 20 jours de bourse, soit 1 fois par mois) :
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
from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trading_bot.data.loader import compute_atr
from trading_bot.strategies.base import SignalType, Strategy, StrategyError


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
    risk_free_rate_annual: float = Field(0.045, ge=0.0, le=0.20)
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

    def set_start_date(self, start_date: datetime | date | str | None) -> None:
        """Fixe la date de début de simulation pour aligner les rebalancements."""
        if start_date is not None:
            ts = pd.Timestamp(start_date)
            self._start_date = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
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
            bench_sma = self._benchmark_close.rolling(self._params.benchmark_trend_sma).mean()
            b_bull = self._benchmark_close > bench_sma
            bench_bull = b_bull.to_dict()

        ref_index = (
            self._benchmark_close.index
            if self._benchmark_close is not None
            else sorted(set().union(*(s.index for s in scores.values())))
        )

        if len(ref_index) <= self.warmup_bars:
            self._top_by_date = {}
            self._rebalance_dates = set()
            return

        rebal_days = self._params.sword_rebalance_days
        start_ts = self._start_date

        if start_ts is not None:
            sub_index = [ts for ts in ref_index if ts >= start_ts]
            if not sub_index:
                sub_index = list(ref_index[self.warmup_bars :])
            rebal_dates_list = [sub_index[i] for i in range(0, len(sub_index), rebal_days)]
        else:
            rebal_dates_list = [
                ref_index[i] for i in range(self.warmup_bars, len(ref_index), rebal_days)
            ]

        rebal_dates = set(rebal_dates_list)
        self._top_by_date = {}

        for dt in rebal_dates_list:
            # Marché baissier sur SPY : zéro action offensive éligible (100 % Bouclier)
            if bench_bull and not bench_bull.get(dt, True):
                self._top_by_date[dt] = set()
                continue

            candidates: list[tuple[float, str]] = []
            for sym, sc in scores.items():
                if dt not in sc.index:
                    continue
                score_val = sc.loc[dt]
                close_val = closes[sym].loc[dt]
                sma_val = smas[sym].loc[dt]
                r60_val = ret60s[sym].loc[dt]

                if (
                    pd.notna(score_val)
                    and pd.notna(close_val)
                    and pd.notna(sma_val)
                    and pd.notna(r60_val)
                    and close_val > sma_val
                    and r60_val > 0.0
                    and score_val > 0.0
                ):
                    candidates.append((float(score_val), sym))

            candidates.sort(key=lambda item: item[0], reverse=True)
            top_syms = {item[1] for item in candidates[: self._params.sword_top_n]}
            self._top_by_date[dt] = top_syms

        self._rebalance_dates = rebal_dates

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Calcule SMA(100), score momentum composite, ATR et stop catastrophe."""
        out = bars.copy()
        close = out["close"].astype(float)
        out["sma_100"] = close.rolling(self._params.sword_trend_sma).mean()
        r60 = (close - close.shift(self._params.fast_mom)) / close.shift(self._params.fast_mom)
        r120 = (close - close.shift(self._params.slow_mom)) / close.shift(self._params.slow_mom)
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
        """Génère les signaux BUY / SELL / HOLD selon le calendrier de rebalancement mensuel."""
        sym = self._current_symbol
        close = prepared["close"].astype(float)
        sma = prepared["sma_100"]
        bench_bull = prepared.get("benchmark_bull")
        if bench_bull is None:
            bench_bull = pd.Series(True, index=prepared.index)

        signals = pd.Series(SignalType.HOLD, index=prepared.index, dtype=object)

        # Si le symbole fait partie du Bouclier (Safe Haven / SGOV), la stratégie offensive ne le trade pas
        if sym and (sym in self._params.shield_symbols or sym == self._params.shield_symbol):
            return signals

        top_by_date: dict[pd.Timestamp, set[str]] = getattr(self, "_top_by_date", {})
        rebal_dates: set[pd.Timestamp] = getattr(self, "_rebalance_dates", set())

        # Cas 1 : Mode isolé sans univers transversal (ex: test unitaire d'un seul ticker)
        if not top_by_date:
            composite = prepared["composite_mom"]
            ret60 = prepared.get("ret_60", pd.Series(1.0, index=prepared.index))
            held = False
            for i, _dt in enumerate(prepared.index):
                if i < self.warmup_bars:
                    continue
                is_rebal = (i - self.warmup_bars) % self._params.sword_rebalance_days == 0
                c_val = close.iloc[i]
                s_val = sma.iloc[i]
                b_val = bool(bench_bull.iloc[i])
                r_val = ret60.iloc[i]
                m_val = composite.iloc[i]

                uptrend = (
                    pd.notna(s_val)
                    and c_val > s_val
                    and b_val
                    and pd.notna(r_val)
                    and r_val > 0.0
                )

                if is_rebal:
                    if uptrend and pd.notna(m_val) and m_val > 0.0:
                        if not held:
                            signals.iloc[i] = SignalType.BUY
                            held = True
                        else:
                            signals.iloc[i] = SignalType.HOLD
                    else:
                        if held:
                            signals.iloc[i] = SignalType.SELL
                            held = False
                else:
                    # Entre deux rebalancements : STRICTEMENT HOLD
                    signals.iloc[i] = SignalType.HOLD
            return signals

        # Cas 2 : Mode multi-symboles complet avec sélection Top N
        held = False
        for i, dt in enumerate(prepared.index):
            if i < self.warmup_bars:
                continue

            if dt in rebal_dates:
                current_top = top_by_date.get(dt, set())
                if sym in current_top:
                    if not held:
                        signals.iloc[i] = SignalType.BUY
                        held = True
                    else:
                        signals.iloc[i] = SignalType.HOLD
                else:
                    if held:
                        signals.iloc[i] = SignalType.SELL
                        held = False
                    else:
                        signals.iloc[i] = SignalType.HOLD
            else:
                # Entre deux dates de rebalancement (jours 1 à 19) : STRICTEMENT HOLD
                signals.iloc[i] = SignalType.HOLD

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
            "risk_free_rate_annual": f"{self._params.risk_free_rate_annual:.1%}",
            "warmup_bars": str(self.warmup_bars),
        }


__all__ = ["ShieldSwordParams", "ShieldSwordStrategy"]
