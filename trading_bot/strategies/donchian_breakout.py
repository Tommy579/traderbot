"""Stratégie de cassure de canal Donchian (variante des Tortues).

Règles (long-only, aucune vente à découvert) :

* **BUY** — la clôture casse au-dessus du plus haut des ``entry_length``
  dernières barres *achevées* (le canal est décalé d'une barre : la clôture du
  jour ne peut pas se comparer à elle-même) *et* la clôture est au-dessus de la
  SMA longue : on n'achète une cassure que dans un régime de tendance.
* **SELL** — la clôture passe sous le plus bas des ``exit_length`` dernières
  barres achevées (canal de sortie plus serré que le canal d'entrée, pour
  laisser courir les tendances).
* **HOLD** — tout le reste, y compris pendant la période de chauffe.

Le stop initial (entrée - 2 ATR14) n'est **pas** calculé ici : il
appartient au moteur de risque, seul composant capable de dimensionner et
d'ancrer une position. La stratégie reste aveugle au compte et au courtier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trading_bot.data.loader import compute_atr
from trading_bot.strategies.base import SignalType, Strategy


class DonchianParams(BaseModel):
    """Paramètres figés de la stratégie Donchian."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry_length: int = Field(20, ge=2, le=200)
    exit_length: int = Field(10, ge=2, le=200)
    sma_length: int = Field(200, ge=5, le=500)
    atr_length: int = Field(14, ge=2, le=100)

    @model_validator(mode="after")
    def _validate(self) -> DonchianParams:
        """Le canal de sortie doit être au plus aussi large que l'entrée."""
        if self.exit_length > self.entry_length:
            raise ValueError("exit_length doit être <= entry_length")
        return self


class DonchianBreakoutStrategy(Strategy):
    """Cassure de canal Donchian filtrée par tendance, sur barres clôturées."""

    name = "donchian_breakout"
    indicator_columns = ("donchian_high", "donchian_low", "sma_long", "atr")

    def __init__(self, params: DonchianParams | None = None) -> None:
        self._params = params or DonchianParams()

    @property
    def params(self) -> DonchianParams:
        """Paramètres effectifs de la stratégie."""
        return self._params

    @property
    def warmup_bars(self) -> int:
        """Barres nécessaires : canaux, SMA longue, ATR et une barre de décision."""
        return (
            max(
                self._params.entry_length,
                self._params.exit_length,
                self._params.sma_length,
                self._params.atr_length + 1,
            )
            + 1
        )

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Calcule les canaux, la SMA longue et l'ATR sans modifier l'entrée."""
        params = self._params
        out = bars.copy()
        high = out["high"].astype(float)
        low = out["low"].astype(float)
        close = out["close"].astype(float)
        # shift(1) : le canal de décision n'utilise que des barres achevées.
        out["donchian_high"] = high.rolling(params.entry_length).max().shift(1)
        out["donchian_low"] = low.rolling(params.exit_length).min().shift(1)
        out["sma_long"] = close.rolling(params.sma_length).mean()
        out["atr"] = compute_atr(bars, length=params.atr_length)
        # Période de chauffe uniforme : aucun signal avant warmup_bars.
        warmup_index = out.index[: self.warmup_bars]
        out.loc[warmup_index, list(self.indicator_columns)] = np.nan
        return out

    def compute_signals(self, prepared: pd.DataFrame) -> pd.Series:
        """Traduit les cassures de canal en signaux, avec filtre de tendance."""
        close = prepared["close"]
        donchian_high = prepared["donchian_high"]
        donchian_low = prepared["donchian_low"]
        sma_long = prepared["sma_long"]
        atr = prepared["atr"]

        valid = (
            donchian_high.notna()
            & donchian_low.notna()
            & sma_long.notna()
            & atr.notna()
            & (atr > 0.0)
        )

        breakout_up = close > donchian_high
        breakout_down = close < donchian_low
        uptrend = close > sma_long

        signals = pd.Series(SignalType.HOLD, index=prepared.index, dtype=object)
        signals.loc[breakout_up & uptrend & valid] = SignalType.BUY
        signals.loc[breakout_down & valid] = SignalType.SELL
        signals.loc[~valid] = SignalType.HOLD
        return signals

    def conviction(self, prepared: pd.DataFrame, action: SignalType) -> float:
        """Ampleur de la cassure normalisée par l'ATR, bornée à 1."""
        if action is SignalType.HOLD:
            return 0.0
        last = prepared.iloc[-1]
        atr = float(last["atr"])
        if not np.isfinite(atr) or atr <= 0.0:
            return 0.0
        close = float(last["close"])
        if action is SignalType.BUY:
            spread = close - float(last["donchian_high"])
        else:
            spread = float(last["donchian_low"]) - close
        return float(np.clip(spread / atr, 0.0, 1.0))

    def explain(self, prepared: pd.DataFrame, action: SignalType) -> str:
        """Message lisible décrivant la règle déclenchée."""
        last = prepared.iloc[-1]
        close = float(last["close"])
        params = self._params
        if action is SignalType.BUY:
            return (
                f"cassure du canal haut {params.entry_length}j "
                f"(clôture {close:.2f} > {float(last['donchian_high']):.2f}), "
                f"tendance confirmée par SMA{params.sma_length} "
                f"{float(last['sma_long']):.2f}"
            )
        if action is SignalType.SELL:
            return (
                f"clôture {close:.2f} sous le canal bas {params.exit_length}j "
                f"({float(last['donchian_low']):.2f})"
            )
        return (
            f"consolidation entre canal bas {float(last['donchian_low']):.2f} "
            f"et canal haut {float(last['donchian_high']):.2f}"
        )

    def describe(self) -> dict[str, str]:
        """Paramètres journalisables, utiles à l'audit d'un backtest."""
        return {
            "strategy": self.name,
            "entry_length": str(self._params.entry_length),
            "exit_length": str(self._params.exit_length),
            "sma_length": str(self._params.sma_length),
            "atr_length": str(self._params.atr_length),
            "warmup_bars": str(self.warmup_bars),
        }


__all__ = ["DonchianBreakoutStrategy", "DonchianParams"]
