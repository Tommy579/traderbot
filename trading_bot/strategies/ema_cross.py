"""Stratégie de référence : croisement EMA 20/50 filtré par le RSI.

Règles (long-only, aucune vente à découvert) :

* **BUY** — l'EMA rapide croise au-dessus de l'EMA lente *et* le RSI est dans la
  zone ``[rsi_entry_min, rsi_entry_max]`` (on évite d'acheter une extension
  déjà sur-achetée).
* **SELL** — l'EMA rapide repasse sous l'EMA lente.
* **HOLD** — tout le reste, y compris pendant la période de chauffe.

Les sorties de protection (stop-loss ATR, trailing) ne sont **pas** gérées ici :
elles appartiennent au moteur de risque, seul composant capable de liquider une
position. La stratégie ne voit donc jamais le prix d'entrée réel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trading_bot.data.loader import IndicatorBackend, compute_indicators
from trading_bot.strategies.base import SignalType, Strategy


class EmaCrossParams(BaseModel):
    """Paramètres figés de la stratégie EMA/RSI."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fast: int = Field(20, ge=2, le=200)
    slow: int = Field(50, ge=3, le=400)
    rsi_length: int = Field(14, ge=2, le=100)
    atr_length: int = Field(14, ge=2, le=100)
    rsi_entry_min: float = Field(50.0, ge=0.0, le=100.0)
    rsi_entry_max: float = Field(70.0, ge=0.0, le=100.0)
    indicator_backend: IndicatorBackend = "auto"

    @model_validator(mode="after")
    def _validate(self) -> EmaCrossParams:
        """Vérifie la cohérence des périodes et des bornes RSI."""
        if self.fast >= self.slow:
            raise ValueError("fast doit être strictement inférieur à slow")
        if self.rsi_entry_min >= self.rsi_entry_max:
            raise ValueError("rsi_entry_min doit être inférieur à rsi_entry_max")
        return self


class EmaCrossStrategy(Strategy):
    """Croisement d'EMA avec filtre RSI, calculé uniquement sur barres clôturées."""

    name = "ema_cross"

    def __init__(self, params: EmaCrossParams | None = None) -> None:
        self._params = params or EmaCrossParams()

    @property
    def params(self) -> EmaCrossParams:
        """Paramètres effectifs de la stratégie."""
        return self._params

    @property
    def warmup_bars(self) -> int:
        """Barres nécessaires : indicateurs + une barre pour le croisement."""
        return (
            max(
                self._params.slow,
                self._params.rsi_length + 1,
                self._params.atr_length + 1,
            )
            + 1
        )

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Calcule EMA rapide/lente, RSI et ATR sans modifier l'entrée."""
        return compute_indicators(
            bars,
            fast=self._params.fast,
            slow=self._params.slow,
            rsi_length=self._params.rsi_length,
            atr_length=self._params.atr_length,
            backend=self._params.indicator_backend,
        )

    def compute_signals(self, prepared: pd.DataFrame) -> pd.Series:
        """Traduit les croisements EMA en signaux, avec filtre RSI à l'achat."""
        ema_fast = prepared["ema_fast"]
        ema_slow = prepared["ema_slow"]
        rsi = prepared["rsi"]
        atr = prepared["atr"]
        previous_fast = ema_fast.shift(1)
        previous_slow = ema_slow.shift(1)

        cross_up = (ema_fast > ema_slow) & (previous_fast <= previous_slow)
        cross_down = (ema_fast < ema_slow) & (previous_fast >= previous_slow)
        rsi_window = (rsi >= self._params.rsi_entry_min) & (
            rsi <= self._params.rsi_entry_max
        )

        valid = (
            ema_fast.notna()
            & ema_slow.notna()
            & rsi.notna()
            & atr.notna()
            & previous_fast.notna()
            & previous_slow.notna()
            & (atr > 0.0)
        )

        signals = pd.Series(SignalType.HOLD, index=prepared.index, dtype=object)
        signals.loc[cross_up & rsi_window & valid] = SignalType.BUY
        signals.loc[cross_down & valid] = SignalType.SELL
        signals.loc[~valid] = SignalType.HOLD
        return signals

    def conviction(self, prepared: pd.DataFrame, action: SignalType) -> float:
        """Distance EMA normalisée par l'ATR, bornée à 1."""
        if action is SignalType.HOLD:
            return 0.0
        last = prepared.iloc[-1]
        atr = float(last.get("atr", np.nan))
        if not np.isfinite(atr) or atr <= 0.0:
            return 0.0
        spread = abs(float(last["ema_fast"]) - float(last["ema_slow"]))
        return float(np.clip(spread / atr, 0.0, 1.0))

    def explain(self, prepared: pd.DataFrame, action: SignalType) -> str:
        """Message lisible décrivant la règle déclenchée."""
        last = prepared.iloc[-1]
        fast = float(last["ema_fast"])
        slow = float(last["ema_slow"])
        rsi = float(last["rsi"])
        if action is SignalType.BUY:
            return (
                f"croisement haussier EMA{self._params.fast}/"
                f"EMA{self._params.slow} (RSI {rsi:.1f} dans "
                f"[{self._params.rsi_entry_min:.0f}, "
                f"{self._params.rsi_entry_max:.0f}])"
            )
        if action is SignalType.SELL:
            return (
                f"croisement baissier EMA{self._params.fast}/"
                f"EMA{self._params.slow} (EMA rapide {fast:.2f} < "
                f"EMA lente {slow:.2f})"
            )
        return f"aucun croisement exploitable (RSI {rsi:.1f})"

    def describe(self) -> dict[str, str]:
        """Paramètres journalisables, utiles à l'audit d'un backtest."""
        return {
            "strategy": self.name,
            "fast": str(self._params.fast),
            "slow": str(self._params.slow),
            "rsi_length": str(self._params.rsi_length),
            "rsi_entry_min": str(self._params.rsi_entry_min),
            "rsi_entry_max": str(self._params.rsi_entry_max),
            "warmup_bars": str(self.warmup_bars),
        }


__all__ = ["EmaCrossParams", "EmaCrossStrategy"]
