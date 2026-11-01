"""Contrat commun à toutes les stratégies.

Une stratégie est un composant **pur** : elle reçoit des barres OHLCV et rend un
:class:`Signal` (BUY / SELL / HOLD). Elle ne connaît ni le solde du compte, ni la
taille de position, ni le courtier — et ne peut donc pas passer d'ordre.

Les mêmes méthodes servent au backtest et au live, ce qui garantit que les
signaux observés en simulation sont exactement ceux exécutés en paper trading.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar

import numpy as np
import pandas as pd

from trading_bot.data.loader import OHLCV_COLUMNS


class StrategyError(RuntimeError):
    """Données incompatibles avec le contrat de la stratégie."""


class SignalType(StrEnum):
    """Décision d'analyse, sans aucune notion de taille ni d'exécution."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True, slots=True)
class Signal:
    """Signal immuable produit par une stratégie sur la dernière barre clôturée.

    Attributes:
        symbol: Ticker concerné.
        action: ``BUY``, ``SELL`` ou ``HOLD``.
        as_of: Horodatage UTC de la barre de décision.
        price: Clôture de la barre de décision (prix de référence, pas un prix
            d'exécution garanti).
        strength: Conviction normalisée dans ``[0, 1]`` (information de
            diagnostic ; le dimensionnement reste piloté par l'ATR).
        reason: Explication lisible du signal.
        indicators: Valeurs d'indicateurs ayant conduit à la décision.
    """

    symbol: str
    action: SignalType
    as_of: pd.Timestamp
    price: float
    strength: float = 0.0
    reason: str = ""
    indicators: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("un signal doit porter un symbole")
        if not math.isfinite(self.price) or self.price <= 0:
            raise ValueError(f"prix de signal invalide: {self.price!r}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength hors de [0, 1]: {self.strength!r}")
        if not isinstance(self.as_of, pd.Timestamp) or self.as_of.tzinfo is None:
            raise ValueError("as_of doit être un Timestamp timezone-aware (UTC)")


class Strategy(ABC):
    """Classe mère des stratégies : analyse uniquement, aucun accès au courtier.

    Le cycle de vie est volontairement en deux étapes :

    1. :meth:`prepare` ajoute les indicateurs aux barres (calcul vectorisé) ;
    2. :meth:`compute_signals` traduit les indicateurs en signaux.

    :meth:`generate_signals` enchaîne les deux et vérifie le contrat, tandis que
    :meth:`latest_signal` expose la dernière barre sous forme de :class:`Signal`.
    """

    name: ClassVar[str] = "strategy"
    # « atr » doit rester présent : le moteur de risque s'en sert pour
    # dimensionner et ancrer le stop de chaque entrée.
    indicator_columns: ClassVar[tuple[str, ...]] = ("ema_fast", "ema_slow", "rsi", "atr")
    # Benchmark exigé par la stratégie (``None`` : aucun). Le nom effectif est
    # celui de la configuration, injecté par l'orchestration : une stratégie ne
    # doit pas figer « SPY » dans son code, sinon ses messages désigneraient une
    # référence qui n'est plus celle réellement utilisée.
    benchmark_symbol: str | None = None

    @property
    @abstractmethod
    def warmup_bars(self) -> int:
        """Nombre de barres nécessaires avant le premier signal exploitable."""

    @abstractmethod
    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les colonnes d'indicateurs à une copie des barres."""

    @abstractmethod
    def compute_signals(self, prepared: pd.DataFrame) -> pd.Series:
        """Retourne une Series de :class:`SignalType` alignée sur ``prepared``."""

    def generate_signals(self, bars: pd.DataFrame, *, symbol: str = "") -> pd.DataFrame:
        """Valide les barres, calcule les indicateurs puis les signaux.

        Args:
            bars: Barres OHLCV normalisées, triées, sans doublons.
            symbol: Ticker (diagnostic uniquement).

        Returns:
            Copie des barres enrichie des indicateurs et d'une colonne
            ``signal`` de type :class:`SignalType`.

        Raises:
            StrategyError: schéma, tri, longueur ou valeurs de signal invalides.
        """
        self._validate_bars(bars)
        prepared = self.prepare(bars.copy())
        signals = self.compute_signals(prepared)
        if not isinstance(signals, pd.Series):
            raise StrategyError("compute_signals doit retourner une pandas.Series")
        if not signals.index.equals(prepared.index):
            raise StrategyError("les signaux ne sont pas alignés sur les barres")

        normalized: list[SignalType] = []
        for value in signals.to_numpy():
            try:
                normalized.append(SignalType(value))
            except ValueError as exc:
                raise StrategyError(f"signal inconnu: {value!r}") from exc

        prepared["signal"] = pd.Series(normalized, index=prepared.index, dtype=object)
        prepared.attrs["strategy"] = self.name
        prepared.attrs["symbol"] = symbol or bars.attrs.get("symbol", "")
        return prepared

    def latest_signal(self, bars: pd.DataFrame, *, symbol: str) -> Signal:
        """Construit le :class:`Signal` de la dernière barre clôturée."""
        enriched = self.generate_signals(bars, symbol=symbol)
        last = enriched.iloc[-1]
        action = SignalType(last["signal"])
        indicators: dict[str, float] = {}
        for column in self.indicator_columns:
            if column not in enriched.columns:
                continue
            value = float(last[column])
            if math.isfinite(value):
                indicators[column] = value

        return Signal(
            symbol=symbol,
            action=action,
            as_of=pd.Timestamp(enriched.index[-1]),
            price=float(last["close"]),
            strength=self.conviction(enriched, action),
            reason=self.explain(enriched, action),
            indicators=indicators,
        )

    def conviction(self, prepared: pd.DataFrame, action: SignalType) -> float:
        """Conviction par défaut : 1 pour un signal directionnel, 0 sinon.

        Les stratégies peuvent surcharger cette méthode pour exprimer une
        intensité (par exemple la distance EMA normalisée par l'ATR).
        """
        return 0.0 if action is SignalType.HOLD else 1.0

    def explain(self, prepared: pd.DataFrame, action: SignalType) -> str:
        """Explication lisible du signal, surchargeable par les stratégies."""
        return action.value

    def describe(self) -> dict[str, str]:
        """Paramètres journalisables de la stratégie."""
        return {"strategy": self.name, "warmup_bars": str(self.warmup_bars)}

    def _validate_bars(self, bars: pd.DataFrame) -> None:
        """Vérifie le contrat d'entrée (schéma, index, profondeur)."""
        if bars.empty:
            raise StrategyError("aucune barre fournie à la stratégie")
        missing = [column for column in OHLCV_COLUMNS if column not in bars.columns]
        if missing:
            raise StrategyError(f"colonnes manquantes: {missing}")
        if not isinstance(bars.index, pd.DatetimeIndex):
            raise StrategyError("l'index des barres doit être un DatetimeIndex")
        if bars.index.tz is None:
            raise StrategyError("l'index des barres doit être timezone-aware (UTC)")
        if not bars.index.is_monotonic_increasing:
            raise StrategyError("les barres doivent être triées par ordre croissant")
        if bars.index.has_duplicates:
            raise StrategyError("l'index des barres contient des doublons")
        if len(bars) <= self.warmup_bars:
            raise StrategyError(
                f"{len(bars)} barres fournies, {self.warmup_bars + 1} requises "
                f"pour la stratégie {self.name}"
            )
        if not np.isfinite(bars["close"].to_numpy(dtype=float)).all():
            raise StrategyError("les clôtures contiennent des valeurs non finies")


__all__ = ["Signal", "SignalType", "Strategy", "StrategyError"]
