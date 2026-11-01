"""Stratégies d'analyse : production de signaux, sans accès au courtier."""

from __future__ import annotations

from trading_bot.strategies.base import Signal, SignalType, Strategy, StrategyError
from trading_bot.strategies.ema_cross import EmaCrossParams, EmaCrossStrategy

__all__ = [
    "EmaCrossParams",
    "EmaCrossStrategy",
    "Signal",
    "SignalType",
    "Strategy",
    "StrategyError",
]
