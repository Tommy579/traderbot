"""Stratégies d'analyse : production de signaux, sans accès au courtier."""

from __future__ import annotations

from trading_bot.strategies.base import Signal, SignalType, Strategy, StrategyError
from trading_bot.strategies.donchian_breakout import (
    DonchianBreakoutStrategy,
    DonchianParams,
)
from trading_bot.strategies.ema_cross import EmaCrossParams, EmaCrossStrategy
from trading_bot.strategies.relative_momentum import (
    RelativeMomentumParams,
    RelativeMomentumStrategy,
)
from trading_bot.strategies.shield_sword import (
    ShieldSwordParams,
    ShieldSwordStrategy,
)

__all__ = [
    "DonchianBreakoutStrategy",
    "DonchianParams",
    "EmaCrossParams",
    "EmaCrossStrategy",
    "RelativeMomentumParams",
    "RelativeMomentumStrategy",
    "ShieldSwordParams",
    "ShieldSwordStrategy",
    "Signal",
    "SignalType",
    "Strategy",
    "StrategyError",
]
