"""Bot de trading systématique — paper trading Alpaca uniquement.

Architecture en quatre couches strictement découplées :

``data`` → ``strategies`` → ``risk`` → ``execution``

La stratégie ne connaît ni le courtier ni la taille de position : elle produit
un :class:`trading_bot.strategies.base.Signal`. Seul
:class:`trading_bot.risk.engine.RiskEngine` peut convertir ce signal en
:class:`trading_bot.risk.engine.OrderIntent` signé, seul type accepté par
:class:`trading_bot.execution.alpaca_client.AlpacaPaperClient`.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
