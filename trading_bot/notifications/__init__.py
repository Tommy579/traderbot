"""Notifications externes du bot (webhook et publication GitHub)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from trading_bot.notifications.github import GitHubStatsPublisher
from trading_bot.notifications.webhook import WebhookNotifier


class Notifier(Protocol):
    """Canal de notification : ``send`` ne lève jamais, il rend un booléen."""

    @property
    def enabled(self) -> bool: ...

    def send(self, event: str, payload: Mapping[str, Any]) -> bool: ...


class FanoutNotifier:
    """Diffuse chaque récapitulatif à plusieurs canaux indépendants.

    L'échec de l'un n'empêche pas les autres, et aucun ne peut interrompre un
    cycle de trading : c'est le même contrat best-effort que chaque canal.

    Args:
        targets: Canaux à alimenter, dans l'ordre.
    """

    def __init__(self, targets: Sequence[Notifier]) -> None:
        self._targets = list(targets)

    def __repr__(self) -> str:
        """Représentation sans secret."""
        return f"FanoutNotifier({len(self._targets)} canal(aux))"

    @property
    def enabled(self) -> bool:
        """``True`` si au moins un canal est actif."""
        return any(target.enabled for target in self._targets)

    def send(self, event: str, payload: Mapping[str, Any]) -> bool:
        """Envoie à tous les canaux ; ``True`` si au moins un a abouti.

        Args:
            event: Nom de l'événement (``cycle_trading``).
            payload: Rapport de cycle.

        Returns:
            ``True`` si au moins un canal a accepté le récapitulatif.
        """
        results = [target.send(event, payload) for target in self._targets]
        return any(results)


__all__ = ["FanoutNotifier", "GitHubStatsPublisher", "Notifier", "WebhookNotifier"]
