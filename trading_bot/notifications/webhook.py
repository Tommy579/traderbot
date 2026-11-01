"""Envoi d'un récapitulatif JSON vers un webhook Discord ou Telegram.

Contrairement au reste du bot, cette couche a le droit d'échouer : une
notification manquante ne doit jamais interrompre un cycle de trading.
``send`` retourne donc un booléen et journalise les échecs au lieu de lever.
Transport par ``urllib`` de la bibliothèque standard : aucune dépendance
ajoutée, injectable pour les tests.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from collections.abc import Mapping
from typing import Any, Final, Protocol

from trading_bot.config import WebhookFormat

LOGGER = logging.getLogger(__name__)

_MAX_TEXT_CHARS: Final[int] = 1800
_JSON_HEADERS: Final[dict[str, str]] = {"Content-Type": "application/json"}


class Transport(Protocol):
    """POST d'un corps JSON ; lève en cas d'échec réseau ou HTTP."""

    def __call__(
        self,
        url: str,
        body: bytes,
        headers: Mapping[str, str],
        *,
        timeout_seconds: float,
    ) -> object: ...


def _urllib_transport(
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    *,
    timeout_seconds: float,
) -> object:
    """POST via urllib (bibliothèque standard uniquement)."""
    request = urllib.request.Request(url, data=body, headers=dict(headers))
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        response.read()
    return None


class WebhookNotifier:
    """Notificateur best-effort : jamais d'exception, au pire un avertissement.

    Args:
        url: URL du webhook. ``None`` désactive toutes les notifications.
        format: Dialecte du corps POST (``discord``, ``telegram``, ``generic``).
        chat_id: Conversation Telegram (requis pour ce format).
        token: Secret partagé, envoyé en ``Authorization: Bearer``. Requis dès
            que l'URL de réception est publique, sans quoi n'importe qui peut
            y injecter de fausses statistiques.
        timeout_seconds: Délai réseau par envoi.
        transport: POST injectable pour les tests (défaut : urllib).
    """

    def __init__(
        self,
        *,
        url: str | None,
        format: WebhookFormat = "discord",
        chat_id: str | None = None,
        token: str | None = None,
        timeout_seconds: float = 10.0,
        transport: Transport | None = None,
    ) -> None:
        if url is not None and not url.startswith(("https://", "http://")):
            raise ValueError(f"URL de webhook invalide: {url!r}")
        if format == "telegram" and url is not None and not chat_id:
            raise ValueError("le format telegram exige webhook_chat_id")
        self._url = url
        self._format: WebhookFormat = format
        self._chat_id = chat_id
        self._token = (token.strip() or None) if token else None
        self._timeout_seconds = timeout_seconds
        self._transport: Transport = transport or _urllib_transport

    @property
    def enabled(self) -> bool:
        """``True`` si un webhook est configuré."""
        return self._url is not None

    def _headers(self) -> dict[str, str]:
        """En-têtes du POST ; ajoute le jeton partagé quand il est configuré."""
        if self._token is None:
            return dict(_JSON_HEADERS)
        return {**_JSON_HEADERS, "Authorization": f"Bearer {self._token}"}

    def send(self, event: str, payload: Mapping[str, Any]) -> bool:
        """POST le récapitulatif ; retourne ``True`` seulement si l'envoi a abouti.

        Aucune exception ne sort de cette méthode : un échec est journalisé
        et rendu à l'appelant sous forme de ``False``.
        """
        if self._url is None:
            LOGGER.debug("webhook absent : notification '%s' ignorée", event)
            return False
        try:
            body = self._build_body(event, payload)
        except ValueError as exc:
            LOGGER.error("payload de notification invalide (%s): %s", event, exc)
            return False
        try:
            self._transport(
                self._url, body, self._headers(), timeout_seconds=self._timeout_seconds
            )
        except Exception as exc:  # noqa: BLE001 - best-effort par contrat
            LOGGER.warning("envoi de la notification '%s' échoué: %s", event, exc)
            return False
        LOGGER.info("notification '%s' envoyée (%s)", event, self._format)
        return True

    def _build_body(self, event: str, payload: Mapping[str, Any]) -> bytes:
        """Construit le corps POST au dialecte configuré."""
        text = json.dumps(payload, ensure_ascii=False, default=str)
        if self._format == "discord":
            # Discord plafonne ``content`` à 2000 caractères.
            content = f"**{event}**\n```json\n{text[:_MAX_TEXT_CHARS]}\n```"
            return json.dumps({"content": content}).encode("utf-8")
        if self._format == "telegram":
            if not self._chat_id:
                raise ValueError("chat_id Telegram absent")
            return json.dumps(
                {"chat_id": self._chat_id, "text": text[:_MAX_TEXT_CHARS]}
            ).encode("utf-8")
        return json.dumps({"event": event, "payload": payload}, default=str).encode("utf-8")


__all__ = ["WebhookNotifier"]
