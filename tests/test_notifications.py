"""Tests du notificateur webhook (transport simulé, aucun réseau)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from trading_bot.notifications.webhook import WebhookNotifier

URL = "https://example.com/hook"


class FakeTransport:
    """Capture les appels POST ; peut simuler un échec réseau."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = fail

    def __call__(
        self,
        url: str,
        body: bytes,
        headers: dict[str, str],
        *,
        timeout_seconds: float,
    ) -> object:
        if self.fail:
            raise OSError("réseau indisponible")
        self.calls.append(
            {
                "url": url,
                "body": json.loads(body),
                "headers": headers,
                "timeout": timeout_seconds,
            }
        )
        return None


def test_disabled_without_url() -> None:
    transport = FakeTransport()
    notifier = WebhookNotifier(url=None, transport=transport)
    assert notifier.enabled is False
    assert notifier.send("cycle_trading", {"equity": 100.0}) is False
    assert transport.calls == []


def test_discord_format_wraps_payload_as_content() -> None:
    transport = FakeTransport()
    notifier = WebhookNotifier(url=URL, format="discord", transport=transport)
    assert notifier.enabled is True
    assert notifier.send("cycle_trading", {"equity": 100.0, "orders": []}) is True
    call = transport.calls[0]
    assert call["url"] == URL
    content = call["body"]["content"]
    assert content.startswith("**cycle_trading**")
    assert '"equity": 100.0' in content
    assert call["headers"]["Content-Type"] == "application/json"


def test_telegram_format_requires_chat_id() -> None:
    with pytest.raises(ValueError, match="chat_id"):
        WebhookNotifier(url=URL, format="telegram")
    transport = FakeTransport()
    notifier = WebhookNotifier(
        url=URL, format="telegram", chat_id="42", transport=transport
    )
    assert notifier.send("cycle_trading", {"equity": 100.0}) is True
    assert transport.calls[0]["body"]["chat_id"] == "42"
    assert "equity" in transport.calls[0]["body"]["text"]


def test_generic_format_posts_event_and_payload() -> None:
    transport = FakeTransport()
    notifier = WebhookNotifier(url=URL, format="generic", transport=transport)
    assert notifier.send("cycle_trading", {"equity": 100.0}) is True
    assert transport.calls[0]["body"] == {
        "event": "cycle_trading",
        "payload": {"equity": 100.0},
    }


def test_transport_failure_never_raises() -> None:
    notifier = WebhookNotifier(url=URL, transport=FakeTransport(fail=True))
    assert notifier.send("cycle_trading", {"equity": 100.0}) is False


def test_invalid_url_rejected() -> None:
    with pytest.raises(ValueError, match="URL"):
        WebhookNotifier(url="file:///etc/passwd")


def test_generic_format_sends_bearer_token() -> None:
    """L'endpoint public doit pouvoir vérifier qui lui parle."""
    transport = FakeTransport()
    notifier = WebhookNotifier(
        url=URL, format="generic", token="s3cret", transport=transport
    )
    assert notifier.send("cycle_trading", {"equity": 100.0}) is True
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer s3cret"


def test_no_authorization_header_without_token() -> None:
    """Discord et Telegram ne doivent pas recevoir d'en-tête parasite."""
    transport = FakeTransport()
    notifier = WebhookNotifier(url=URL, format="discord", transport=transport)
    assert notifier.send("cycle_trading", {"equity": 100.0}) is True
    assert "Authorization" not in transport.calls[0]["headers"]


def test_token_is_stripped_and_blank_means_absent() -> None:
    """Un jeton vide ou fait d'espaces ne doit pas produire d'en-tête."""
    for blank in ("", "   ", None):
        transport = FakeTransport()
        notifier = WebhookNotifier(url=URL, token=blank, transport=transport)
        notifier.send("cycle_trading", {"equity": 1.0})
        assert "Authorization" not in transport.calls[0]["headers"]

    transport = FakeTransport()
    notifier = WebhookNotifier(url=URL, token="  abc  ", transport=transport)
    notifier.send("cycle_trading", {"equity": 1.0})
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer abc"
