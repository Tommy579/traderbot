"""Tests de la publication des statistiques dans le dépôt GitHub (aucun réseau)."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from typing import Any

import pytest

from trading_bot.notifications.github import GitHubStatsPublisher

NOW = datetime(2026, 11, 2, 21, 0, tzinfo=UTC)
REPO = "Tommy579/traderbot"
TOKEN = "ghp_token_de_test"


class FakeGitHub:
    """Transport simulé : file de réponses ``(statut, corps)`` et journal des appels."""

    def __init__(self, responses: list[tuple[int, bytes]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: Any,
        *,
        timeout_seconds: float,
    ) -> tuple[int, bytes]:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "body": json.loads(body) if body else None,
                "headers": dict(headers),
            }
        )
        if not self.responses:
            raise AssertionError("réponse non prévue par le test")
        return self.responses.pop(0)


def _file_response(entries: list[dict[str, Any]], sha: str = "sha-existant") -> tuple[int, bytes]:
    """Réponse de l'API Contents décrivant un fichier existant."""
    content = json.dumps({"updated_at": None, "entries": entries})
    document = {
        "sha": sha,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
    }
    return 200, json.dumps(document).encode("utf-8")


def _publisher(transport: FakeGitHub, **overrides: Any) -> GitHubStatsPublisher:
    kwargs: dict[str, Any] = {
        "repo": REPO,
        "token": TOKEN,
        "transport": transport,
        "clock": lambda: NOW,
    }
    kwargs.update(overrides)
    return GitHubStatsPublisher(**kwargs)


def _report(session: str = "2026-11-02", equity: float = 101_250.0) -> dict[str, Any]:
    return {"session_date": session, "skipped": False, "equity": equity}


def test_first_publication_creates_the_file() -> None:
    """404 à la lecture = première publication : PUT sans sha."""
    transport = FakeGitHub([(404, b"{}"), (201, b"{}")])
    assert _publisher(transport).send("cycle_trading", _report()) is True

    assert transport.calls[0]["method"] == "GET"
    assert transport.calls[0]["url"].endswith("/contents/docs/data/stats.json?ref=main")

    put = transport.calls[1]
    assert put["method"] == "PUT"
    assert "sha" not in put["body"], "inutile de fournir un sha pour une création"

    published = json.loads(base64.b64decode(put["body"]["content"]).decode("utf-8"))
    assert published["updated_at"] == NOW.isoformat()
    assert len(published["entries"]) == 1
    assert published["entries"][0]["report"]["session_date"] == "2026-11-02"
    assert put["body"]["message"] == "stats: séance 2026-11-02"


def test_update_sends_the_existing_sha() -> None:
    """L'API GitHub exige le sha courant pour remplacer un fichier existant."""
    transport = FakeGitHub(
        [
            _file_response([{"received_at": NOW.isoformat(), "report": _report("2026-10-30")}]),
            (200, b"{}"),
        ]
    )
    assert _publisher(transport).send("cycle_trading", _report("2026-11-02")) is True

    put = transport.calls[1]
    assert put["body"]["sha"] == "sha-existant"
    published = json.loads(base64.b64decode(put["body"]["content"]).decode("utf-8"))
    assert [e["report"]["session_date"] for e in published["entries"]] == [
        "2026-10-30",
        "2026-11-02",
    ]


def test_same_session_replaces_the_previous_entry() -> None:
    """Relancer un cycle le même jour ne doit pas créer de doublon."""
    transport = FakeGitHub(
        [
            _file_response([{"report": _report("2026-11-02", equity=100_000.0)}]),
            (200, b"{}"),
        ]
    )
    assert _publisher(transport).send("cycle_trading", _report("2026-11-02", 102_000.0)) is True

    published = json.loads(
        base64.b64decode(transport.calls[1]["body"]["content"]).decode("utf-8")
    )
    assert len(published["entries"]) == 1
    assert published["entries"][0]["report"]["equity"] == pytest.approx(102_000.0)


def test_read_failure_aborts_without_overwriting() -> None:
    """Mieux vaut ne rien publier que d'écraser un historique qu'on n'a pas lu."""
    transport = FakeGitHub([(500, b"boom")])
    assert _publisher(transport).send("cycle_trading", _report()) is False
    assert [c["method"] for c in transport.calls] == ["GET"]


def test_missing_content_field_aborts() -> None:
    """Fichier trop gros (> 1 Mo) : contenu absent, on ne touche à rien."""
    transport = FakeGitHub([(200, json.dumps({"sha": "abc"}).encode())])
    assert _publisher(transport).send("cycle_trading", _report()) is False
    assert [c["method"] for c in transport.calls] == ["GET"]


def test_corrupt_remote_content_is_rebuilt_not_fatal() -> None:
    """Un fichier distant corrompu ne doit pas empêcher toute publication."""
    broken = base64.b64encode(b"{ ceci n'est pas du JSON").decode("ascii")
    transport = FakeGitHub(
        [(200, json.dumps({"sha": "abc", "content": broken}).encode()), (200, b"{}")]
    )
    assert _publisher(transport).send("cycle_trading", _report()) is True
    published = json.loads(
        base64.b64decode(transport.calls[1]["body"]["content"]).decode("utf-8")
    )
    assert len(published["entries"]) == 1


def test_put_failure_never_raises() -> None:
    """Contrat best-effort : une publication ratée ne casse jamais un cycle."""
    transport = FakeGitHub([(404, b"{}"), (403, b"jeton sans les droits")])
    assert _publisher(transport).send("cycle_trading", _report()) is False


def test_transport_exception_is_swallowed() -> None:
    """Une panne réseau ne doit pas remonter jusqu'au cycle de trading."""

    def exploding(*args: Any, **kwargs: Any) -> tuple[int, bytes]:
        raise OSError("réseau indisponible")

    assert _publisher(exploding).send("cycle_trading", _report()) is False  # type: ignore[arg-type]


def test_entries_are_capped() -> None:
    """Le fichier publié ne doit pas croître indéfiniment."""
    existing = [{"report": {"session_date": f"2024-01-01-run{i}"}} for i in range(400)]
    transport = FakeGitHub([_file_response(existing), (200, b"{}")])
    assert _publisher(transport).send("cycle_trading", _report("2026-11-02")) is True

    published = json.loads(
        base64.b64decode(transport.calls[1]["body"]["content"]).decode("utf-8")
    )
    assert len(published["entries"]) == 400
    assert published["entries"][-1]["report"]["session_date"] == "2026-11-02"
    assert published["entries"][0]["report"]["session_date"] == "2024-01-01-run1"


def test_headers_carry_token_and_user_agent() -> None:
    """L'API GitHub refuse une requête sans User-Agent."""
    transport = FakeGitHub([(404, b"{}"), (201, b"{}")])
    _publisher(transport).send("cycle_trading", _report())
    headers = transport.calls[0]["headers"]
    assert headers["Authorization"] == f"Bearer {TOKEN}"
    assert headers["User-Agent"]
    assert headers["Accept"] == "application/vnd.github+json"


def test_repr_hides_the_token() -> None:
    transport = FakeGitHub([])
    assert TOKEN not in repr(_publisher(transport))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"repo": "sans-slash"}, "dépôt"),
        ({"repo": ""}, "dépôt"),
        ({"token": ""}, "jeton"),
        ({"path": "/absolu/stats.json"}, "chemin"),
    ],
)
def test_construction_rejects_invalid_settings(
    overrides: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _publisher(FakeGitHub([]), **overrides)
