"""Tests de l'historique des cycles (aucun accès réseau)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from trading_bot.history import HistoryStore


def _report(session: str, *, equity: float = 100_000.0, **extra: Any) -> dict[str, Any]:
    """Rapport de cycle minimal, calqué sur ``CycleReport.to_dict()``."""
    return {
        "timestamp": f"{session}T19:55:00+00:00",
        "session_date": session,
        "skipped": False,
        "equity": equity,
        **extra,
    }


def test_missing_file_yields_no_entry(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.jsonl")
    assert store.entries() == []


def test_record_then_read_newest_first(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.jsonl")
    store.record(_report("2026-10-28"), source="timer")
    store.record(_report("2026-10-29"), source="timer")
    store.record(_report("2026-10-30"), source="gui")

    entries = store.entries()
    assert [entry["session_date"] for entry in entries] == [
        "2026-10-30",
        "2026-10-29",
        "2026-10-28",
    ]
    assert entries[0]["source"] == "gui"
    assert entries[0]["equity"] == pytest.approx(100_000.0)
    assert "recorded_at" in entries[0]


def test_same_session_is_replaced_not_appended(tmp_path: Path) -> None:
    """Une seule ligne par séance : le dernier passage fait foi."""
    store = HistoryStore(tmp_path / "history.jsonl")
    store.record(_report("2026-10-30", equity=100_000.0), source="gui")
    store.record(_report("2026-10-30", equity=101_500.0), source="timer")

    entries = store.entries()
    assert len(entries) == 1
    assert entries[0]["equity"] == pytest.approx(101_500.0)
    assert entries[0]["source"] == "timer"


def test_repeated_polling_keeps_one_line_per_session(tmp_path: Path) -> None:
    """La boucle de polling (60 s) ne doit pas noyer l'historique."""
    store = HistoryStore(tmp_path / "history.jsonl")
    for _ in range(50):
        store.record(_report("2026-10-30"), source="cli")
    assert len(store.entries()) == 1


def test_corrupt_line_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    store = HistoryStore(path)
    store.record(_report("2026-10-29"), source="timer")

    with path.open("a", encoding="utf-8") as handle:
        handle.write("{ ceci n'est pas du JSON\n")
        handle.write("42\n")  # JSON valide, mais pas un objet

    store.record(_report("2026-10-30"), source="timer")
    entries = store.entries()
    assert [entry["session_date"] for entry in entries] == [
        "2026-10-30",
        "2026-10-29",
    ]


def test_limit_keeps_most_recent(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.jsonl", limit=3)
    for day in range(1, 7):
        store.record(_report(f"2026-10-{day:02d}"), source="timer")

    assert [entry["session_date"] for entry in store.entries()] == [
        "2026-10-06",
        "2026-10-05",
        "2026-10-04",
    ]


def test_entries_respects_its_own_limit(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.jsonl")
    for day in range(1, 6):
        store.record(_report(f"2026-10-{day:02d}"), source="timer")

    assert len(store.entries(limit=2)) == 2
    assert store.entries(limit=0) == []


def test_file_stays_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    store = HistoryStore(path)
    store.record(_report("2026-10-30"), source="gui")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["session_date"] == "2026-10-30"
    assert payload["source"] == "gui"


def test_extra_report_fields_are_preserved(tmp_path: Path) -> None:
    """Le rapport complet (ordres, décisions, alertes) doit survivre au disque."""
    store = HistoryStore(tmp_path / "history.jsonl")
    report = _report(
        "2026-10-30",
        signals={"AAPL": "BUY"},
        orders=[{"symbol": "AAPL", "side": "buy", "qty": 200.0, "status": "filled"}],
        decisions=[{"approved": True, "symbol": "AAPL", "reason": "cassure"}],
        halts=[{"scope": "daily", "reason": "perte quotidienne", "equity": 97_000.0}],
        errors=["MSFT: données indisponibles"],
    )
    store.record(report, source="gui")

    entry = store.entries()[0]
    assert entry["signals"] == {"AAPL": "BUY"}
    assert entry["orders"][0]["status"] == "filled"
    assert entry["decisions"][0]["reason"] == "cassure"
    assert entry["halts"][0]["scope"] == "daily"
    assert entry["errors"] == ["MSFT: données indisponibles"]
