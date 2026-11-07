"""Historique des cycles du bot, consultable depuis l'interface web.

Le fichier est un journal JSON Lines où **un enregistrement correspond à une
séance de marché**. Le dernier passage d'une même séance remplace le
précédent : sans cette règle, une boucle de polling (un cycle toutes les 60 s)
noyerait l'historique sous des centaines de lignes identiques.

Le service systemd et l'interface web peuvent écrire le même fichier en même
temps (cycle du timer pendant qu'un cycle est lancé depuis la page) : le
lecture-modification-écriture est donc protégé par un verrou ``flock``.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Final, TextIO

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows n'expose pas fcntl
    fcntl = None  # type: ignore[assignment]

from trading_bot.clock import get_online_now

_MAX_ENTRIES: Final[int] = 400
"""Enregistrements conservés : environ un an et demi de séances."""

_THREAD_LOCK = threading.Lock()


def _utc_now() -> datetime:
    """Horloge par défaut : maintenant en UTC, aligné sur l'heure en ligne."""
    return get_online_now()


class HistoryStore:
    """Journal JSON Lines des rapports de cycle, un enregistrement par séance.

    Args:
        path: Fichier de l'historique (créé au premier enregistrement).
        limit: Nombre maximal d'enregistrements conservés.
    """

    def __init__(self, path: Path, *, limit: int = _MAX_ENTRIES) -> None:
        self._path = path
        self._limit = limit

    @property
    def path(self) -> Path:
        """Chemin du fichier d'historique."""
        return self._path

    def record(self, report: Mapping[str, Any], *, source: str) -> dict[str, Any]:
        """Enregistre un rapport de cycle et rend l'entrée écrite.

        Pour une vigie horaire (``watch_only``), une entrée est conservée par
        créneau horaire. Pour un cycle complet (``full``), une seule entrée est
        conservée par séance (le dernier passage remplace le précédent).

        Args:
            report: Rapport de cycle (``CycleReport.to_dict()``).
            source: Origine de l'exécution (``timer``, ``gui`` ou ``cli``).

        Returns:
            L'entrée effectivement persistée.
        """
        entry: dict[str, Any] = {
            "recorded_at": _utc_now().isoformat(),
            "source": source,
            **report,
        }
        with self._locked() as handle:
            entries = self._parse(handle.read())
            session = entry.get("session_date")
            mode = entry.get("mode", "full")
            ts = entry.get("timestamp") or entry.get("recorded_at", "")
            hour = str(ts)[11:13] if len(str(ts)) >= 13 and "T" in str(ts) else ""

            replaced = False
            if entries and session is not None:
                last = entries[-1]
                last_session = last.get("session_date")
                last_mode = last.get("mode", "full")
                last_ts = last.get("timestamp") or last.get("recorded_at", "")
                last_hour = str(last_ts)[11:13] if len(str(last_ts)) >= 13 and "T" in str(last_ts) else ""

                same_full = mode == "full" and last_session == session and last_mode == "full"
                same_watch = (
                    mode == "watch_only"
                    and last_session == session
                    and last_mode == "watch_only"
                    and last_hour == hour
                )
                if same_full or same_watch:
                    entries[-1] = entry
                    replaced = True
                elif mode == "full":
                    # Si une entrée 'full' existait plus tôt pour la séance, on la remplace
                    for idx in range(len(entries) - 1, -1, -1):
                        if entries[idx].get("session_date") == session and entries[idx].get("mode", "full") == "full":
                            entries[idx] = entry
                            replaced = True
                            break

            if not replaced:
                entries.append(entry)

            if len(entries) > self._limit:
                del entries[: len(entries) - self._limit]
            self._rewrite(handle, entries)
        return entry

    def entries(self, *, limit: int = 60) -> list[dict[str, Any]]:
        """Rend les derniers cycles, du plus récent au plus ancien.

        Args:
            limit: Nombre maximal d'entrées rendues (négatif : tout).
        """
        with self._locked() as handle:
            rows = self._parse(handle.read())
        rows.reverse()
        return rows[:limit] if limit >= 0 else rows

    @contextmanager
    def _locked(self) -> Iterator[TextIO]:
        """Ouvre le fichier en lecture/écriture sous verrou inter-processus."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with _THREAD_LOCK:
            handle = self._path.open("a+", encoding="utf-8")
            try:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.seek(0)
                yield handle
            finally:
                try:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                finally:
                    handle.close()

    @staticmethod
    def _parse(text: str) -> list[dict[str, Any]]:
        """Interprète le journal ; une ligne corrompue est ignorée, jamais fatale."""
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
        return rows

    @staticmethod
    def _rewrite(handle: TextIO, entries: list[dict[str, Any]]) -> None:
        """Réécrit intégralement le journal puis force l'écriture sur disque."""
        handle.seek(0)
        handle.truncate()
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


__all__ = ["HistoryStore"]
