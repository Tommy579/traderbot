"""Tâches de fond de l'interface web.

Un backtest télécharge des barres et peut durer des dizaines de secondes ;
un cycle live attend l'API du courtier. L'interface soumet donc une tâche
(POST), affiche « en cours… », puis interroge son résultat (GET). Le
gestionnaire est volontairement minuscule : un dictionnaire, un verrou, des
threads démons.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final

_MAX_JOBS: Final[int] = 20


class JobStatus(StrEnum):
    """Cycle de vie d'une tâche."""

    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass(slots=True)
class Job:
    """Tâche en cours ou terminée."""

    id: str
    kind: str
    label: str
    status: JobStatus = JobStatus.RUNNING
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def snapshot(self) -> dict[str, Any]:
        """Vue JSON-compatible pour l'interface."""
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
        }


class JobManager:
    """File de tâches de fond, consultable par identifiant.

    Args:
        history_limit: Nombre de tâches conservées (les plus anciennes disparaissent).
    """

    def __init__(self, *, history_limit: int = _MAX_JOBS) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._history_limit = history_limit

    @property
    def running_kinds(self) -> set[str]:
        """Types de tâches actuellement en exécution."""
        with self._lock:
            return {job.kind for job in self._jobs.values() if job.status is JobStatus.RUNNING}

    def get(self, job_id: str) -> Job | None:
        """Retourne la tâche si elle est encore connue."""
        with self._lock:
            return self._jobs.get(job_id)

    def submit(
        self,
        kind: str,
        label: str,
        target: Callable[[], dict[str, Any]],
        *,
        lock: threading.Lock | None = None,
    ) -> Job:
        """Exécute ``target`` dans un thread démon et retourne la tâche.

        ``lock`` sérialise les tâches qui se marchent dessus (ex : deux
        backtests écrivant le même cache disque) — la tâche attend alors son
        tour, son statut reste « running ».
        """
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, label=label)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > self._history_limit:
                self._jobs.pop(self._order.pop(0), None)

        thread = threading.Thread(
            target=self._run,
            args=(job, target, lock),
            name=f"webjob-{job.id}",
            daemon=True,
        )
        thread.start()
        return job

    def _run(
        self,
        job: Job,
        target: Callable[[], dict[str, Any]],
        lock: threading.Lock | None,
    ) -> None:
        """Corps du thread : exécute, capture succès ou échec."""
        try:
            if lock is not None:
                with lock:
                    result = target()
            else:
                result = target()
            with self._lock:
                job.status = JobStatus.DONE
                job.result = result
        except Exception as exc:  # noqa: BLE001 - remonté à l'interface, pas au thread
            with self._lock:
                job.status = JobStatus.FAILED
                job.error = str(exc)


__all__ = ["Job", "JobManager", "JobStatus"]
