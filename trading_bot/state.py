"""État persistant du bot : risque, stops et journal d'audit.

Extrait de l'orchestrateur pour être partagé sans cycle d'imports avec les
retraits manuels (``trading_bot.withdrawal``) et l'interface web. Les écritures
sont atomiques (fichier temporaire + ``os.replace`` + ``fsync``) : un arrêt
brutal ne doit jamais laisser un ``state.json`` tronqué.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from trading_bot.clock import get_online_now
from trading_bot.config import ConfigurationError


def utc_now() -> datetime:
    """Horloge par défaut : maintenant en UTC, aligné sur l'heure en ligne."""
    return get_online_now()


class PersistedState(BaseModel):
    """État de risque minimal survivant aux redémarrages.

    Les stops sont la seule information que le courtier ne connaît pas : ils
    doivent être restaurés à l'identique pour éviter qu'une position reste
    « nue » après un redémarrage.
    """

    model_config = ConfigDict(extra="ignore")

    trading_day: str | None = None
    day_start_equity: float | None = None
    peak_equity: float | None = None
    halted: bool = False
    halt_scope: str | None = None
    halt_reason: str | None = None
    stops: dict[str, dict[str, Any]] = {}


class StateStore:
    """Persistance atomique de l'état de risque (JSON)."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """Chemin du fichier d'état."""
        return self._path

    def load(self) -> PersistedState:
        """Charge l'état, ou retourne un état vierge s'il n'existe pas.

        Raises:
            ConfigurationError: si le fichier existe mais est illisible.
        """
        if not self._path.exists():
            return PersistedState()
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            return PersistedState.model_validate(payload)
        except (OSError, ValueError) as exc:
            raise ConfigurationError(
                f"état de risque illisible ({self._path}): {exc}. "
                "Corrigez ou déplacez le fichier avant de redémarrer."
            ) from exc

    def save(self, state: PersistedState) -> None:
        """Écrit l'état de manière atomique (fichier temporaire + ``os.replace``)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = state.model_dump(mode="json")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self._path)
        if os.name != "nt":
            directory = os.open(self._path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)


class AuditLog:
    """Journal d'audit append-only au format JSON Lines."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def record(self, event: str, payload: Mapping[str, Any]) -> None:
        """Ajoute un événement horodaté et force l'écriture sur disque."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": utc_now().isoformat(),
            "event": event,
            "payload": payload,
        }
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
