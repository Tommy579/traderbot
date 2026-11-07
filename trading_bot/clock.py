"""Horloge unifiée : aligne le bot sur l'heure en ligne.

Le bot ne se fie pas uniquement à l'horloge locale de la machine hôte, qui peut
être faussée par un double démarrage (RTC Windows/Linux), un changement d'heure
non appliqué ou une dérive matérielle.

Fonctionnement :
- Interroge une source en ligne fiable (Alpaca ou en-tête Date HTTP).
- Calcule le décalage par rapport à ``time.monotonic()`` pour garantir la régularité
  du temps sans saturer le réseau par des requêtes répétées.
- Mémorise la référence et re-synchronise périodiquement (toutes les 30 minutes)
  ou dès qu'un appel au courtier fournit une horloge officielle.
- Repli transparent sur l'horloge système en cas de coupure réseau complète.
"""

from __future__ import annotations

import email.utils
import logging
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Final

LOGGER = logging.getLogger("trading_bot.clock")

_ONLINE_SOURCES: Final[tuple[str, ...]] = (
    "https://paper-api.alpaca.markets",
    "https://api.alpaca.markets",
    "https://www.google.com",
    "https://cloudflare.com",
)

_SYNC_TTL_SECONDS: Final[float] = 1800.0  # 30 minutes
_HTTP_TIMEOUT_SECONDS: Final[float] = 1.5


class OnlineClock:
    """Horloge synchronisée sur l'heure en ligne via monotonic offset."""

    def __init__(
        self,
        *,
        sources: tuple[str, ...] = _ONLINE_SOURCES,
        sync_ttl: float = _SYNC_TTL_SECONDS,
        timeout: float = _HTTP_TIMEOUT_SECONDS,
    ) -> None:
        self._sources = sources
        self._sync_ttl = sync_ttl
        self._timeout = timeout
        self._offset_seconds: float | None = None
        self._last_sync_mono: float = 0.0
        self._lock = threading.Lock()
        self._is_synced = False

    @property
    def is_synced(self) -> bool:
        """Indique si l'horloge a déjà pu être synchronisée en ligne."""
        with self._lock:
            return self._is_synced

    def offset_seconds(self) -> float:
        """Décalage (en secondes) entre l'heure en ligne et l'heure système locale.

        Une valeur positive signifie que l'heure réelle en ligne est en avance
        sur l'horloge du PC hôte.
        """
        with self._lock:
            if self._offset_seconds is None:
                return 0.0
            online_ts = time.monotonic() + self._offset_seconds
            system_ts = time.time()
            return online_ts - system_ts

    def set_reference_time(self, online_dt: datetime) -> None:
        """Enregistre un instant de référence en ligne (ex: horloge Alpaca)."""
        dt_utc = online_dt.astimezone(UTC)
        with self._lock:
            mono = time.monotonic()
            self._offset_seconds = dt_utc.timestamp() - mono
            self._last_sync_mono = mono
            self._is_synced = True
            diff_with_system = dt_utc.timestamp() - time.time()
            if abs(diff_with_system) >= 1.0:
                LOGGER.info(
                    "horloge synchronisée via courtier : décalage de %+.2f s par rapport au PC",
                    diff_with_system,
                )

    def sync(self, force: bool = False) -> bool:
        """Tente de synchroniser l'horloge avec les serveurs en ligne."""
        with self._lock:
            now_mono = time.monotonic()
            if (
                not force
                and self._offset_seconds is not None
                and (now_mono - self._last_sync_mono) < self._sync_ttl
            ):
                return True

        online_dt = self._fetch_online_time()
        if online_dt is None:
            return False

        with self._lock:
            mono = time.monotonic()
            self._offset_seconds = online_dt.timestamp() - mono
            self._last_sync_mono = mono
            self._is_synced = True

            diff_with_system = online_dt.timestamp() - time.time()
            if abs(diff_with_system) >= 1.0:
                LOGGER.info(
                    "horloge en ligne synchronisée : décalage de %+.2f s par rapport au PC",
                    diff_with_system,
                )
            else:
                LOGGER.debug(
                    "horloge en ligne synchronisée (décalage système %+.3f s)",
                    diff_with_system,
                )
            return True

    def _fetch_online_time(self) -> datetime | None:
        """Interroge les serveurs en ligne pour lire l'en-tête Date HTTP standard."""
        for url in self._sources:
            try:
                req = urllib.request.Request(
                    url,
                    method="HEAD",
                    headers={"User-Agent": "TradingBot/1.0"},
                )
                date_str: str | None = None
                try:
                    with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                        date_str = resp.headers.get("date")
                except urllib.error.HTTPError as exc:
                    date_str = exc.headers.get("date")
                except (urllib.error.URLError, TimeoutError, OSError):
                    continue

                if date_str:
                    dt = email.utils.parsedate_to_datetime(date_str)
                    if dt is not None:
                        return dt.astimezone(UTC)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("échec de lecture de l'heure sur %s: %s", url, exc)
                continue
        return None

    def now(self) -> datetime:
        """Retourne l'instant actuel précis (UTC), synchronisé en ligne."""
        now_mono = time.monotonic()
        needs_sync = False
        with self._lock:
            if (
                self._offset_seconds is None
                or (now_mono - self._last_sync_mono) >= self._sync_ttl
            ):
                needs_sync = True

        if needs_sync:
            self.sync()

        with self._lock:
            if self._offset_seconds is not None:
                current_ts = time.monotonic() + self._offset_seconds
                return datetime.fromtimestamp(current_ts, tz=UTC)

        # Repli gracieux si hors ligne dès le démarrage
        return datetime.now(UTC)


_GLOBAL_CLOCK = OnlineClock()


def get_online_now() -> datetime:
    """Retourne l'instant actuel UTC, aligné sur l'heure en ligne."""
    return _GLOBAL_CLOCK.now()


def sync_online_clock(reference: datetime | None = None) -> bool:
    """Synchronise l'horloge globale.

    Si ``reference`` est fourni (ex: ``clock.timestamp`` d'Alpaca), s'aligne
    directement dessus. Sinon, interroge les serveurs distants.
    """
    if reference is not None:
        _GLOBAL_CLOCK.set_reference_time(reference)
        return True
    return _GLOBAL_CLOCK.sync(force=True)


def clock_offset_seconds() -> float:
    """Retourne le décalage actuel (en secondes) entre l'heure en ligne et le PC."""
    return _GLOBAL_CLOCK.offset_seconds()
