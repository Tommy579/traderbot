"""Publication des statistiques dans le dépôt GitHub (GitHub Pages).

GitHub Pages ne sert que des fichiers : il ne peut pas *recevoir* un POST.
Plutôt que d'héberger un service de réception, le bot **commite** son
récapitulatif dans le dépôt via l'API Contents ; Pages sert ensuite ce fichier
tel quel, et la page de consultation en a besoin d'aucune infrastructure.

Comme la couche webhook, cette couche a le droit d'échouer : une publication
manquée ne doit jamais interrompre un cycle de trading. ``send`` retourne donc
un booléen et journalise les échecs au lieu de lever.

Transport par ``urllib`` (bibliothèque standard) : aucune dépendance ajoutée.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Final, Protocol

from trading_bot.clock import get_online_now

LOGGER = logging.getLogger(__name__)

_API_ROOT: Final[str] = "https://api.github.com"
_MAX_ENTRIES: Final[int] = 400
_DEFAULT_PATH: Final[str] = "docs/data/stats.json"
_DEFAULT_BRANCH: Final[str] = "main"


class Transport(Protocol):
    """Requête HTTP générique ; lève en cas d'échec réseau."""

    def __call__(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: Mapping[str, str],
        *,
        timeout_seconds: float,
    ) -> tuple[int, bytes]: ...


def _make_opener() -> urllib.request.OpenerDirector:
    """Ouvreur urllib laissant remonter les erreurs HTTP (401, 404, 409…)."""
    return urllib.request.build_opener()


def _urllib_transport(
    method: str,
    url: str,
    body: bytes | None,
    headers: Mapping[str, str],
    *,
    timeout_seconds: float,
) -> tuple[int, bytes]:
    """Exécute la requête et renvoie ``(statut, corps)`` sans lever sur 4xx/5xx."""
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    opener = _make_opener()
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()



def _utc_now() -> datetime:
    """Horloge par défaut : maintenant en UTC, aligné sur l'heure en ligne."""
    return get_online_now()


class GitHubStatsPublisher:
    """Écrit le récapitulatif de cycle dans un fichier du dépôt.

    Le fichier produit est un JSON ``{"updated_at": …, "entries": [...]}``,
    consommé directement par la page GitHub Pages. Un seul enregistrement est
    conservé par séance : relancer un cycle le même jour remplace l'entrée,
    comme ``state/history.jsonl``.

    Args:
        repo: Dépôt cible, au format ``compte/nom``.
        token: Jeton GitHub avec le droit d'écriture sur le contenu.
        path: Chemin du fichier dans le dépôt.
        branch: Branche publiée.
        timeout_seconds: Délai réseau par requête.
        transport: Requête injectable pour les tests.

    Raises:
        ValueError: si le dépôt ou le jeton est absent, ou le chemin invalide.
    """

    def __init__(
        self,
        *,
        repo: str,
        token: str,
        path: str = _DEFAULT_PATH,
        branch: str = _DEFAULT_BRANCH,
        timeout_seconds: float = 15.0,
        transport: Transport | None = None,
        clock: Any = _utc_now,
    ) -> None:
        if not repo or "/" not in repo:
            raise ValueError(f"dépôt GitHub invalide: {repo!r} (attendu « compte/nom »)")
        if not token:
            raise ValueError("jeton GitHub absent : publication impossible")
        if not path or path.startswith("/"):
            raise ValueError(f"chemin invalide: {path!r} (relatif au dépôt, sans /)")
        self._repo = repo
        self._token = token
        self._path = path
        self._branch = branch
        self._timeout_seconds = timeout_seconds
        self._transport: Transport = transport or _urllib_transport
        self._clock = clock

    def __repr__(self) -> str:
        """Représentation sans secret."""
        return (
            f"GitHubStatsPublisher(repo={self._repo!r}, path={self._path!r}, "
            f"branch={self._branch!r})"
        )

    @property
    def enabled(self) -> bool:
        """Toujours ``True`` : l'objet n'existe que s'il est configuré."""
        return True

    def _headers(self) -> dict[str, str]:
        """En-têtes exigés par l'API GitHub (dont le User-Agent obligatoire)."""
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "trading-bot-stats-publisher",
            "Content-Type": "application/json",
        }

    @property
    def _contents_url(self) -> str:
        """URL de l'API Contents pour le fichier suivi."""
        return f"{_API_ROOT}/repos/{self._repo}/contents/{self._path}"

    def send(self, event: str, payload: Mapping[str, Any]) -> bool:
        """Publie le récapitulatif ; ``True`` seulement si le commit a abouti.

        Aucune exception ne sort de cette méthode.

        Args:
            event: Nom de l'événement (``cycle_trading``).
            payload: Rapport de cycle (``CycleReport.to_dict()``).

        Returns:
            ``True`` si le fichier a bien été écrit dans le dépôt.
        """
        try:
            remote = self._fetch_remote()
            if remote is None:
                return False
            entries, sha = remote
            entries = self._merge(entries, event, payload)
            return self._write_entries(entries, sha)
        except Exception as exc:  # noqa: BLE001 - best-effort par contrat
            LOGGER.warning("publication des statistiques échouée: %s", exc)
            return False

    def _fetch_remote(self) -> tuple[list[dict[str, Any]], str | None] | None:
        """Lit le fichier distant : ``(entrées, sha)`` ou ``None`` si illisible.

        Un seul GET rapporte à la fois le contenu et le ``sha`` nécessaire au
        remplacement : deux appels successifs pourraient voir deux versions
        différentes du fichier.

        Un 404 est le cas normal de la première publication (liste vide, pas
        d'erreur). Toute autre anomalie rend ``None`` : mieux vaut ne rien
        publier que d'écraser un historique qu'on n'a pas su lire.
        """
        url = f"{self._contents_url}?ref={self._branch}"
        status, body = self._transport(
            "GET", url, None, self._headers(), timeout_seconds=self._timeout_seconds
        )
        if status == 404:
            return [], None
        if status != 200:
            LOGGER.warning("lecture des statistiques refusée (HTTP %s)", status)
            return None
        try:
            document = json.loads(body)
        except json.JSONDecodeError as exc:
            LOGGER.warning("réponse distante illisible: %s", exc)
            return None
        if not isinstance(document, dict):
            LOGGER.warning("réponse distante inattendue")
            return None

        raw_sha = document.get("sha")
        sha = str(raw_sha) if raw_sha else None

        encoded = document.get("content")
        if not isinstance(encoded, str):
            # Contenu absent (fichier > 1 Mo) ou réponse inattendue.
            LOGGER.warning("contenu distant absent ou non exploitable")
            return None
        try:
            decoded = base64.b64decode(encoded).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as exc:
            LOGGER.warning("contenu distant indécodable: %s", exc)
            return None
        if not decoded.strip():
            return [], sha
        try:
            parsed = json.loads(decoded)
        except json.JSONDecodeError:
            LOGGER.warning("statistiques distantes corrompues : repartir à zéro")
            return [], sha
        entries = parsed.get("entries") if isinstance(parsed, dict) else None
        return (entries if isinstance(entries, list) else []), sha

    def _merge(
        self,
        entries: list[dict[str, Any]],
        event: str,
        payload: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Insère ou remplace l'entrée en respectant la règle par séance / créneau."""
        entry = {
            "received_at": self._clock().isoformat(),
            "event": event,
            "report": dict(payload),
        }
        session = payload.get("session_date")
        mode = payload.get("mode", "full")
        ts = payload.get("timestamp") or entry.get("received_at", "")
        hour = str(ts)[11:13] if len(str(ts)) >= 13 and "T" in str(ts) else ""

        if session is not None:
            replaced = False
            new_entries = []
            for existing in entries:
                rep = existing.get("report", {}) if isinstance(existing, dict) else {}
                ex_session = rep.get("session_date")
                ex_mode = rep.get("mode", "full")
                ex_ts = rep.get("timestamp") or existing.get("received_at", "")
                ex_hour = str(ex_ts)[11:13] if len(str(ex_ts)) >= 13 and "T" in str(ex_ts) else ""

                same_full = mode == "full" and ex_mode == "full"
                same_watch = mode == "watch_only" and ex_mode == "watch_only" and ex_hour == hour
                match = ex_session == session and (same_full or same_watch)

                if match:
                    new_entries.append(entry)
                    replaced = True
                else:
                    new_entries.append(existing)
            entries = new_entries if replaced else [*entries, entry]
        else:
            entries = [*entries, entry]

        if len(entries) > _MAX_ENTRIES:
            entries = entries[-_MAX_ENTRIES:]
        return entries

    def _write_entries(self, entries: list[dict[str, Any]], sha: str | None) -> bool:
        """Commit le fichier via l'API Contents (création ou mise à jour).

        ``sha`` est celui lu juste avant : l'API GitHub l'exige pour remplacer
        un fichier existant, et l'omettre pour en créer un nouveau.
        """
        document = {
            "updated_at": self._clock().isoformat(),
            "entries": entries,
        }
        content = json.dumps(document, ensure_ascii=False, indent=1) + "\n"
        body = {
            "message": f"stats: {self._session_label(entries)}",
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": self._branch,
        }
        if sha:
            body["sha"] = sha
        status, response = self._transport(
            "PUT",
            self._contents_url,
            json.dumps(body).encode("utf-8"),
            self._headers(),
            timeout_seconds=self._timeout_seconds,
        )
        if status in (200, 201):
            LOGGER.info(
                "statistiques publiées dans %s (%d enregistrement(s))",
                self._path,
                len(entries),
            )
            return True
        LOGGER.warning(
            "publication refusée (HTTP %s): %s",
            status,
            response[:200].decode("utf-8", "replace"),
        )
        return False

    @staticmethod
    def _session_label(entries: list[dict[str, Any]]) -> str:
        """Étiquette lisible du commit : la séance publiée."""
        if not entries:
            return "aucune séance"
        report = entries[-1].get("report", {})
        session = report.get("session_date") if isinstance(report, dict) else None
        return f"séance {session}" if session else "cycle"


__all__ = ["GitHubStatsPublisher"]
