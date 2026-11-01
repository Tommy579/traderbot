"""Notation de sentiment des actualités financières.

Ce module combine l'API News d'Alpaca (fil Benzinga) et un petit modèle
OpenRouter pour transformer une actualité en un score exploitable par le
moteur de risque.

Deux principes gouvernent l'ensemble :

* **Coût** : un article déjà noté n'est jamais renvoyé au LLM. Le registre
  ``state/processed_news_ids.json`` mémorise les identifiants déjà vus ; tant
  qu'aucun article nouveau n'est paru, l'API n'est pas appelée du tout.
* **Disponibilité** : le bot ne doit jamais être bloqué par une panne d'API.
  Toute erreur ou tout dépassement de délai produit un score **neutre**, et
  le cycle de trading continue normalement.

Forme de l'API News (vérifiée contre l'API réelle) : ``NewsSet`` expose les
champs ``data`` et ``next_page_token``, où ``data`` est un dictionnaire
``{"news": [...]}`` de dictionnaires bruts — et non l'attribut ``.news``
documenté par les versions antérieures du SDK. Les identifiants d'articles
sont des entiers. Le code tolère donc aussi bien les dictionnaires que les
objets, pour survivre à un changement de version du SDK.
"""

from __future__ import annotations

import json
import logging
import math
import os
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "NEUTRAL_ASSESSMENT",
    "NewsArticle",
    "NewsAssessment",
    "NewsSentimentAnalyzer",
    "Transport",
]

LOGGER = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
PROCESSED_IDS_FILE = "processed_news_ids.json"
ASSESSMENTS_FILE = "news_sentiment.json"

#: Nombre d'identifiants d'articles conservés dans le registre. Au-delà, les
#: plus anciens sont oubliés : le pire effet est de renvoyer un vieil article
#: au LLM une seconde fois, jamais de perdre une actualité récente.
MAX_REGISTRY_ENTRIES = 5000

#: Longueur maximale d'un texte d'article transmis au modèle, en caractères.
#: Borne le coût par appel sans amputer les faits utiles.
MAX_TEXT_CHARS = 600


@dataclass(frozen=True, slots=True)
class NewsArticle:
    """Article de presse normalisé, prêt à être soumis au modèle."""

    article_id: str
    headline: str
    summary: str
    created_at: datetime | None
    symbols: tuple[str, ...]

    def as_prompt_text(self) -> str:
        """Rend l'article sous une forme compacte pour le prompt."""
        parts = [f"- {self.headline.strip()}"]
        summary = " ".join(self.summary.split())
        if summary:
            parts.append(f"  {summary[:MAX_TEXT_CHARS]}")
        return "\n".join(parts)


@dataclass(frozen=True, slots=True)
class NewsAssessment:
    """Verdict de sentiment pour un symbole sur la fenêtre analysée."""

    symbol: str
    sentiment_score: float
    confidence: float
    is_critical_threat: bool
    rationale: str
    articles_count: int
    updated_at: datetime

    def __post_init__(self) -> None:
        if not math.isfinite(self.sentiment_score):
            raise ValueError("sentiment_score doit être fini")
        if not math.isfinite(self.confidence):
            raise ValueError("confidence doit être fini")
        if not -1.0 <= self.sentiment_score <= 1.0:
            raise ValueError("sentiment_score doit appartenir à [-1, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence doit appartenir à [0, 1]")

    @classmethod
    def neutral(
        cls,
        symbol: str,
        *,
        now: datetime,
        rationale: str = "aucune analyse disponible",
        articles_count: int = 0,
    ) -> NewsAssessment:
        """Verdict inerte : ni veto, ni panique.

        C'est la valeur de repli du mode *fail-safe*, et celle d'un symbole
        sans actualité. Un score nul ne bloque rien : le pire défaut d'une
        panne doit être « filtre inopérant », jamais « bot qui refuse de
        trader ».
        """
        return cls(
            symbol=symbol,
            sentiment_score=0.0,
            confidence=0.0,
            is_critical_threat=False,
            rationale=rationale,
            articles_count=articles_count,
            updated_at=now,
        )

    def as_dict(self) -> dict[str, Any]:
        """Sérialise le verdict pour la persistance."""
        return {
            "symbol": self.symbol,
            "sentiment_score": self.sentiment_score,
            "confidence": self.confidence,
            "is_critical_threat": self.is_critical_threat,
            "rationale": self.rationale,
            "articles_count": self.articles_count,
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> NewsAssessment:
        """Relit un verdict persisté."""
        updated_at = payload.get("updated_at")
        if isinstance(updated_at, str):
            stamp = datetime.fromisoformat(updated_at)
        elif isinstance(updated_at, datetime):
            stamp = updated_at
        else:
            stamp = datetime.now(UTC)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        return cls(
            symbol=str(payload["symbol"]),
            sentiment_score=float(payload["sentiment_score"]),
            confidence=float(payload["confidence"]),
            is_critical_threat=bool(payload["is_critical_threat"]),
            rationale=str(payload.get("rationale", "")),
            articles_count=int(payload.get("articles_count", 0)),
            updated_at=stamp,
        )


#: Verdict utilisé comme ultime repli lorsqu'aucune évaluation n'a pu être
#: produite (import du SDK impossible, configuration absente, etc.).
NEUTRAL_ASSESSMENT = {
    "sentiment_score": 0.0,
    "confidence": 0.0,
    "is_critical_threat": False,
    "rationale": "aucune analyse disponible",
}


class Transport(Protocol):
    """Transport HTTP minimal, injectable pour les tests."""

    def __call__(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> tuple[int, str]:
        """Envoie ``payload`` en JSON et retourne ``(statut, corps)``."""
        ...


def _urllib_post(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout: float,
) -> tuple[int, str]:
    """Transport par défaut, sans dépendance externe.

    Contrairement aux autres transports du projet, ``HTTPError`` est converti
    en couple ``(statut, corps)`` au lieu d'être propagé : le code appelant
    doit pouvoir journaliser la réponse d'erreur d'OpenRouter (souvent
    explicite) et retomber en mode neutre.
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    for key, value in headers.items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return exc.code, detail
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, f"transport: {exc}"


def _field(article: Any, name: str, default: Any = None) -> Any:
    """Lit un champ sur un dictionnaire **ou** un objet.

    Le SDK Alpaca a changé la forme de ses réponses entre versions (objets
    typés ``News`` d'un côté, dictionnaires bruts de l'autre). Cette fonction
    absorbe les deux, ce qui évite un plantage silencieux à la prochaine mise
    à jour du SDK.
    """
    if isinstance(article, Mapping):
        value = article.get(name, default)
    else:
        value = getattr(article, name, default)
    return default if value is None else value


class NewsSentimentAnalyzer:
    """Note les actualités d'un ensemble de symboles via Alpaca News + LLM."""

    def __init__(
        self,
        *,
        state_dir: Path,
        alpaca_api_key: str,
        alpaca_secret_key: str,
        openrouter_api_key: str | None,
        model: str,
        lookback_hours: int = 24,
        max_articles: int = 10,
        timeout_seconds: float = 5.0,
        news_client: Any | None = None,
        transport: Transport | None = None,
    ) -> None:
        """Initialise l'analyseur.

        Args:
            state_dir: Répertoire d'état (registre et verdicts persistés).
            alpaca_api_key: Clé Alpaca (l'API News partage l'authentification
                du reste des données de marché).
            alpaca_secret_key: Secret Alpaca associé.
            openrouter_api_key: Clé OpenRouter. ``None`` ou vide désactive
                l'appel au modèle : les scores restent neutres.
            model: Identifiant du modèle OpenRouter.
            lookback_hours: Profondeur de la fenêtre d'actualités.
            max_articles: Plafond d'articles envoyés au modèle par appel.
            timeout_seconds: Délai maximal d'un appel OpenRouter.
            news_client: Client Alpaca News injectable (tests).
            transport: Transport HTTP injectable (tests).
        """
        self.state_dir = Path(state_dir)
        self.lookback_hours = lookback_hours
        self.max_articles = max_articles
        self.timeout_seconds = timeout_seconds
        self.model = model
        self._api_key = alpaca_api_key
        self._secret_key = alpaca_secret_key
        stripped = (openrouter_api_key or "").strip()
        self._openrouter_key = stripped or None
        self._news_client = news_client
        self._transport: Transport = transport or _urllib_post

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------
    @property
    def llm_enabled(self) -> bool:
        """``True`` si une clé OpenRouter permet d'interroger le modèle."""
        return self._openrouter_key is not None

    @property
    def registry_path(self) -> Path:
        """Chemin du registre des identifiants d'articles déjà notés."""
        return self.state_dir / PROCESSED_IDS_FILE

    @property
    def assessments_path(self) -> Path:
        """Chemin des derniers verdicts par symbole."""
        return self.state_dir / ASSESSMENTS_FILE

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        """Lit un fichier JSON, ou retourne un dictionnaire vide.

        Un fichier corrompu est traité comme absent : mieux vaut une note
        perdue qu'un cycle de trading interrompu.
        """
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:  # pragma: no cover - défense en profondeur
            LOGGER.warning("lecture de %s impossible: %s", path, exc)
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            LOGGER.warning("%s illisible (%s), il est ignoré", path, exc)
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        """Écrit un JSON de façon atomique (fichier temporaire puis renommage).

        Une écriture directe laisserait un fichier tronqué si le processus
        meurt en cours de route — au pire moment, puisque c'est précisément
        l'état qui protège du gaspillage de jetons.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _load_registry(self) -> list[str]:
        """Retourne les identifiants d'articles déjà soumis au modèle."""
        raw = self._read_json(self.registry_path).get("ids", [])
        if not isinstance(raw, list):
            return []
        return [str(item) for item in raw]

    def _load_assessments(self) -> dict[str, NewsAssessment]:
        """Recharge les verdicts précédents, en ignorant les entrées illisibles."""
        raw = self._read_json(self.assessments_path).get("assessments", {})
        if not isinstance(raw, Mapping):
            return {}
        restored: dict[str, NewsAssessment] = {}
        for symbol, payload in raw.items():
            if not isinstance(payload, Mapping):
                continue
            try:
                restored[str(symbol)] = NewsAssessment.from_dict(payload)
            except (KeyError, TypeError, ValueError) as exc:
                LOGGER.debug("verdict ignoré pour %s: %s", symbol, exc)
        return restored

    # ------------------------------------------------------------------
    # Récupération des articles
    # ------------------------------------------------------------------
    def _client(self) -> Any:
        """Construit (ou retourne) le client Alpaca News."""
        if self._news_client is None:
            from alpaca.data.historical.news import NewsClient

            self._news_client = NewsClient(self._api_key, self._secret_key)
        return self._news_client

    def fetch_articles(
        self, symbol: str, *, now: datetime | None = None
    ) -> list[NewsArticle]:
        """Récupère les actualités récentes d'un symbole.

        Returns:
            Les articles de la fenêtre, du plus récent au plus ancien. Une
            liste vide signifie « pas d'actualité », jamais une erreur : une
            panne du fil d'actualités ne doit pas interrompre le trading.
        """
        from alpaca.data.requests import NewsRequest

        moment = now or datetime.now(UTC)
        try:
            response = self._client().get_news(
                NewsRequest(
                    symbols=symbol,
                    start=moment - timedelta(hours=self.lookback_hours),
                    end=moment,
                    limit=self.max_articles,
                    sort="desc",
                )
            )
        except Exception as exc:  # noqa: BLE001 - le fil d'actualités ne doit jamais bloquer
            LOGGER.warning("actualités indisponibles pour %s: %s", symbol, exc)
            return []

        raw = _field(response, "data", response)
        items: Iterable[Any] = _field(raw, "news", []) or []
        articles: list[NewsArticle] = []
        for item in items:
            article_id = _field(item, "id")
            if article_id is None:
                continue
            created = _field(item, "created_at")
            if not isinstance(created, datetime):
                created = None
            symbols = _field(item, "symbols", []) or []
            articles.append(
                NewsArticle(
                    article_id=str(article_id),
                    headline=str(_field(item, "headline", "")),
                    summary=str(_field(item, "summary", "")),
                    created_at=created,
                    symbols=tuple(str(s) for s in symbols),
                )
            )
        return articles

    # ------------------------------------------------------------------
    # Appel au modèle
    # ------------------------------------------------------------------
    def _build_prompt(self, symbol: str, articles: Sequence[NewsArticle]) -> str:
        """Construit l'invite envoyée au modèle."""
        corpus = "\n".join(article.as_prompt_text() for article in articles)
        return (
            f"Actualités récentes concernant {symbol} :\n{corpus}\n\n"
            "Évalue leur tonalité pour ce titre sur les prochaines séances.\n"
            "Réponds uniquement par un objet JSON avec les clés :\n"
            '  "sentiment_score": nombre de -1.0 (catastrophique) à 1.0 (excellent),\n'
            '  "confidence": nombre de 0.0 à 1.0 (confiance dans ta note),\n'
            '  "is_critical_threat": booléen, vrai seulement pour un risque '
            "existentiel ou une fraude ou un krach imminent,\n"
            '  "rationale": phrase courte justifiant la note, en français.'
        )

    def _ask_model(
        self, symbol: str, articles: Sequence[NewsArticle], *, now: datetime
    ) -> tuple[NewsAssessment, bool]:
        """Interroge OpenRouter et convertit la réponse en verdict.

        Toute anomalie — clé absente, statut non 200, délai dépassé, JSON
        invalide, champs hors bornes — produit un verdict neutre. Le filtre
        est *fail-safe* : son pire défaut est de ne rien filtrer.

        Returns:
            Le verdict, et un booléen disant si le modèle a **réellement**
            répondu. Ce second membre décide de l'inscription des articles au
            registre : marquer comme « traité » un article jamais noté
            condamnerait le filtre à rester neutre jusqu'à la prochaine
            actualité — c'est-à-dire, typiquement, au moment où l'on aurait
            eu besoin de lui.
        """
        if self._openrouter_key is None:
            return (
                NewsAssessment.neutral(
                    symbol,
                    now=now,
                    rationale="clé OpenRouter absente : filtre inactif",
                    articles_count=len(articles),
                ),
                False,
            )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Tu es un analyste de marché actions. Tu notes la tonalité "
                        "d'actualités pour un titre, avec prudence et sans emphase."
                    ),
                },
                {"role": "user", "content": self._build_prompt(symbol, articles)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        headers = {
            "Authorization": f"Bearer {self._openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Tommy579/traderbot",
            "X-Title": "tradingbot",
        }
        try:
            status, body = self._transport(
                OPENROUTER_URL, payload, headers, self.timeout_seconds
            )
        except Exception as exc:  # noqa: BLE001 - fail-safe assumé
            LOGGER.warning("OpenRouter injoignable pour %s: %s", symbol, exc)
            # Panne passagère : l'article n'est pas inscrit au registre, il
            # sera retenté au cycle suivant.
            return (
                NewsAssessment.neutral(
                    symbol,
                    now=now,
                    rationale=f"erreur de transport: {exc}",
                    articles_count=len(articles),
                ),
                False,
            )

        if status != 200:
            LOGGER.warning("OpenRouter a répondu %s pour %s: %s", status, symbol, body[:200])
            return (
                NewsAssessment.neutral(
                    symbol,
                    now=now,
                    rationale=f"réponse HTTP {status}",
                    articles_count=len(articles),
                ),
                False,
            )
        return self._parse_response(symbol, body, now=now, count=len(articles))

    def _parse_response(
        self, symbol: str, body: str, *, now: datetime, count: int
    ) -> tuple[NewsAssessment, bool]:
        """Extrait le verdict d'une réponse OpenRouter, ou retombe en neutre."""

        def fallback(reason: str) -> tuple[NewsAssessment, bool]:
            return (
                NewsAssessment.neutral(
                    symbol, now=now, rationale=reason, articles_count=count
                ),
                False,
            )

        try:
            envelope = json.loads(body)
            content = envelope["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            LOGGER.warning("réponse OpenRouter inexploitable pour %s: %s", symbol, exc)
            return fallback("réponse du modèle inexploitable")

        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, Mapping)
            )
        try:
            verdict = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            LOGGER.warning("JSON invalide du modèle pour %s: %s", symbol, exc)
            return fallback("JSON du modèle invalide")
        if not isinstance(verdict, Mapping):
            return fallback("le modèle n'a pas renvoyé d'objet JSON")

        try:
            score = float(verdict["sentiment_score"])
            confidence = float(verdict["confidence"])
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("champs de sentiment manquants pour %s: %s", symbol, exc)
            return fallback("champs de sentiment manquants")
        if not math.isfinite(score) or not math.isfinite(confidence):
            return fallback("note non finie")

        score = max(-1.0, min(1.0, score))
        confidence = max(0.0, min(1.0, confidence))
        return (
            NewsAssessment(
                symbol=symbol,
                sentiment_score=score,
                confidence=confidence,
                is_critical_threat=bool(verdict.get("is_critical_threat", False)),
                rationale=str(verdict.get("rationale", ""))[:400],
                articles_count=count,
                updated_at=now,
            ),
            True,
        )

    # ------------------------------------------------------------------
    # Point d'entrée
    # ------------------------------------------------------------------
    def assess(
        self, symbols: Iterable[str], *, now: datetime | None = None
    ) -> dict[str, NewsAssessment]:
        """Note chaque symbole, en ne payant que pour les actualités neuves.

        Args:
            symbols: Symboles à évaluer.
            now: Horodatage de référence (injection pour les tests).

        Returns:
            Un verdict par symbole. Un symbole dont les articles ont déjà tous
            été notés conserve son verdict précédent, sans appel au modèle.
        """
        moment = now or datetime.now(UTC)
        known = self._load_registry()
        known_set = set(known)
        previous = self._load_assessments()
        results: dict[str, NewsAssessment] = {}
        fresh_ids: list[str] = []

        for symbol in symbols:
            articles = self.fetch_articles(symbol, now=moment)
            unseen = [a for a in articles if a.article_id not in known_set]
            if not unseen:
                # Rien de neuf : on réutilise la note précédente. C'est ici
                # que le module gagne son argument « coût zéro ».
                results[symbol] = previous.get(symbol) or NewsAssessment.neutral(
                    symbol,
                    now=moment,
                    rationale="aucune actualité récente",
                )
                continue
            assessment, analysed = self._ask_model(symbol, unseen, now=moment)
            results[symbol] = assessment
            if analysed:
                # Seul un article réellement noté entre au registre. Une note
                # neutre d'origine « clé absente » ou « panne » ne doit pas
                # faire croire que l'article a été traité.
                fresh_ids.extend(article.article_id for article in unseen)

        # Fusion avec les verdicts déjà connus : un passage partiel (par
        # exemple « news --symbols NVDA ») ne doit pas effacer les notes des
        # autres symboles. Sans cela, le passage suivant les retrouve absents,
        # y substitue une note neutre de repli, et le filtre devient aveugle
        # sur tout l'univers sans que rien ne le signale.
        if results:
            self._save_state({**previous, **results}, known + fresh_ids, moment)
        return results

    def _save_state(
        self,
        assessments: Mapping[str, NewsAssessment],
        registry: Sequence[str],
        now: datetime,
    ) -> None:
        """Persiste verdicts et registre d'articles.

        Le registre est dédoublonné en conservant l'ordre : un même article
        revient légitimement plusieurs fois dans un même passage (il est tagué
        de plusieurs symboles et alimente donc plusieurs invites), mais il ne
        doit figurer qu'une fois sur disque.
        """
        trimmed = list(dict.fromkeys(str(item) for item in registry))[
            -MAX_REGISTRY_ENTRIES:
        ]
        self._write_json(
            self.registry_path,
            {"updated_at": now.isoformat(), "ids": trimmed},
        )
        self._write_json(
            self.assessments_path,
            {
                "updated_at": now.isoformat(),
                "model": self.model,
                "assessments": {
                    symbol: assessment.as_dict()
                    for symbol, assessment in assessments.items()
                },
            },
        )
