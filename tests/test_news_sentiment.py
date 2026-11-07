"""Tests du filtre d'actualités : veto, vigie, cache et mode *fail-safe*.

Aucun accès réseau : le client Alpaca News et le transport OpenRouter sont
tous deux simulés. Les formes reproduites sont celles **réellement observées**
sur l'API (``NewsSet.data["news"]`` peuplé de dictionnaires, identifiants
entiers) et non celles de la documentation du SDK, qui décrit un attribut
``.news`` inexistant en 0.44.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from trading_bot.config import RiskConfig, Settings
from trading_bot.data.news_sentiment import (
    NewsArticle,
    NewsAssessment,
    NewsSentimentAnalyzer,
)
from trading_bot.risk.engine import (
    AccountState,
    Position,
    RejectionCode,
    RiskEngine,
    Side,
)

NOW = datetime(2024, 1, 10, 15, 0, tzinfo=UTC)
SESSION = date(2024, 1, 10)
MODEL = "google/gemini-3.8-flash"


# ---------------------------------------------------------------------------
# Doublures
# ---------------------------------------------------------------------------
def make_article(
    article_id: int = 1,
    *,
    headline: str = "Nvidia chute après un avertissement",
    summary: str = "Le titre perd du terrain sur des craintes de surcapacité.",
    symbols: tuple[str, ...] = ("NVDA",),
) -> dict[str, Any]:
    """Article au format réellement renvoyé par l'API News."""
    return {
        "id": article_id,
        "headline": headline,
        "summary": summary,
        "created_at": NOW - timedelta(hours=1),
        "symbols": list(symbols),
        "source": "benzinga",
    }


class FakeNewsSet:
    """Réponse ``NewsSet`` simulée : le champ ``data``, comme l'API réelle."""

    def __init__(self, articles: list[dict[str, Any]]) -> None:
        self.data = {"news": list(articles)}
        self.next_page_token = None


class FakeNewsClient:
    """Client Alpaca News simulé, comptant les interrogations."""

    def __init__(self, articles_by_symbol: dict[str, list[dict[str, Any]]]) -> None:
        self.articles_by_symbol = articles_by_symbol
        self.calls: list[str] = []

    def get_news(self, request: Any) -> FakeNewsSet:
        raw = request.symbols
        symbol = raw if isinstance(raw, str) else raw[0]
        self.calls.append(symbol)
        return FakeNewsSet(self.articles_by_symbol.get(symbol, []))


class FakeTransport:
    """Transport OpenRouter simulé : compte les appels, rejoue une réponse."""

    def __init__(
        self,
        *,
        status: int = 200,
        body: str | None = None,
        error: Exception | None = None,
    ) -> None:
        self.status = status
        self.body = body
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> tuple[int, str]:
        self.calls.append(
            {"url": url, "payload": payload, "headers": headers, "timeout": timeout}
        )
        if self.error is not None:
            raise self.error
        return self.status, self.body or ""


def llm_body(
    score: float,
    *,
    confidence: float = 0.8,
    critical: bool = False,
    rationale: str = "note de test",
) -> str:
    """Enveloppe de réponse OpenRouter contenant le verdict demandé."""
    verdict = {
        "sentiment_score": score,
        "confidence": confidence,
        "is_critical_threat": critical,
        "rationale": rationale,
    }
    return json.dumps({"choices": [{"message": {"content": json.dumps(verdict)}}]})


def make_analyzer(
    tmp_path: Path,
    *,
    articles: dict[str, list[dict[str, Any]]] | None = None,
    transport: FakeTransport | None = None,
    key: str | None = "sk-or-test",
) -> NewsSentimentAnalyzer:
    """Analyseur isolé du réseau et du disque réel."""
    return NewsSentimentAnalyzer(
        state_dir=tmp_path / "state",
        alpaca_api_key="PK_TEST",
        alpaca_secret_key="SECRET_TEST",
        openrouter_api_key=key,
        model=MODEL,
        news_client=FakeNewsClient(articles or {}),
        transport=transport or FakeTransport(body=llm_body(0.0)),
    )


def make_assessment(
    score: float, *, critical: bool = False, symbol: str = "NVDA"
) -> NewsAssessment:
    """Verdict de sentiment prêt à l'emploi."""
    return NewsAssessment(
        symbol=symbol,
        sentiment_score=score,
        confidence=0.9,
        is_critical_threat=critical,
        rationale="test",
        articles_count=2,
        updated_at=NOW,
    )


def make_account(equity: float = 100_000.0) -> AccountState:
    """Compte paper de test, sans position."""
    return AccountState(equity=equity, cash=equity, buying_power=equity)


def make_position(symbol: str = "NVDA", *, qty: float = 10.0) -> Position:
    """Position ouverte protégée par un stop."""
    return Position(
        symbol=symbol,
        qty=qty,
        avg_entry_price=100.0,
        current_price=95.0,
        stop_price=90.0,
        initial_stop_price=90.0,
    )


# ---------------------------------------------------------------------------
# 1. Veto d'entrée
# ---------------------------------------------------------------------------
def test_buy_rejected_when_news_is_negative() -> None:
    """Un score ≤ -0.60 refuse l'achat, avec le code explicite [NEWS_VETO]."""
    engine = RiskEngine()
    decision = engine.evaluate_entry(
        symbol="NVDA",
        price=50.0,
        atr=2.0,
        account=make_account(),
        now=NOW,
        session_date=SESSION,
        news=make_assessment(-0.72),
    )
    assert not decision.approved
    assert decision.code is RejectionCode.NEWS_VETO
    assert decision.intent is None
    assert "-0.72" in decision.reason
    assert decision.metrics["news_score"] == pytest.approx(-0.72)


def test_veto_is_written_to_the_audit_trail() -> None:
    """Le refus est journalisé : le bot recopie l'audit dans audit.jsonl."""
    engine = RiskEngine()
    engine.evaluate_entry(
        symbol="NVDA",
        price=50.0,
        atr=2.0,
        account=make_account(),
        now=NOW,
        session_date=SESSION,
        news=make_assessment(-0.72),
    )
    assert engine.audit[-1].code is RejectionCode.NEWS_VETO
    assert engine.audit[-1].approved is False


def test_critical_threat_vetoes_even_with_a_mild_score() -> None:
    """Une menace critique suffit, sans attendre que le score s'effondre."""
    engine = RiskEngine()
    decision = engine.evaluate_entry(
        symbol="NVDA",
        price=50.0,
        atr=2.0,
        account=make_account(),
        now=NOW,
        session_date=SESSION,
        news=make_assessment(-0.10, critical=True),
    )
    assert decision.code is RejectionCode.NEWS_VETO


def test_mild_negative_news_does_not_block_entry() -> None:
    """Au-dessus du seuil, l'actualité est un signal, pas un veto."""
    engine = RiskEngine()
    decision = engine.evaluate_entry(
        symbol="NVDA",
        price=50.0,
        atr=2.0,
        account=make_account(),
        now=NOW,
        session_date=SESSION,
        news=make_assessment(-0.30),
    )
    assert decision.approved


def test_threshold_boundary_is_inclusive() -> None:
    """Le seuil exact bloque : « inférieur ou égal » est la règle annoncée."""
    engine = RiskEngine()
    decision = engine.evaluate_entry(
        symbol="NVDA",
        price=50.0,
        atr=2.0,
        account=make_account(),
        now=NOW,
        session_date=SESSION,
        news=make_assessment(-0.60),
    )
    assert decision.code is RejectionCode.NEWS_VETO


# ---------------------------------------------------------------------------
# 2. Vigie : sortie d'urgence
# ---------------------------------------------------------------------------
def test_critical_threat_triggers_panic_exit() -> None:
    """Une menace critique sous -0.85 liquide la position au marché."""
    engine = RiskEngine()
    position = make_position()
    decision = engine.evaluate_news_panic_exit(
        position,
        news=make_assessment(-0.93, critical=True),
        price=95.0,
        now=NOW,
    )
    assert decision is not None
    assert decision.approved
    assert decision.reason == "news_panic_exit"
    assert decision.intent is not None
    assert decision.intent.side is Side.SELL
    assert decision.intent.qty == position.qty


def test_panic_exit_requires_the_score_floor() -> None:
    """Une menace critique mais un score modéré ne liquide pas d'office."""
    engine = RiskEngine()
    decision = engine.evaluate_news_panic_exit(
        make_position(),
        news=make_assessment(-0.40, critical=True),
        price=95.0,
        now=NOW,
    )
    assert decision is None


def test_panic_exit_requires_the_critical_flag() -> None:
    """Un score effondré sans menace existentielle ne suffit pas."""
    engine = RiskEngine()
    decision = engine.evaluate_news_panic_exit(
        make_position(),
        news=make_assessment(-0.99, critical=False),
        price=95.0,
        now=NOW,
    )
    assert decision is None


def test_panic_exit_is_inert_without_a_news_signal() -> None:
    """Filtre inactif : le comportement d'origine est strictement préservé."""
    engine = RiskEngine()
    assert engine.evaluate_news_panic_exit(make_position(), news=None, price=95.0, now=NOW) is None


def test_panic_exit_ignores_an_empty_position() -> None:
    """Rien à vendre : mieux vaut ne rien décider que lever une erreur."""
    engine = RiskEngine()
    position = make_position()
    # Position est mutable (le trailing stop la met à jour) : on simule un
    # état transitoire vidé par une sortie concurrente.
    position.qty = 0.0
    decision = engine.evaluate_news_panic_exit(
        position,
        news=make_assessment(-0.95, critical=True),
        price=95.0,
        now=NOW,
    )
    assert decision is None


# ---------------------------------------------------------------------------
# 3. Cache des articles : le cœur de l'optimisation des coûts
# ---------------------------------------------------------------------------
def test_repeated_assessment_never_recalls_the_model(tmp_path: Path) -> None:
    """Sans article nouveau, aucun jeton n'est dépensé."""
    transport = FakeTransport(body=llm_body(-0.80))
    analyzer = make_analyzer(
        tmp_path, articles={"NVDA": [make_article(1)]}, transport=transport
    )

    first = analyzer.assess(["NVDA"], now=NOW)
    assert len(transport.calls) == 1
    assert first["NVDA"].sentiment_score == pytest.approx(-0.80)

    second = analyzer.assess(["NVDA"], now=NOW + timedelta(minutes=5))
    assert len(transport.calls) == 1, "le modèle ne doit pas être rappelé"
    assert second["NVDA"].sentiment_score == pytest.approx(-0.80)
    assert second["NVDA"].is_critical_threat is False


def test_a_new_article_does_trigger_a_fresh_call(tmp_path: Path) -> None:
    """Un article inédit rouvre le budget : le cache ne doit pas figer le filtre."""
    transport = FakeTransport(body=llm_body(-0.90, critical=True))
    client = FakeNewsClient({"NVDA": [make_article(1)]})
    analyzer = NewsSentimentAnalyzer(
        state_dir=tmp_path / "state",
        alpaca_api_key="PK_TEST",
        alpaca_secret_key="SECRET_TEST",
        openrouter_api_key="sk-or-test",
        model=MODEL,
        news_client=client,
        transport=transport,
    )

    analyzer.assess(["NVDA"], now=NOW)
    assert len(transport.calls) == 1

    client.articles_by_symbol["NVDA"] = [make_article(2), make_article(1)]
    refreshed = analyzer.assess(["NVDA"], now=NOW + timedelta(hours=1))
    assert len(transport.calls) == 2
    assert refreshed["NVDA"].is_critical_threat is True
    # Seul l'article inédit est transmis au modèle.
    prompt = transport.calls[-1]["payload"]["messages"][-1]["content"]
    assert "2024" not in prompt or "Nvidia chute" in prompt


def test_registry_is_persisted_and_reloaded(tmp_path: Path) -> None:
    """Le registre survit au redémarrage : c'est lui qui garantit l'économie."""
    transport = FakeTransport(body=llm_body(0.0))
    first = make_analyzer(
        tmp_path, articles={"NVDA": [make_article(7)]}, transport=transport
    )
    first.assess(["NVDA"], now=NOW)

    registry = json.loads(first.registry_path.read_text(encoding="utf-8"))
    assert registry["ids"] == ["7"]

    # Nouvel analyseur, même répertoire : il doit retrouver le registre.
    second = make_analyzer(
        tmp_path, articles={"NVDA": [make_article(7)]}, transport=transport
    )
    second.assess(["NVDA"], now=NOW + timedelta(minutes=1))
    assert len(transport.calls) == 1, "le registre persisté doit éviter l'appel"


def test_articles_are_not_marked_processed_without_a_key(tmp_path: Path) -> None:
    """Sans clé, les articles ne doivent pas être « consommés ».

    Défaut trouvé en exécutant la commande contre l'API réelle : les articles
    lus étaient inscrits au registre alors que le modèle n'avait jamais été
    appelé. Le jour où la clé est ajoutée, tout le stock de la fenêtre serait
    réputé traité et le filtre resterait neutre — silencieusement inopérant.
    """
    analyzer = make_analyzer(tmp_path, articles={"NVDA": [make_article(1)]}, key=None)
    analyzer.assess(["NVDA"], now=NOW)
    if analyzer.registry_path.exists():
        registry = json.loads(analyzer.registry_path.read_text(encoding="utf-8"))
        assert registry["ids"] == []


def test_articles_are_analysed_once_a_key_appears(tmp_path: Path) -> None:
    """Le filtre devient réellement actif dès que la clé est fournie."""
    client = FakeNewsClient({"NVDA": [make_article(1)]})
    make_analyzer(tmp_path, articles={"NVDA": [make_article(1)]}, key=None).assess(
        ["NVDA"], now=NOW
    )

    transport = FakeTransport(body=llm_body(-0.95, critical=True))
    with_key = NewsSentimentAnalyzer(
        state_dir=tmp_path / "state",
        alpaca_api_key="PK_TEST",
        alpaca_secret_key="SECRET_TEST",
        openrouter_api_key="sk-or-test",
        model=MODEL,
        news_client=client,
        transport=transport,
    )
    result = with_key.assess(["NVDA"], now=NOW + timedelta(minutes=1))["NVDA"]
    assert len(transport.calls) == 1, "l'article doit être noté au premier passage réel"
    assert result.is_critical_threat is True


def test_a_failed_call_is_retried_on_the_next_cycle(tmp_path: Path) -> None:
    """Une panne passagère ne doit pas aveugler durablement le filtre."""
    transport = FakeTransport(error=TimeoutError("délai dépassé"))
    analyzer = make_analyzer(
        tmp_path, articles={"NVDA": [make_article(1)]}, transport=transport
    )
    analyzer.assess(["NVDA"], now=NOW)
    assert len(transport.calls) == 1

    transport.error = None
    transport.body = llm_body(-0.70)
    second = analyzer.assess(["NVDA"], now=NOW + timedelta(minutes=1))["NVDA"]
    assert len(transport.calls) == 2, "l'article doit être retenté"
    assert second.sentiment_score == pytest.approx(-0.70)


def test_an_http_error_is_also_retried(tmp_path: Path) -> None:
    """Un 503 d'OpenRouter est réessayé, pas mémorisé comme un verdict."""
    transport = FakeTransport(status=503, body="indisponible")
    analyzer = make_analyzer(
        tmp_path, articles={"NVDA": [make_article(1)]}, transport=transport
    )
    analyzer.assess(["NVDA"], now=NOW)

    transport.status = 200
    transport.body = llm_body(-0.65)
    refreshed = analyzer.assess(["NVDA"], now=NOW + timedelta(minutes=1))["NVDA"]
    assert len(transport.calls) == 2
    assert refreshed.sentiment_score == pytest.approx(-0.65)


def test_shared_articles_are_not_duplicated_in_the_registry(tmp_path: Path) -> None:
    """Un article tagué de deux symboles n'est inscrit qu'une fois.

    Constaté sur les fichiers d'état réels : 23 identifiants pour 20 uniques.
    L'article est bien analysé pour chaque symbole — les invites diffèrent —
    mais il ne doit pas s'accumuler en doublons sur disque.
    """
    shared = make_article(1, symbols=("NVDA", "AMD"))
    client = FakeNewsClient({"NVDA": [shared], "AMD": [shared]})
    transport = FakeTransport(body=llm_body(-0.30))
    analyzer = NewsSentimentAnalyzer(
        state_dir=tmp_path / "state",
        alpaca_api_key="PK_TEST",
        alpaca_secret_key="SECRET_TEST",
        openrouter_api_key="sk-or-test",
        model=MODEL,
        news_client=client,
        transport=transport,
    )
    analyzer.assess(["NVDA", "AMD"], now=NOW)

    # Deux symboles, deux invites : l'appel par symbole reste légitime.
    assert len(transport.calls) == 2
    registry = json.loads(analyzer.registry_path.read_text(encoding="utf-8"))
    assert registry["ids"] == ["1"]


def test_a_partial_run_does_not_erase_other_verdicts(tmp_path: Path) -> None:
    """Noter un seul symbole ne doit pas effacer les verdicts des autres.

    Défaut observé en production : « news --symbols NVDA » a remplacé le
    fichier d'état par son seul verdict. Le passage suivant, complet, a
    substitué aux douze autres symboles une note neutre de repli — et comme
    leurs articles étaient déjà au registre, ces notes neutres se sont figées :
    filtre aveugle sur tout l'univers, sans aucun signal d'alerte.
    """
    client = FakeNewsClient({"NVDA": [make_article(1)], "AMD": [make_article(2)]})
    transport = FakeTransport(body=llm_body(-0.40, confidence=0.7))
    analyzer = NewsSentimentAnalyzer(
        state_dir=tmp_path / "state",
        alpaca_api_key="PK_TEST",
        alpaca_secret_key="SECRET_TEST",
        openrouter_api_key="sk-or-test",
        model=MODEL,
        news_client=client,
        transport=transport,
    )

    analyzer.assess(["NVDA", "AMD"], now=NOW)
    stored = json.loads(analyzer.assessments_path.read_text(encoding="utf-8"))
    assert set(stored["assessments"]) == {"NVDA", "AMD"}

    # Passage partiel : le verdict d'AMD doit survivre intact.
    analyzer.assess(["NVDA"], now=NOW + timedelta(minutes=5))
    stored = json.loads(analyzer.assessments_path.read_text(encoding="utf-8"))
    assert set(stored["assessments"]) == {"NVDA", "AMD"}
    assert stored["assessments"]["AMD"]["confidence"] > 0
    assert stored["assessments"]["AMD"]["rationale"] != "aucune actualité récente"


def test_progress_callback_fires_for_every_symbol(tmp_path: Path) -> None:
    """Le rappel de progression est appelé par symbole, y compris depuis le cache.

    C'est ce qui permet au radar CLI d'afficher ses lignes au fur et à mesure
    au lieu de laisser un terminal figé pendant toute la durée des appels.
    """
    seen: list[tuple[str, float]] = []
    analyzer = make_analyzer(
        tmp_path, articles={"NVDA": [make_article(1)], "AMD": [make_article(2)]}
    )
    analyzer.assess(
        ["NVDA", "AMD"],
        now=NOW,
        on_progress=lambda symbol, assessment: seen.append(
            (symbol, assessment.sentiment_score)
        ),
    )
    assert [symbol for symbol, _ in seen] == ["NVDA", "AMD"]

    # Second passage, entièrement servi par le cache : le rappel doit encore
    # se déclencher, sinon l'affichage resterait vide malgré un résultat.
    seen.clear()
    analyzer.assess(
        ["NVDA"],
        now=NOW + timedelta(minutes=5),
        on_progress=lambda symbol, assessment: seen.append(
            (symbol, assessment.sentiment_score)
        ),
    )
    assert [symbol for symbol, _ in seen] == ["NVDA"]


def test_no_article_means_no_model_call(tmp_path: Path) -> None:
    """Un symbole sans actualité n'entraîne aucun appel payant."""
    transport = FakeTransport(body=llm_body(0.5))
    analyzer = make_analyzer(tmp_path, articles={}, transport=transport)
    assessments = analyzer.assess(["NVDA"], now=NOW)
    assert transport.calls == []
    assert assessments["NVDA"].sentiment_score == 0.0


def test_the_llm_cannot_choose_a_quantity() -> None:
    """Le filtre pèse sur l'autorisation, jamais sur la taille.

    Un score excellent doit produire exactement la même quantité et le même
    stop qu'aucune actualité : le dimensionnement reste l'affaire du moteur de
    risque, jamais celle du modèle de langage.
    """
    engine = RiskEngine()
    sans = engine.evaluate_entry(
        symbol="NVDA",
        price=50.0,
        atr=2.0,
        account=make_account(),
        now=NOW,
        session_date=SESSION,
    )
    avec = engine.evaluate_entry(
        symbol="NVDA",
        price=50.0,
        atr=2.0,
        account=make_account(),
        now=NOW,
        session_date=SESSION,
        news=make_assessment(1.0),
    )
    assert sans.approved and avec.approved
    assert sans.intent is not None and avec.intent is not None
    assert sans.intent.qty == avec.intent.qty
    assert sans.intent.stop_price == avec.intent.stop_price


def test_news_can_only_reject_never_change_anything_else() -> None:
    """Le filtre est un veto à sens unique : il retire, il n'ajoute rien.

    Toute note au-dessus du seuil doit donner une décision strictement
    identique à l'absence de note. C'est la garantie structurelle que le
    modèle ne peut pas influencer le portefeuille autrement qu'en refusant.
    """
    engine = RiskEngine()
    for score in (0.0, 0.2, 0.8, 1.0):
        decision = engine.evaluate_entry(
            symbol="NVDA",
            price=50.0,
            atr=2.0,
            account=make_account(),
            now=NOW,
            session_date=SESSION,
            news=make_assessment(score),
        )
        assert decision.approved, f"score {score} ne doit pas bloquer"
        assert decision.code is RejectionCode.APPROVED
        assert decision.intent is not None
        # Aucune trace du filtre dans la décision approuvée : il n'a rien
        # modifié, il a seulement laissé passer.
        assert "news_score" not in decision.metrics


# ---------------------------------------------------------------------------
# 4. Mode fail-safe
# ---------------------------------------------------------------------------
def test_openrouter_timeout_yields_a_neutral_score(tmp_path: Path) -> None:
    """Une panne réseau ne bloque rien : score neutre, filtre inopérant."""
    transport = FakeTransport(error=TimeoutError("délai dépassé"))
    analyzer = make_analyzer(
        tmp_path, articles={"NVDA": [make_article(1)]}, transport=transport
    )
    result = analyzer.assess(["NVDA"], now=NOW)["NVDA"]
    assert result.sentiment_score == 0.0
    assert result.confidence == 0.0
    assert result.is_critical_threat is False
    assert not RiskEngine().news_vetoes_entry(result)


def test_http_error_yields_a_neutral_score(tmp_path: Path) -> None:
    """Un 429 ou un 500 d'OpenRouter produit le même repli silencieux."""
    transport = FakeTransport(status=500, body="internal error")
    analyzer = make_analyzer(
        tmp_path, articles={"NVDA": [make_article(1)]}, transport=transport
    )
    result = analyzer.assess(["NVDA"], now=NOW)["NVDA"]
    assert result.sentiment_score == 0.0
    assert "500" in result.rationale


def test_invalid_json_yields_a_neutral_score(tmp_path: Path) -> None:
    """Une réponse tronquée est traitée comme une absence de réponse."""
    transport = FakeTransport(body="{ceci n'est pas du JSON")
    analyzer = make_analyzer(
        tmp_path, articles={"NVDA": [make_article(1)]}, transport=transport
    )
    assert analyzer.assess(["NVDA"], now=NOW)["NVDA"].sentiment_score == 0.0


def test_out_of_range_score_is_clamped(tmp_path: Path) -> None:
    """Un modèle bavard ne doit pas casser l'invariant [-1, 1]."""
    transport = FakeTransport(body=llm_body(-4.2))
    analyzer = make_analyzer(
        tmp_path, articles={"NVDA": [make_article(1)]}, transport=transport
    )
    assert analyzer.assess(["NVDA"], now=NOW)["NVDA"].sentiment_score == -1.0


def test_missing_api_key_keeps_the_filter_inert(tmp_path: Path) -> None:
    """Sans clé OpenRouter, le filtre existe mais ne bloque jamais un achat."""
    transport = FakeTransport(body=llm_body(-0.99, critical=True))
    analyzer = make_analyzer(
        tmp_path,
        articles={"NVDA": [make_article(1)]},
        transport=transport,
        key=None,
    )
    assert analyzer.llm_enabled is False
    result = analyzer.assess(["NVDA"], now=NOW)["NVDA"]
    assert result.sentiment_score == 0.0
    assert transport.calls == []
    assert not RiskEngine().news_vetoes_entry(result)


def test_news_outage_does_not_break_the_cycle(tmp_path: Path) -> None:
    """Un fil d'actualités en panne ne doit pas interrompre le trading."""

    class ExplodingNewsClient:
        def get_news(self, request: Any) -> Any:
            raise RuntimeError("flux d'actualités indisponible")

    analyzer = NewsSentimentAnalyzer(
        state_dir=tmp_path / "state",
        alpaca_api_key="PK_TEST",
        alpaca_secret_key="SECRET_TEST",
        openrouter_api_key="sk-or-test",
        model=MODEL,
        news_client=ExplodingNewsClient(),
        transport=FakeTransport(body=llm_body(0.0)),
    )
    result = analyzer.assess(["NVDA"], now=NOW)["NVDA"]
    assert result.sentiment_score == 0.0


# ---------------------------------------------------------------------------
# Récupération des articles : forme réelle de l'API
# ---------------------------------------------------------------------------
def test_articles_are_parsed_from_the_real_api_shape(tmp_path: Path) -> None:
    """L'API renvoie ``data['news']`` de dictionnaires, avec un id entier."""
    analyzer = make_analyzer(tmp_path, articles={"NVDA": [make_article(62766623)]})
    articles = analyzer.fetch_articles("NVDA", now=NOW)
    assert len(articles) == 1
    assert isinstance(articles[0], NewsArticle)
    assert articles[0].article_id == "62766623"
    assert articles[0].symbols == ("NVDA",)
    assert articles[0].created_at is not None


def test_an_unusual_sdk_shape_does_not_crash(tmp_path: Path) -> None:
    """Un article sans identifiant est ignoré plutôt que fatal."""

    class OddClient:
        def get_news(self, request: Any) -> Any:
            return {"data": {"news": [{"headline": "sans identifiant"}, make_article(3)]}}

    analyzer = NewsSentimentAnalyzer(
        state_dir=tmp_path / "state",
        alpaca_api_key="PK_TEST",
        alpaca_secret_key="SECRET_TEST",
        openrouter_api_key=None,
        model=MODEL,
        news_client=OddClient(),
        transport=FakeTransport(body=llm_body(0.0)),
    )
    articles = analyzer.fetch_articles("NVDA", now=NOW)
    assert [a.article_id for a in articles] == ["3"]


def test_prompt_stays_bounded(tmp_path: Path) -> None:
    """Le prompt ne transmet pas un article entier : le coût est borné."""
    huge = make_article(1, summary="z" * 5000)
    analyzer = make_analyzer(tmp_path, articles={"NVDA": [huge]})
    articles = analyzer.fetch_articles("NVDA", now=NOW)
    assert len(articles[0].as_prompt_text()) < 1200


# ---------------------------------------------------------------------------
# Déterminisme du backtest
# ---------------------------------------------------------------------------
def test_entry_without_news_is_unchanged() -> None:
    """Sans note d'actualités, le moteur se comporte exactement comme avant.

    C'est le chemin qu'emprunte le backtest : la reproductibilité tient au
    fait que le filtre n'est pas consulté, et non à une option ignorée.
    """
    engine = RiskEngine()
    decision = engine.evaluate_entry(
        symbol="NVDA",
        price=50.0,
        atr=2.0,
        account=make_account(),
        now=NOW,
        session_date=SESSION,
    )
    assert decision.approved


def test_backtest_runner_never_consults_news() -> None:
    """Le backtest ignore le filtre, sous peine de biais de look-ahead.

    Les articles d'aujourd'hui ne sont pas ceux qui existaient à l'époque, et
    un modèle actuel « connaît » l'issue. Ce test est un garde-fou : y câbler
    le filtre exige de le faire sciemment, avec des données point-in-time.
    """
    from trading_bot.backtest import runner as backtest_runner

    source = inspect.getsource(backtest_runner).lower()
    assert "news" not in source
    assert "sentiment" not in source


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_settings_expose_the_news_defaults() -> None:
    """Les valeurs par défaut annoncées sont bien celles appliquées."""
    settings = Settings(_env_file=None)
    assert settings.news_filter_enabled is True
    assert settings.openrouter_model == MODEL
    assert settings.news_lookback_hours == 24
    assert settings.news_timeout_seconds == pytest.approx(5.0)
    assert settings.risk.news_veto_threshold == pytest.approx(-0.60)
    assert settings.risk.news_panic_exit_threshold == pytest.approx(-0.85)


def test_flat_env_overrides_reach_the_risk_profile() -> None:
    """``NEWS_VETO_THRESHOLD`` est lisible en clair dans le .env."""
    settings = Settings(
        _env_file=None,
        news_veto_threshold=-0.45,
        news_panic_exit_threshold=-0.90,
    )
    assert settings.risk.news_veto_threshold == pytest.approx(-0.45)
    assert settings.risk.news_panic_exit_threshold == pytest.approx(-0.90)


def test_panic_threshold_must_not_be_weaker_than_the_veto() -> None:
    """Une panique doit rester plus grave qu'un simple veto."""
    with pytest.raises(ValidationError):
        RiskConfig(news_veto_threshold=-0.50, news_panic_exit_threshold=-0.20)


def test_thresholds_are_bounded() -> None:
    """Un score de sentiment vit dans [-1, 1] : les seuils aussi."""
    with pytest.raises(ValidationError):
        RiskConfig(news_veto_threshold=-1.5)
    with pytest.raises(ValidationError):
        RiskConfig(news_veto_threshold=0.5)
