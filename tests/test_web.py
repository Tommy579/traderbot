"""Tests de l'interface web locale (serveur éphémère, courtier simulé)."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from datetime import UTC, datetime
from http.server import ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

from trading_bot.config import Settings
from trading_bot.data.loader import DataError
from trading_bot.execution.alpaca_client import (
    AccountSnapshot,
    AssetInfo,
    MarketClosure,
    MarketStatus,
    OrderReceipt,
)
from trading_bot.history import HistoryStore
from trading_bot.main import PersistedState, StateStore
from trading_bot.risk.engine import OrderIntent, Position
from trading_bot.web.app import WebContext, build_handler
from trading_bot.web.jobs import JobManager

NOW = datetime(2024, 1, 10, 15, 0, tzinfo=UTC)


class FakeWebBroker:
    """Courtier complet : compte, horloge, positions et barres par symbole."""

    def __init__(self, bars: dict[str, pd.DataFrame], *, market_open: bool = True) -> None:
        self.bars = bars
        self.market_open = market_open
        self.orders: list[OrderIntent] = []

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(equity=100_000.0, cash=100_000.0, buying_power=100_000.0)

    def is_market_open(self) -> bool:
        return self.market_open

    def market_status(self) -> MarketStatus:
        if self.market_open:
            return MarketStatus(is_open=True)
        return MarketStatus(is_open=False, closure=MarketClosure.OUTSIDE_HOURS)

    def get_positions(self) -> dict[str, Position]:
        return {}

    def get_latest_prices(self, symbols) -> dict[str, float]:
        return {
            symbol: float(self.bars[symbol]["close"].iloc[-1])
            for symbol in symbols
            if symbol in self.bars
        }

    def fetch_bars(self, symbol, *, timeframe="1Day", lookback_days=400, now=None):
        return self.bars[symbol]

    def submit(self, intent, *, now=None) -> OrderReceipt:
        self.orders.append(intent)
        return OrderReceipt(
            order_id=f"order-{len(self.orders)}",
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            status="filled",
            filled_qty=intent.qty,
            avg_fill_price=100.0,
            submitted_at=now or NOW,
        )

    def wait_for_fill(self, order_id, *, timeout_seconds=30.0) -> OrderReceipt:
        """Rend l'ordre rempli : même quantité que l'intention, comme Alpaca.

        Un ordre partiellement exécuté n'est jamais au statut « filled » (le
        client lève ``OrderRejectedError``/``OrderTimeoutError``), donc un
        remplissage complet porte exactement la quantité approuvée.
        """
        assert self.orders
        intent = self.orders[-1]
        return OrderReceipt(
            order_id=order_id,
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            status="filled",
            filled_qty=intent.qty,
            avg_fill_price=100.0,
            submitted_at=NOW,
        )


def make_context(
    settings: Settings,
    bars: dict[str, pd.DataFrame],
    *,
    broker: FakeWebBroker | None = None,
) -> WebContext:
    effective = broker or FakeWebBroker(bars)
    return WebContext(
        settings=settings,
        broker_factory=lambda _settings: effective,  # type: ignore[arg-type,return-value]
        env_path=settings.state_dir / ".env",
        install_script=Path("/inexistant/install.sh"),
        jobs=JobManager(),
        backtest_lock=threading.Lock(),
    )


def start_server(context: WebContext) -> tuple[ThreadingHTTPServer, int]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(context))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def request(
    port: int,
    path: str,
    *,
    method: str = "GET",
    body: dict | None = None,
    headers: dict | None = None,
):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def wait_job(port: int, job_id: str, *, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _, snapshot = request(port, f"/api/job/{job_id}")
        if snapshot["status"] != "running":
            return snapshot
        time.sleep(0.1)
    raise AssertionError("la tâche n'a pas terminé à temps")


@pytest.fixture
def web_bars(bars_factory) -> dict[str, pd.DataFrame]:
    return {
        "AAPL": bars_factory([100.0 + i * 0.1 for i in range(80)]),
        "SPY": bars_factory([400.0 + i * 0.2 for i in range(80)]),
    }


def test_status_and_security_guards(settings, web_bars) -> None:
    httpd, port = start_server(make_context(settings, web_bars))
    try:
        code, payload = request(port, "/api/status")
        assert code == 200
        assert payload["ok"] is True
        assert payload["account"]["equity"] == pytest.approx(100_000.0)
        assert payload["settings"]["benchmark"] == "SPY"
        assert "timer_installed" in payload["automation"]

        code, payload = request(port, "/api/status", headers={"Origin": "https://mal.site"})
        assert code == 403
        code, payload = request(port, "/api/status", headers={"Host": "evil.example"})
        assert code == 403
    finally:
        httpd.shutdown()


def test_same_origin_requests_are_allowed(settings, web_bars) -> None:
    """Le navigateur envoie ``Origin`` même en same-origin : ce n'est pas croix-site.

    Régression : refuser tout ``Origin`` faisait échouer chaque bouton de la
    page (« Requête croix-site refusée. »), car ``fetch`` en POST en ajoute un.
    """
    httpd, port = start_server(make_context(settings, web_bars))
    local_origin = f"http://127.0.0.1:{port}"
    try:
        code, payload = request(port, "/api/status", headers={"Origin": local_origin})
        assert code == 200
        assert payload["ok"] is True

        code, payload = request(
            port,
            "/api/check",
            method="POST",
            body={"strategy": "ema_cross", "symbols": ["AAPL"]},
            headers={"Origin": local_origin},
        )
        assert code == 200
        assert payload["ok"] is True

        for origin in ("http://localhost:8484", "http://[::1]:8484"):
            code, _ = request(port, "/api/status", headers={"Origin": origin})
            assert code == 200, origin

        # Hôtes locaux IPv4 et IPv6 acceptés
        for host in ("localhost:8484", "127.0.0.1:8484", "[::1]:8484", "[::1]"):
            code, _ = request(port, "/api/status", headers={"Host": host})
            assert code == 200, host

        for origin in ("https://mal.site", "null", "http://127.0.0.1.mal.site"):
            code, payload = request(port, "/api/status", headers={"Origin": origin})
            assert code == 403, origin
            assert payload["error"] == "Requête croix-site refusée."
    finally:
        httpd.shutdown()


def test_check_job_returns_signals(settings, web_bars) -> None:
    httpd, port = start_server(make_context(settings, web_bars))
    try:
        code, payload = request(
            port,
            "/api/check",
            method="POST",
            body={"strategy": "ema_cross", "symbols": ["AAPL"]},
        )
        assert code == 200
        snapshot = wait_job(port, payload["job"]["id"])
        assert snapshot["status"] == "done"
        assert snapshot["result"]["market_open"] is True
        rows = snapshot["result"]["signals"]
        assert len(rows) == 1
        assert rows[0]["symbol"] == "AAPL"
        assert rows[0]["action"] in {"BUY", "SELL", "HOLD"}
    finally:
        httpd.shutdown()


def test_cycle_job_dry_run(settings, web_bars) -> None:
    httpd, port = start_server(make_context(settings, web_bars))
    try:
        _code, payload = request(
            port,
            "/api/cycle",
            method="POST",
            body={"strategy": "ema_cross", "dry_run": True},
        )
        snapshot = wait_job(port, payload["job"]["id"])
        assert snapshot["status"] == "done"
        report = snapshot["result"]
        assert report["dry_run"] is True
        assert not report["skipped"]
        assert report["signals"]["AAPL"] in {"BUY", "SELL", "HOLD"}
    finally:
        httpd.shutdown()


def test_cycle_job_watch_only(settings, web_bars) -> None:
    httpd, port = start_server(make_context(settings, web_bars))
    try:
        _code, payload = request(
            port,
            "/api/cycle",
            method="POST",
            body={"strategy": "ema_cross", "watch_only": True},
        )
        snapshot = wait_job(port, payload["job"]["id"])
        assert snapshot["status"] == "done"
        report = snapshot["result"]
        assert report["mode"] == "watch_only"
        assert not report["skipped"]
    finally:
        httpd.shutdown()


def test_backtest_job_with_fake_provider(
    settings, web_bars, monkeypatch, bars_factory
) -> None:
    import trading_bot.web.app as webapp

    long_bars = {"AAA": bars_factory([100.0] * 208 + [105.0, 106.0])}
    monkeypatch.setattr(
        webapp, "CachedProvider", lambda provider, cache_dir, **kw: provider
    )
    monkeypatch.setattr(
        webapp,
        "YFinanceProvider",
        lambda: type(
            "P",
            (),
            {
                "fetch": staticmethod(
                    lambda symbol, start, end, interval="1d": long_bars[symbol]
                )
            },
        )(),
    )
    httpd, port = start_server(make_context(settings, web_bars))
    try:
        _code, payload = request(
            port,
            "/api/backtest",
            method="POST",
            body={
                "strategy": "donchian_breakout",
                "symbols": ["AAA"],
                "start": "2023-01-01",
            },
        )
        snapshot = wait_job(port, payload["job"]["id"], timeout=60)
        assert snapshot["status"] == "done", snapshot.get("error")
        result = snapshot["result"]
        assert result["metrics"]["open_positions_at_end"] >= 1
        assert len(result["equity_points"]) >= 2
    finally:
        httpd.shutdown()


def test_config_update_roundtrip(settings, web_bars) -> None:
    context = make_context(settings, web_bars)
    context.env_path.parent.mkdir(parents=True, exist_ok=True)
    context.env_path.write_text("ALPACA_API_KEY=PK_TEST\n\n", encoding="utf-8")
    httpd, port = start_server(context)
    try:
        code, payload = request(
            port,
            "/api/config",
            method="POST",
            body={"symbols": ["MSFT", "NVDA"], "webhook_url": "https://example.com/hook"},
        )
        assert code == 200 and payload["ok"] is True
        assert payload["webhook_enabled"] is True
        fresh = Settings(_env_file=str(context.env_path))
        assert fresh.symbols == ["MSFT", "NVDA"]
        assert fresh.webhook_url is not None
        assert "WEBHOOK_URL=https://example.com/hook" in context.env_path.read_text(
            encoding="utf-8"
        )

        # Désactivation du webhook via chaîne vide
        code, payload = request(
            port,
            "/api/config",
            method="POST",
            body={"webhook_url": ""},
        )
        assert code == 200 and payload["ok"] is True
        assert payload["webhook_enabled"] is False
        env_content = context.env_path.read_text(encoding="utf-8")
        assert "# WEBHOOK_URL=" in env_content

        code, payload = request(
            port, "/api/config", method="POST", body={"symbols": ["pas!!valide"]}
        )
        assert payload["ok"] is False
        restored = Settings(_env_file=str(context.env_path))
        assert restored.symbols == ["MSFT", "NVDA"]  # .env restauré
    finally:
        httpd.shutdown()


def test_cycle_concurrency_guard(settings, web_bars) -> None:
    """Deux cycles ne peuvent pas tourner simultanément."""
    from trading_bot.web.jobs import JobStatus

    context = make_context(settings, web_bars)
    httpd, port = start_server(context)
    try:
        # On simule un job cycle déjà en cours
        job = context.jobs.submit("cycle", "test", lambda: time.sleep(2) or {})
        assert job.status is JobStatus.RUNNING

        code, payload = request(port, "/api/cycle", method="POST", body={"dry_run": True})
        assert code == 200
        assert payload["ok"] is False
        assert "déjà en cours" in payload["error"]
    finally:
        httpd.shutdown()


def test_helpers_ipv6_and_timestamp() -> None:
    from trading_bot.web.app import _extract_host, _parse_systemd_timestamp

    assert _extract_host("127.0.0.1:8484") == "127.0.0.1"
    assert _extract_host("[::1]:8484") == "::1"
    assert _extract_host("[::1]") == "::1"
    assert _extract_host("localhost:8484") == "localhost"

    iso = _parse_systemd_timestamp("Mon 2026-11-02 21:55:00 CET")
    assert iso == "2026-11-02T21:55:00"
    assert _parse_systemd_timestamp("0") is None
    assert _parse_systemd_timestamp("n/a") is None


def test_permanent_halt_reset_via_api(settings, web_bars) -> None:
    StateStore(settings.state_dir / "state.json").save(
        PersistedState(
            trading_day="2024-01-10",
            halted=True,
            halt_scope="permanent",
            halt_reason="drawdown 16 %",
        )
    )
    httpd, port = start_server(make_context(settings, web_bars))
    try:
        _code, payload = request(
            port, "/api/halt/reset", method="POST", body={"confirm": False}
        )
        assert payload["ok"] is False  # sans confirmation ni raison

        _code, payload = request(
            port,
            "/api/halt/reset",
            method="POST",
            body={"confirm": True, "reason": "J'ai vérifié les positions."},
        )
        assert payload["ok"] is True
        state = StateStore(settings.state_dir / "state.json").load()
        assert state.halted is False
    finally:
        httpd.shutdown()


def test_history_endpoint_on_empty_store(settings, web_bars) -> None:
    httpd, port = start_server(make_context(settings, web_bars))
    try:
        code, payload = request(port, "/api/history")
        assert code == 200
        assert payload["ok"] is True
        assert payload["entries"] == []
    finally:
        httpd.shutdown()


def test_history_endpoint_returns_recorded_cycles(settings, web_bars) -> None:
    store = HistoryStore(settings.state_dir / "history.jsonl")
    store.record(
        {"session_date": "2026-10-29", "skipped": False, "equity": 100_000.0},
        source="timer",
    )
    store.record(
        {"session_date": "2026-10-30", "skipped": False, "equity": 101_000.0},
        source="timer",
    )

    httpd, port = start_server(make_context(settings, web_bars))
    try:
        _code, payload = request(port, "/api/history")
        entries = payload["entries"]
        assert [entry["session_date"] for entry in entries] == [
            "2026-10-30",
            "2026-10-29",
        ]
        assert entries[0]["source"] == "timer"
        assert entries[0]["equity"] == pytest.approx(101_000.0)
    finally:
        httpd.shutdown()


def test_cycle_triggered_from_gui_lands_in_history(settings, web_bars) -> None:
    """Le bouton « cycle » doit alimenter l'historique consulté par la page."""
    context = make_context(settings, web_bars)
    httpd, port = start_server(context)
    try:
        _code, payload = request(port, "/api/cycle", method="POST", body={"dry_run": True})
        assert payload["ok"] is True
        snapshot = wait_job(port, payload["job"]["id"])
        assert snapshot["status"] == "done", snapshot.get("error")

        _code, history = request(port, "/api/history")
        entries = history["entries"]
        assert len(entries) == 1
        assert entries[0]["source"] == "gui"
        assert entries[0]["dry_run"] is True
    finally:
        httpd.shutdown()


def _bump_mtime(path: Path) -> None:
    """Force une empreinte distincte pour que la relecture soit détectée.

    Sans cela, deux écritures rapprochées pourraient porter la même date sur
    un système de fichiers à granularité grossière.
    """
    current = path.stat().st_mtime_ns
    os.utime(path, ns=(current + 10**9, current + 10**9))


def _write_env(path: Path, symbols: str, *, state_dir: Path | None = None) -> None:
    """Écrit un .env minimal, puis force une empreinte distincte.

    ``state_dir`` est reporté dans le fichier pour garder le test hermétique :
    sinon la configuration rechargée retomberait sur le dossier d'état réel
    du projet.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ALPACA_API_KEY=PK_TEST",
        "ALPACA_SECRET_KEY=SECRET_TEST",
        f"SYMBOLS={symbols}",
    ]
    if state_dir is not None:
        lines.append(f"STATE_DIR={state_dir}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _bump_mtime(path)


def test_status_reloads_env_edited_outside_the_gui(settings, web_bars) -> None:
    """Une édition du .env hors interface ne doit pas laisser la page mentir.

    Régression : la page servait la liste chargée au démarrage, alors que le
    bot relit le fichier à chaque cycle — l'utilisateur croyait piloter un
    univers que le robot n'utilisait plus.
    """
    env_path = settings.state_dir / ".env"
    _write_env(env_path, "AAPL,MSFT", state_dir=settings.state_dir)

    httpd, port = start_server(make_context(settings, web_bars))
    try:
        _code, payload = request(port, "/api/status")
        assert payload["settings"]["symbols"] == ["AAPL", "MSFT"]

        _write_env(env_path, "AAPL,MSFT,NVDA", state_dir=settings.state_dir)

        _code, payload = request(port, "/api/status")
        assert payload["settings"]["symbols"] == ["AAPL", "MSFT", "NVDA"]
    finally:
        httpd.shutdown()


def test_check_uses_env_edited_outside_the_gui(settings, web_bars, bars_factory) -> None:
    """Le diagnostic doit porter sur la même liste que celle lue par le bot."""
    env_path = settings.state_dir / ".env"
    _write_env(env_path, "AAPL", state_dir=settings.state_dir)

    bars = {
        "AAPL": bars_factory([100.0 + i * 0.1 for i in range(80)]),
        "MSFT": bars_factory([300.0 + i * 0.2 for i in range(80)]),
    }
    httpd, port = start_server(make_context(settings, bars))
    try:
        _write_env(env_path, "AAPL,MSFT", state_dir=settings.state_dir)

        _code, payload = request(port, "/api/check", method="POST", body={})
        assert payload["ok"] is True
        snapshot = wait_job(port, payload["job"]["id"])
        assert snapshot["status"] == "done", snapshot.get("error")

        result = snapshot["result"]
        assert [row["symbol"] for row in result["signals"]] == ["AAPL", "MSFT"]
    finally:
        httpd.shutdown()


def test_check_excludes_benchmark_asked_as_a_symbol(settings, web_bars) -> None:
    """SPY demandé explicitement est écarté : c'est la référence, pas un actif.

    Le filtre de configuration couvre l'univers habituel ; celui-ci protège le
    chemin où la liste vient directement de la page (champ de backtest).
    """
    httpd, port = start_server(make_context(settings, web_bars))
    try:
        _code, payload = request(
            port,
            "/api/check",
            method="POST",
            body={"symbols": ["AAPL", "SPY"]},
        )
        assert payload["ok"] is True
        snapshot = wait_job(port, payload["job"]["id"])
        assert snapshot["status"] == "done", snapshot.get("error")

        result = snapshot["result"]
        assert [row["symbol"] for row in result["signals"]] == ["AAPL"]
        assert any("SPY" in error for error in result["errors"])
    finally:
        httpd.shutdown()


def test_config_keeps_webhook_token_when_field_left_blank(settings, web_bars) -> None:
    """Enregistrer les réglages sans ressaisir le jeton ne doit pas l'effacer.

    Régression évitée : accepter un champ vide aurait supprimé
    l'authentification d'un endpoint public à chaque sauvegarde, en silence.
    """
    env_path = settings.state_dir / ".env"
    _write_env(env_path, "AAPL", state_dir=settings.state_dir)
    env_path.write_text(
        env_path.read_text(encoding="utf-8") + "WEBHOOK_TOKEN=secret-existant\n",
        encoding="utf-8",
    )
    _bump_mtime(env_path)

    httpd, port = start_server(make_context(settings, web_bars))
    try:
        _code, payload = request(
            port,
            "/api/config",
            method="POST",
            body={"symbols": ["AAPL"], "webhook_token": ""},
        )
        assert payload["ok"] is True
    finally:
        httpd.shutdown()

    assert "WEBHOOK_TOKEN=secret-existant" in (_env_text(settings) or "")


def test_config_stores_a_new_webhook_token(settings, web_bars) -> None:
    env_path = settings.state_dir / ".env"
    _write_env(env_path, "AAPL", state_dir=settings.state_dir)

    httpd, port = start_server(make_context(settings, web_bars))
    try:
        _code, payload = request(
            port,
            "/api/config",
            method="POST",
            body={"webhook_token": "nouveau-secret"},
        )
        assert payload["ok"] is True
    finally:
        httpd.shutdown()

    assert "WEBHOOK_TOKEN=nouveau-secret" in (_env_text(settings) or "")


def test_check_skips_unknown_symbol_instead_of_failing(settings, web_bars) -> None:
    """Un symbole mal orthographié est écarté et signalé, jamais fatal.

    Le cycle réel du bot tolère déjà cette faute (il ignore le symbole et
    poursuit) : le diagnostic de la page doit se comporter pareil, sinon il
    laisse croire à une panne générale alors qu'un seul nom est en cause.
    """

    class BadSymbolBroker(FakeWebBroker):
        """Courtier dont un symbole inconnu lève ``DataError``."""

        def fetch_bars(self, symbol, *, timeframe="1Day", lookback_days=400, now=None):
            if symbol not in self.bars:
                raise DataError(f"{symbol}: aucune barre disponible")
            return self.bars[symbol]

    broker = BadSymbolBroker(web_bars)
    context = WebContext(
        settings=settings,
        broker_factory=lambda _settings: broker,  # type: ignore[arg-type,return-value]
        env_path=settings.state_dir / ".env",
        install_script=Path("/inexistant/install.sh"),
        jobs=JobManager(),
        backtest_lock=threading.Lock(),
    )
    httpd, port = start_server(context)
    try:
        _code, payload = request(
            port,
            "/api/check",
            method="POST",
            body={"symbols": ["AAPL", "ZZZZZZZZ"]},
        )
        assert payload["ok"] is True
        snapshot = wait_job(port, payload["job"]["id"])
        assert snapshot["status"] == "done", snapshot.get("error")

        result = snapshot["result"]
        assert [row["symbol"] for row in result["signals"]] == ["AAPL"]
        assert result["errors"] == ["ZZZZZZZZ: aucune barre disponible"]
    finally:
        httpd.shutdown()


class RegistryWebBroker(FakeWebBroker):
    """Courtier doté d'un registre d'actifs, comme l'API Alpaca."""

    def __init__(
        self, bars: dict[str, pd.DataFrame], registry: dict[str, AssetInfo]
    ) -> None:
        super().__init__(bars)
        self.registry = registry

    def describe_asset(self, symbol: str) -> AssetInfo | None:
        return self.registry.get(symbol)


ASSET_BOOK: dict[str, AssetInfo] = {
    "AAPL": AssetInfo("AAPL", "Apple Inc. Common Stock", "us_equity", True),
    "MSFT": AssetInfo("MSFT", "Microsoft Corporation Common Stock", "us_equity", True),
    "BTC": AssetInfo("BTC", "Grayscale Bitcoin Mini Trust ETF", "us_equity", True),
    "BTC/USD": AssetInfo("BTC/USD", "Bitcoin  / US Dollar", "crypto", True),
    "HALTED": AssetInfo("HALTED", "Delisted Corp", "us_equity", False),
}


def _registry_context(settings, web_bars) -> WebContext:
    return make_context(settings, web_bars, broker=RegistryWebBroker(web_bars, ASSET_BOOK))


def _env_text(settings) -> str | None:
    path = settings.state_dir / ".env"
    return path.read_text(encoding="utf-8") if path.exists() else None


def test_config_reports_what_the_symbols_really_are(settings, web_bars) -> None:
    """« BTC » doit être annoncé comme l'ETF Grayscale qu'il est."""
    httpd, port = start_server(_registry_context(settings, web_bars))
    try:
        _code, payload = request(
            port, "/api/config", method="POST", body={"symbols": ["AAPL", "BTC"]}
        )
        assert payload["ok"] is True
        resolved = {row["symbol"]: row for row in payload["resolved"]}
        assert resolved["AAPL"]["name"] == "Apple Inc. Common Stock"
        assert resolved["BTC"]["name"] == "Grayscale Bitcoin Mini Trust ETF"
        assert resolved["AAPL"]["asset_class"] == "us_equity"
        assert payload["warning"] is None
    finally:
        httpd.shutdown()


def test_config_refuses_unknown_symbol_without_touching_env(settings, web_bars) -> None:
    """Un symbole inconnu est refusé et ne doit rien écrire sur le disque."""
    before = _env_text(settings)
    httpd, port = start_server(_registry_context(settings, web_bars))
    try:
        _code, payload = request(
            port, "/api/config", method="POST", body={"symbols": ["AAPL", "ZZZZZZZZ"]}
        )
        assert payload["ok"] is False
        assert "ZZZZZZZZ" in payload["error"]
        assert "inconnu d'Alpaca" in payload["error"]
    finally:
        httpd.shutdown()
    assert _env_text(settings) == before


def test_config_refuses_crypto_pair(settings, web_bars) -> None:
    """Le bot ne sait pas placer d'ordre crypto : la paire est refusée.

    Alpaca n'accepte ni ordre ``day`` (entrée) ni stop simple (filet) sur le
    crypto : accepter la paire promettrait un trading qui échouerait.
    """
    before = _env_text(settings)
    httpd, port = start_server(_registry_context(settings, web_bars))
    try:
        _code, payload = request(
            port, "/api/config", method="POST", body={"symbols": ["BTC/USD"]}
        )
        assert payload["ok"] is False
        assert "crypto" in payload["error"]
    finally:
        httpd.shutdown()
    assert _env_text(settings) == before


def test_config_refuses_non_tradable_asset(settings, web_bars) -> None:
    before = _env_text(settings)
    httpd, port = start_server(_registry_context(settings, web_bars))
    try:
        _code, payload = request(
            port, "/api/config", method="POST", body={"symbols": ["HALTED"]}
        )
        assert payload["ok"] is False
        assert "non négociable" in payload["error"]
    finally:
        httpd.shutdown()
    assert _env_text(settings) == before


def test_config_without_registry_still_saves(settings, web_bars) -> None:
    """Un courtier sans registre (fake) ne bloque pas l'enregistrement."""
    httpd, port = start_server(make_context(settings, web_bars))
    try:
        _code, payload = request(
            port, "/api/config", method="POST", body={"symbols": ["AAPL", "MSFT"]}
        )
        assert payload["ok"] is True
        assert payload["symbols"] == ["AAPL", "MSFT"]
        assert payload["resolved"] == []
        assert payload["warning"] is None
    finally:
        httpd.shutdown()


def test_static_assets_and_pwa_manifest(settings, web_bars) -> None:
    httpd, port = start_server(make_context(settings, web_bars))
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/favicon.svg")
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type") == "image/svg+xml"
            assert b"<svg" in resp.read()

        req_manifest = urllib.request.Request(
            f"http://127.0.0.1:{port}/manifest.webmanifest"
        )
        with urllib.request.urlopen(req_manifest, timeout=10) as resp:
            assert resp.status == 200
            assert "application/manifest+json" in resp.headers.get("Content-Type", "")
            manifest = json.loads(resp.read().decode())
            assert manifest["short_name"] == "TradingBot"

        req_ico = urllib.request.Request(f"http://127.0.0.1:{port}/favicon.ico")
        with urllib.request.urlopen(req_ico, timeout=10) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type") == "image/x-icon"
            assert len(resp.read()) > 0
    finally:
        httpd.shutdown()


def test_withdraw_endpoints_and_floors(settings, web_bars) -> None:
    broker = FakeWebBroker(web_bars)
    broker.get_positions = lambda: {  # type: ignore[method-assign]
        "AAPL": Position(
            symbol="AAPL",
            qty=20.0,
            avg_entry_price=100.0,
            current_price=100.0,
            stop_price=90.0,
        )
    }
    broker.cancel_open_orders = lambda sym: 1  # type: ignore[attr-defined]

    context = make_context(settings, web_bars, broker=broker)
    context.env_path.parent.mkdir(parents=True, exist_ok=True)
    context.env_path.write_text("ALPACA_API_KEY=PK_TEST\n", encoding="utf-8")
    httpd, port = start_server(context)
    try:
        # 1. Retrait partiel par pourcentage
        code, payload = request(
            port,
            "/api/withdraw",
            method="POST",
            body={"symbol": "AAPL", "pct": 50.0},
        )
        assert code == 200 and payload["ok"] is True
        assert len(payload["orders"]) == 1
        assert payload["orders"][0]["symbol"] == "AAPL"
        assert payload["orders"][0]["qty"] == 10.0

        # 2. Retrait global avec sanctuarisation du plancher de cash
        code, payload = request(
            port,
            "/api/withdraw",
            method="POST",
            body={"free_cash": 15000.0},
        )
        assert code == 200 and payload["ok"] is True
        env_text = context.env_path.read_text(encoding="utf-8")
        assert "WITHDRAWABLE_CASH_FLOOR_USD=15000.00" in env_text

        # 3. Modification des planchers via /api/config
        code, payload = request(
            port,
            "/api/config",
            method="POST",
            body={"withdrawable_cash_floor_usd": "3000", "equity_hard_floor_usd": "50000"},
        )
        assert code == 200 and payload["ok"] is True
        fresh = Settings(_env_file=str(context.env_path))
        assert fresh.risk.withdrawable_cash_floor_usd == 3000.0
        assert fresh.risk.equity_hard_floor_usd == 50000.0
    finally:
        httpd.shutdown()


def test_automation_view_is_cached_between_requests(monkeypatch) -> None:
    """L'état des timers ne relance pas deux `systemctl` à chaque rafraîchissement."""
    import subprocess
    from types import SimpleNamespace

    from trading_bot.web import app as web_app

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(web_app, "_automation_cache", None)

    first = web_app._automation_view()
    second = web_app._automation_view()

    # Deux timers inspectés une seule fois, pas une fois par appel.
    assert len(calls) == 2
    assert first == second

    web_app._invalidate_automation_cache()
    web_app._automation_view()
    assert len(calls) == 4
