"""Interface web locale du TradingBot, en bibliothèque standard uniquement.

Objectif : permettre à un utilisateur non financier de piloter le bot sans
toucher à la ligne de commande — diagnostic, backtest, cycle paper,
automatisation et réglages, tout en français simple.

Sécurité :

* écoute sur 127.0.0.1 uniquement ;
* toute requête portant un ``Host`` étranger est refusée, et un en-tête
  ``Origin`` n'est accepté que s'il désigne une origine locale : un site
  malveillant visité par l'utilisateur ne peut pas déclencher d'ordre. Le
  navigateur envoie ``Origin`` même sur ses propres POST ``fetch``, donc
  refuser tout ``Origin`` rendrait l'interface inutilisable ;
* seul le paper trading Alpaca est joignable — l'architecture du bot rend
  un ordre réel impossible, quel que soit le bouton cliqué.

Les actions longues (backtest, cycle) tournent en tâche de fond ; la page
interroge leur résultat.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from datetime import time as dt_time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from trading_bot.backtest.runner import BacktestRunner
from trading_bot.bot import (
    TradingBot,
    _assess_news,
    _build_news_analyzer,
    _build_notifier,
    _inject_benchmark,
    _symbol_message,
    build_strategy,
)
from trading_bot.clock import clock_offset_seconds, get_online_now
from trading_bot.config import (
    ConfigurationError,
    Settings,
    configure_logging,
    update_env_file,
)
from trading_bot.data.loader import (
    CachedProvider,
    DataError,
    MarketDataProvider,
    YFinanceProvider,
    drop_incomplete_bar,
    timeframe_interval,
)
from trading_bot.execution.alpaca_client import AlpacaPaperClient, BrokerError
from trading_bot.history import HistoryStore
from trading_bot.main import _withdrawal_payload, apply_equity_rebase, news_effect_label
from trading_bot.risk.engine import RiskEngine
from trading_bot.state import AuditLog, StateStore
from trading_bot.strategies.base import StrategyError
from trading_bot.web.jobs import JobManager
from trading_bot.withdrawal import WithdrawalService

_MAX_BODY_BYTES = 1_000_000
#: Durée de validité de l'état des timers. Chaque calcul lance deux
#: ``systemctl --user show`` : la page interroge le statut toutes les deux
#: secondes, donc sans cache on forkerait un processus par seconde.
_AUTOMATION_CACHE_TTL_SECONDS = 15.0
_automation_cache: tuple[float, dict[str, Any]] | None = None
_automation_lock = threading.Lock()
_CHART_POINTS = 240
_UNIT_SERVICE = Path.home() / ".config/systemd/user/tradingbot.service"
_UNIT_TIMER = Path.home() / ".config/systemd/user/tradingbot.timer"
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def is_local_origin(origin: str) -> bool:
    """Indique si ``origin`` désigne une origine locale (donc non croix-site).

    Args:
        origin: Valeur brute de l'en-tête ``Origin``.

    Returns:
        ``True`` si l'URL est en ``http(s)`` et pointe vers localhost.
    """
    try:
        parsed = urlparse(origin)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    return (parsed.hostname or "").lower() in _LOCAL_HOSTS


def _extract_host(host_header: str | None) -> str:
    """Extrait le nom d'hôte ou l'IP sans port, y compris les adresses IPv6 [::1]."""
    if not host_header:
        return ""
    raw = host_header.strip()
    if raw.startswith("["):
        end = raw.find("]")
        if end != -1:
            return raw[1:end].lower()
    return raw.split(":")[0].lower()


def _parse_systemd_timestamp(value: str) -> str | None:
    """Convertit un timestamp systemd (ex: 'Mon 2026-11-02 21:55:00 CET') en format ISO."""
    if not value or value in {"0", "n/a", "infinity"}:
        return None
    match = re.search(r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})", value)
    if not match:
        return None
    date_part, time_part = match.groups()
    return f"{date_part}T{time_part}"


@dataclass(slots=True)
class WebContext:
    """Dépendances du serveur, injectables pour les tests.

    Args:
        settings: Configuration effective.
        broker_factory: Construit la passerelle Alpaca (fake de test possible).
        env_path: Fichier ``.env`` édité par la page Réglages.
        install_script: Script d'installation du timer quotidien.
        jobs: Gestionnaire de tâches asynchrones.
        backtest_lock: Verrou d'exclusion mutuelle pour les backtests.
        cycle_lock: Verrou d'exclusion mutuelle pour les cycles live / paper.
        config_lock: Verrou de relecture du ``.env``.
        env_mtime: Empreinte du ``.env`` au dernier chargement.
    """

    settings: Settings
    broker_factory: Callable[[Settings], AlpacaPaperClient]
    env_path: Path
    install_script: Path
    jobs: JobManager
    backtest_lock: threading.Lock
    cycle_lock: threading.Lock = field(default_factory=threading.Lock)
    config_lock: threading.Lock = field(default_factory=threading.Lock)
    env_mtime: int | None = None


def _default_broker_factory(settings: Settings) -> AlpacaPaperClient:
    """Passe-relle Alpaca réelle (endpoint paper vérifié à la construction)."""
    return AlpacaPaperClient(settings)


def _friendly(exc: Exception) -> str:
    """Message d'erreur compréhensible, sans stack trace ni jargon."""
    if isinstance(exc, ConfigurationError) and "credentials" in str(exc).lower():
        return (
            "Les clés Alpaca sont absentes du fichier .env "
            "(ALPACA_API_KEY / ALPACA_SECRET_KEY)."
        )
    if isinstance(exc, BrokerError):
        return f"Le courtier Alpaca n'a pas répondu correctement : {exc}"
    if isinstance(exc, DataError):
        return f"Les données de marché sont indisponibles : {exc}"
    if isinstance(exc, StrategyError):
        return f"La stratégie n'a pas pu analyser les données : {exc}"
    if isinstance(exc, ValueError):
        # Les refus volontaires passent par ValueError : leur message est déjà
        # rédigé pour l'utilisateur, ne pas l'affubler d'« erreur inattendue ».
        text = str(exc).strip()
        return text or "Requête invalide."
    text = str(exc).strip() or exc.__class__.__name__
    return f"Erreur inattendue : {text}"


def _positions_view(
    broker: AlpacaPaperClient,
    settings: Settings,
) -> list[dict[str, Any]]:
    """Positions ouvertes, enrichies des stops persistés, en valeurs lisibles."""
    try:
        positions = broker.get_positions()
    except BrokerError:
        return []
    stops = StateStore(settings.state_dir / "state.json").load().stops
    rows: list[dict[str, Any]] = []
    for symbol, position in sorted(positions.items()):
        stored = stops.get(symbol, {})
        rows.append(
            {
                "symbol": symbol,
                "qty": position.qty,
                "entry_price": position.avg_entry_price,
                "current_price": position.current_price,
                "value": position.market_value,
                "pnl": position.unrealized_pnl,
                "pnl_pct": position.unrealized_pnl_pct,
                "stop_price": stored.get("stop_price"),
                "partial_tp_taken": stored.get("partial_tp_taken", False),
                "is_shield": symbol == settings.risk.shield_symbol,
            }
        )
    return rows


def _automation_view() -> dict[str, Any]:
    """État des timers, mis en cache quelques secondes.

    Le tableau de bord interroge ``/api/status`` en boucle ; sans cache, chaque
    requête lancerait deux ``systemctl`` synchrones.
    """
    global _automation_cache
    now = time.monotonic()
    with _automation_lock:
        cached = _automation_cache
        if cached is not None and now - cached[0] < _AUTOMATION_CACHE_TTL_SECONDS:
            return dict(cached[1])

    view = _compute_automation_view()
    with _automation_lock:
        _automation_cache = (now, dict(view))
    return view


def _invalidate_automation_cache() -> None:
    """Force le prochain calcul de l'état des timers (après une installation)."""
    global _automation_cache
    with _automation_lock:
        _automation_cache = None


def _compute_automation_view() -> dict[str, Any]:
    """État des timers (cycle complet et vigie horaire), sans cache."""
    watch_timer_path = _UNIT_TIMER.parent / "tradingbot-watch.timer"
    view: dict[str, Any] = {
        "timer_installed": _UNIT_TIMER.exists() or watch_timer_path.exists(),
        "timer_active": False,
        "watch_timer_active": False,
        "next_run": None,
        "strategy": "ema_cross",
    }
    if _UNIT_SERVICE.exists():
        match = re.search(r"--strategy (\S+)", _UNIT_SERVICE.read_text(encoding="utf-8"))
        if match:
            view["strategy"] = match.group(1)

    if shutil.which("systemctl") is None:
        return view

    for timer_unit, active_key in [
        ("tradingbot.timer", "timer_active"),
        ("tradingbot-watch.timer", "watch_timer_active"),
    ]:
        try:
            res = subprocess.run(
                ["systemctl", "--user", "show", timer_unit],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    key, _, value = line.partition("=")
                    if key == "ActiveState" and value.strip() == "active":
                        view[active_key] = True
                    if key == "NextElapseUSecRealtime" and value.strip():
                        parsed_iso = _parse_systemd_timestamp(value.strip())
                        if parsed_iso and (
                            view["next_run"] is None or parsed_iso < view["next_run"]
                        ):
                            view["next_run"] = parsed_iso
        except (OSError, subprocess.SubprocessError):
            pass

    return view


def _tradable_symbols(symbols: list[str], benchmark: str) -> list[str]:
    """Retire le benchmark de l'univers tradable.

    Le benchmark sert de référence (régime de marché, force relative) : le
    laisser dans l'univers ferait acheter au bot sa propre référence, ce qui
    doublerait l'exposition au marché au lieu de la mesurer.

    Args:
        symbols: Univers demandé.
        benchmark: Symbole de référence configuré.

    Returns:
        L'univers, benchmark exclu.
    """
    return [symbol for symbol in symbols if symbol != benchmark]


def _run_check(
    settings: Settings,
    broker_factory: Callable[[Settings], AlpacaPaperClient],
    strategy_name: str,
    symbols: list[str],
) -> dict[str, Any]:
    """Diagnostic complet : compte, marché, signal du jour pour chaque symbole."""
    broker = broker_factory(settings)
    account = broker.get_account()
    market_open = broker.is_market_open()
    strategy = build_strategy(strategy_name, settings=settings)
    now = get_online_now()
    errors: list[str] = []
    if settings.benchmark_symbol in symbols:
        errors.append(
            f"{settings.benchmark_symbol} écarté de l'univers : il sert de référence"
        )
    symbols = _tradable_symbols(symbols, settings.benchmark_symbol)
    # Le Bouclier fait partie du portefeuille que le cycle va construire : le
    # diagnostic doit le montrer, sinon il annonce une allocation incomplète.
    shield_symbol = getattr(getattr(strategy, "params", None), "shield_symbol", None)
    if isinstance(shield_symbol, str) and shield_symbol not in symbols:
        symbols = [*symbols, shield_symbol]
    # Le nom du benchmark vient de la configuration ; la stratégie ne dit que
    # s'il lui en faut un. Se fier à sa valeur interne ferait diverger le
    # diagnostic d'un bot configuré sur une autre référence.
    benchmark_symbol = (
        settings.benchmark_symbol
        if getattr(strategy, "benchmark_symbol", None) is not None
        else None
    )
    if benchmark_symbol is not None:
        # Un benchmark indisponible ne doit pas priver l'utilisateur du
        # diagnostic des autres symboles (le bot, lui, bascule alors en
        # régime prudent).
        try:
            benchmark_raw = broker.fetch_bars(
                benchmark_symbol,
                timeframe=settings.timeframe,
                lookback_days=settings.lookback_days,
                now=now,
            )
            _inject_benchmark(
                strategy,
                drop_incomplete_bar(
                    benchmark_raw,
                    settings.timeframe,
                    now,
                    market_timezone=settings.market_timezone,
                ),
                benchmark_symbol,
            )
        except (DataError, BrokerError, StrategyError) as exc:
            errors.append(f"benchmark {benchmark_symbol}: {exc}")

    # Le filtre d'actualités est consulté ici comme il l'est dans le cycle :
    # le diagnostic doit annoncer le sort réel d'un signal d'achat, pas le
    # signal brut. Le cache d'articles rend l'opération gratuite tant qu'aucune
    # actualité nouvelle n'est parue.
    assessments = _assess_news(settings, symbols, errors=errors)
    engine = RiskEngine(settings.risk)
    panic = settings.risk.news_panic_exit_threshold

    bars_by_symbol: dict[str, Any] = {}
    for symbol in symbols:
        # Même tolérance que le cycle réel : un symbole mal orthographié est
        # écarté et signalé, il n'empêche pas d'analyser les autres.
        try:
            bars = broker.fetch_bars(
                symbol,
                timeframe=settings.timeframe,
                lookback_days=settings.lookback_days,
                now=now,
            )
            complete = drop_incomplete_bar(
                bars,
                settings.timeframe,
                now,
                market_timezone=settings.market_timezone,
            )
            bars_by_symbol[symbol] = complete
        except (DataError, BrokerError, StrategyError) as exc:
            errors.append(_symbol_message(symbol, exc))
            continue

    if hasattr(strategy, "set_universe"):
        strategy.set_universe(bars_by_symbol)

    rows: list[dict[str, Any]] = []
    for symbol, complete in bars_by_symbol.items():
        try:
            signal = strategy.latest_signal(complete, symbol=symbol)
        except (DataError, BrokerError, StrategyError) as exc:
            errors.append(_symbol_message(symbol, exc))
            continue
        atr = signal.indicators.get("atr", 0.0)
        news = assessments.get(symbol)
        row: dict[str, Any] = {
            "symbol": symbol,
            "action": signal.action.value,
            "reason": signal.reason,
            "price": signal.price,
            "atr": atr,
            "proposed_stop": round(signal.price - atr * settings.risk.atr_stop_multiple, 2)
            if atr > 0 and not engine.is_defensive_symbol(symbol)
            else None,
            "as_of": signal.as_of.isoformat(),
        }
        if news is not None:
            row["news"] = {
                "score": news.sentiment_score,
                "confidence": news.confidence,
                "critical": news.is_critical_threat,
                "articles": news.articles_count,
                "rationale": news.rationale,
                "effect": news_effect_label(news, engine=engine, panic=panic),
                "blocks_entry": engine.news_vetoes_entry(news),
            }
        rows.append(row)
    return {
        "account": {
            "equity": account.equity,
            "cash": account.cash,
            "buying_power": account.buying_power,
        },
        "market_open": market_open,
        "strategy": strategy_name,
        "signals": rows,
        "news": {
            "consulted": bool(assessments),
            "model": settings.openrouter_model,
            "veto_threshold": settings.risk.news_veto_threshold,
            "panic_threshold": settings.risk.news_panic_exit_threshold,
        },
        "errors": errors,
    }


def _run_backtest(
    settings: Settings,
    strategy_name: str,
    symbols: list[str],
    start: str,
    end: str | None,
) -> dict[str, Any]:
    """Backtest complet, courbe d'équité sous-échantillonnée pour le graphique."""
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()
    if end_date <= start_date:
        raise ValueError("La date de fin doit être après la date de début.")
    start_dt = datetime.combine(start_date, dt_time.min, tzinfo=UTC)
    end_dt = datetime.combine(end_date, dt_time.min, tzinfo=UTC) + timedelta(days=1)
    interval = timeframe_interval(settings.timeframe)

    warmup_days = 320 if interval in {"1d", "1Day"} else 30
    fetch_start_dt = start_dt - timedelta(days=warmup_days)

    provider: MarketDataProvider = CachedProvider(
        YFinanceProvider(), settings.state_dir / "cache"
    )
    strategy = build_strategy(strategy_name, settings=settings)
    symbols = _tradable_symbols(symbols, settings.benchmark_symbol)
    data = {
        symbol: provider.fetch(symbol, fetch_start_dt, end_dt, interval)
        for symbol in symbols
    }
    # Le nom du benchmark vient de la configuration, pas du code de la
    # stratégie : backtest et live doivent utiliser la même référence.
    benchmark_symbol = (
        settings.benchmark_symbol
        if getattr(strategy, "benchmark_symbol", None) is not None
        else None
    )
    benchmark_bars = None
    if benchmark_symbol is not None:
        benchmark_bars = provider.fetch(benchmark_symbol, fetch_start_dt, end_dt, interval)
        _inject_benchmark(strategy, benchmark_bars, benchmark_symbol)

    runner = BacktestRunner(strategy, risk_config=settings.risk)
    result = runner.run(data, benchmark=benchmark_bars, start_date=start_dt)

    curve = result.equity_curve
    step = max(1, len(curve) // _CHART_POINTS)
    points = [
        {
            "date": stamp.date().isoformat(),
            "equity": float(row["equity"]),
            "drawdown_pct": float(row["drawdown_pct"]),
        }
        for stamp, row in curve.iloc[::step].iterrows()
    ]
    if points and points[-1]["date"] != curve.index[-1].date().isoformat():
        last = curve.iloc[-1]
        points.append(
            {
                "date": curve.index[-1].date().isoformat(),
                "equity": float(last["equity"]),
                "drawdown_pct": float(last["drawdown_pct"]),
            }
        )
    return {
        "strategy": result.strategy_name,
        "symbols": list(result.symbols),
        "metrics": dict(result.metrics),
        "equity_points": points,
        "trades": [
            {
                "symbol": trade.symbol,
                "entry_date": trade.entry_time.date().isoformat(),
                "exit_date": trade.exit_time.date().isoformat(),
                "pnl": trade.pnl,
                "return_pct": trade.return_pct,
                "reason": trade.exit_reason,
            }
            for trade in result.trades[-100:]
        ],
        "halts": [
            {"date": halt.session_date.isoformat(), "reason": halt.reason}
            for halt in result.halts
        ],
    }


def _run_cycle(
    settings: Settings,
    broker_factory: Callable[[Settings], AlpacaPaperClient],
    strategy_name: str,
    *,
    dry_run: bool,
    watch_only: bool = False,
) -> dict[str, Any]:
    """Un cycle de trading paper (cycle complet ou vigie horaire)."""
    effective = settings.model_copy(update={"dry_run": True}) if dry_run else settings
    strategy = build_strategy(strategy_name, settings=effective)
    broker = broker_factory(effective)
    bot = TradingBot(
        effective,
        strategy,
        broker,
        notifier=_build_notifier(effective),
        news_analyzer=_build_news_analyzer(effective),
    )
    report = bot.run_watch_only(source="gui") if watch_only else bot.run_once(source="gui")
    payload = report.to_dict()
    payload["dry_run"] = dry_run
    return payload


_update_env = update_env_file


def _parse_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    """Lit un corps JSON borné ; lève ``ValueError`` sur entrée invalide."""
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0 or length > _MAX_BODY_BYTES:
        raise ValueError("requête sans corps ou trop volumineuse")
    payload = json.loads(handler.rfile.read(length).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("le corps doit être un objet JSON")
    return payload


def _describe_symbols(
    context: WebContext,
    symbols: list[str],
) -> tuple[list[dict[str, Any]], list[str], str | None]:
    """Résout chaque symbole auprès d'Alpaca avant de l'enregistrer.

    Le registre du courtier est la seule autorité qui compte : c'est lui qui
    exécutera les ordres. Il révèle aussi les pièges d'orthographe — « BTC »
    y désigne un ETF Grayscale, pas le Bitcoin.

    Args:
        context: Contexte du serveur (fournit le courtier).
        symbols: Symboles demandés, déjà normalisés en majuscules.

    Returns:
        ``(instruments reconnus, anomalies, avertissement)``. L'anomalie est
        un message prêt à afficher. L'avertissement n'est renseigné que si le
        contrôle n'a pas pu être mené : l'enregistrement reste alors possible.
    """
    try:
        broker = context.broker_factory(context.settings)
    except Exception as exc:  # noqa: BLE001 - la page doit rester utilisable
        return [], [], f"symboles enregistrés sans vérification ({_friendly(exc)})"

    describe = getattr(broker, "describe_asset", None)
    if describe is None:
        # Courtier simulé (tests) : pas de registre à interroger.
        return [], [], None

    resolved: list[dict[str, Any]] = []
    problems: list[str] = []
    for symbol in symbols:
        try:
            info = describe(symbol)
        except BrokerError as exc:
            return [], [], f"symboles enregistrés sans vérification ({exc})"
        if info is None:
            problems.append(f"{symbol} : inconnu d'Alpaca")
            continue
        if not info.tradable:
            problems.append(f"{symbol} ({info.name}) : actif non négociable")
            continue
        if info.is_crypto:
            # Le bot place une entrée « day » et un stop simple : deux ordres
            # que l'API crypto d'Alpaca refuse. Accepter la paire
            # promettrait un trading qui échouerait à chaque cycle.
            problems.append(
                f"{symbol} ({info.name}) : les paires crypto ne sont pas gérées "
                "par ce bot (Alpaca n'accepte ni ordre « day » ni stop simple "
                "sur le crypto)"
            )
            continue
        resolved.append(
            {
                "symbol": info.symbol,
                "name": info.name,
                "asset_class": info.asset_class,
            }
        )
    return resolved, problems, None


def _apply_config(
    context: WebContext,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Met à jour symboles et webhook dans le ``.env``, avec validation."""
    updates: dict[str, str] = {}
    wanted: list[str] = []
    symbols = request.get("symbols")
    if symbols is not None:
        if not isinstance(symbols, list) or not all(isinstance(s, str) for s in symbols):
            raise ValueError("la liste de symboles est invalide")
        wanted = [s.strip().upper() for s in symbols if s.strip()]
        updates["SYMBOLS"] = ",".join(wanted)
    if "webhook_url" in request:
        val = request["webhook_url"]
        updates["WEBHOOK_URL"] = str(val).strip() if val is not None else ""
    if request.get("webhook_format") is not None:
        updates["WEBHOOK_FORMAT"] = str(request["webhook_format"]).strip()
    if "webhook_chat_id" in request:
        val = request["webhook_chat_id"]
        updates["WEBHOOK_CHAT_ID"] = str(val).strip() if val is not None else ""
    if request.get("webhook_token"):
        # Champ vide = « ne touche pas au jeton existant ». Sans cette règle,
        # chaque enregistrement de réglages effacerait silencieusement
        # l'authentification d'un endpoint public.
        updates["WEBHOOK_TOKEN"] = str(request["webhook_token"]).strip()
    if (
        "withdrawable_cash_floor_usd" in request
        and str(request["withdrawable_cash_floor_usd"]).strip()
    ):
        try:
            floor_val = float(request["withdrawable_cash_floor_usd"])
            if floor_val < 0:
                raise ValueError
            updates["WITHDRAWABLE_CASH_FLOOR_USD"] = f"{floor_val:.2f}"
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "Le plancher de cash retirable doit être un montant positif ou nul."
            ) from exc
    if "equity_hard_floor_usd" in request and str(request["equity_hard_floor_usd"]).strip():
        try:
            eq_val = float(request["equity_hard_floor_usd"])
            if eq_val < 0:
                raise ValueError
            updates["EQUITY_HARD_FLOOR_USD"] = f"{eq_val:.2f}"
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "Le plancher de sécurité global doit être un montant positif ou nul."
            ) from exc
    if not updates:
        return {"ok": True, "message": "Rien à modifier."}

    # Contrôle auprès du courtier AVANT toute écriture : un symbole refusé ne
    # doit rien changer dans le .env.
    resolved: list[dict[str, Any]] = []
    warning: str | None = None
    if wanted:
        resolved, problems, warning = _describe_symbols(context, wanted)
        if problems:
            raise ValueError(
                "Symboles refusés : " + " ; ".join(problems) + ". Rien n'a été enregistré."
            )

    backup = (
        context.env_path.read_text(encoding="utf-8") if context.env_path.exists() else None
    )
    _update_env(context.env_path, updates)
    try:
        # Relit strictement le .env qui vient d'être écrit (pas un autre fichier
        # que Settings pourrait charger par défaut).
        fresh = Settings(_env_file=str(context.env_path))
    except Exception as exc:
        if backup is not None:
            context.env_path.write_text(backup, encoding="utf-8")
        raise ValueError(f"Configuration refusée ({exc}) : .env restauré.") from exc
    context.settings = fresh
    context.env_mtime = _env_mtime(context.env_path)
    webhook_enabled = bool(
        fresh.webhook_url and fresh.webhook_url.get_secret_value().strip()
    )
    return {
        "ok": True,
        "message": "Réglages enregistrés. Ils s'appliquent au prochain cycle.",
        "symbols": fresh.symbols,
        "webhook_enabled": webhook_enabled,
        "resolved": resolved,
        "warning": warning,
    }


def _optional_float(value: Any, message: str) -> float | None:
    """Convertit une valeur de formulaire en float, ou ``None`` si absente."""
    if value is None or not str(value).strip():
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc


def _run_withdraw(
    context: WebContext,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Exécute un retrait via le service partagé avec la commande ``withdraw``.

    Toute la mécanique (annulation vérifiée des stops, validation par le moteur
    de risque, audit, plancher de cash) vit dans ``trading_bot.withdrawal`` :
    l'interface ne fait que traduire le formulaire et le résultat.

    Le verrou est pris sans attendre : si un cycle tourne déjà, l'utilisateur
    reçoit un refus immédiat plutôt qu'une requête HTTP figée.
    """
    if not context.cycle_lock.acquire(blocking=False):
        raise ValueError("Un cycle ou un retrait est déjà en cours d'exécution.")
    try:
        settings = _refresh_settings(context)
        broker = context.broker_factory(settings)
        service = WithdrawalService(
            settings,
            broker,
            state_store=StateStore(settings.state_dir / "state.json"),
            audit=AuditLog(settings.state_dir / "audit.jsonl"),
            env_path=context.env_path,
        )

        free_cash = _optional_float(
            request.get("free_cash"), "Le montant de cash cible est invalide."
        )
        if free_cash is not None:
            result = service.withdraw_free_cash(free_cash)
            context.settings = Settings(_env_file=str(context.env_path))
            context.env_mtime = _env_mtime(context.env_path)
            payload = _withdrawal_payload("free_cash", result)
            payload["orders"] = payload.pop("fills")
            payload["withdrawable_floor"] = free_cash
            payload["message"] = (
                f"Cash cible de {free_cash:.2f} $ atteint. "
                f"Plancher sanctuarisé dans .env ({len(result.fills)} ordre(s) exécuté(s))."
            )
            if not payload["ok"]:
                payload["error"] = " ; ".join(result.errors)
            return payload

        symbol = str(request.get("symbol") or "").strip().upper()
        if not symbol:
            raise ValueError(
                "Veuillez sélectionner un symbole ou renseigner un montant de cash global."
            )
        result = service.withdraw_symbol(
            symbol,
            amount=_optional_float(request.get("amount"), "Montant en dollars invalide."),
            pct=_optional_float(request.get("pct"), "Pourcentage invalide."),
        )
        payload = _withdrawal_payload("symbol", result)
        payload["orders"] = payload.pop("fills")
        if not payload["ok"]:
            payload["error"] = " ; ".join(result.errors) or "Le retrait n'a pas abouti."
            return payload
        fill = result.fills[0]
        payload["message"] = (
            f"Retrait de {fill.qty:.4f} {fill.symbol} exécuté "
            f"(~{fill.cash_freed:.2f} $ dégagés)."
        )
        return payload
    finally:
        context.cycle_lock.release()


def _reset_permanent_halt(
    context: WebContext,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Lève l'arrêt permanent après confirmation explicite."""
    if request.get("confirm") is not True:
        raise ValueError("confirmation manquante")
    reason = str(request.get("reason") or "").strip()
    if not reason:
        raise ValueError(
            "Expliquez en une phrase ce que vous avez vérifié avant de relancer."
        )
    store = StateStore(context.settings.state_dir / "state.json")
    state = store.load()
    if not state.halted or state.halt_scope != "permanent":
        raise ValueError(
            "Aucun arrêt permanent actif : rien à lever "
            "(un arrêt journalier se relance tout seul au prochain jour de marché)."
        )
    state.halted = False
    state.halt_scope = None
    state.halt_reason = None
    store.save(state)
    return {
        "ok": True,
        "message": "Arrêt permanent levé. Le robot reprendra au prochain cycle.",
    }


def _rebase(context: WebContext, request: dict[str, Any]) -> dict[str, Any]:
    """Ré-ancre la référence d'équité après un dépôt ou un retrait chez Alpaca.

    Même opération que ``./tradingbot rebase`` : sans elle, un retrait de 20 %
    est lu comme une perte et le robot s'arrête définitivement.
    """
    reason = str(request.get("reason") or "").strip()
    if not reason:
        raise ValueError(
            "Expliquez en une phrase le mouvement de cash "
            "(par exemple « retrait de 5 000 $ vers mon compte »)."
        )
    settings = _refresh_settings(context)
    broker = context.broker_factory(settings)
    payload = apply_equity_rebase(
        settings,
        broker.get_account(),
        reason=reason,
        state_store=StateStore(settings.state_dir / "state.json"),
        audit=AuditLog(settings.state_dir / "audit.jsonl"),
    )
    previous = payload["previous_peak"] or 0.0
    return {
        "ok": True,
        "message": (
            f"Référence recalée sur {payload['new_peak']:.2f} $ "
            f"(pic précédent : {previous:.2f} $)."
        ),
        **payload,
    }


def _install_automation(
    context: WebContext,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Installe (ou met à jour) le timer quotidien avec la stratégie choisie."""
    strategy = str(request.get("strategy") or "ema_cross")
    if not context.install_script.exists():
        raise ValueError(f"Script d'installation introuvable : {context.install_script}")
    if shutil.which("systemctl") is None or shutil.which("bash") is None:
        raise ValueError(
            "Le pilotage automatique par service systemd nécessite un environnement "
            "Linux (avec bash et systemctl). Sur Windows, utilisez les boutons de "
            "lancement manuel ci-dessous ou le Planificateur de tâches."
        )
    command = ["bash", str(context.install_script)]
    if strategy != "ema_cross":
        command += ["--strategy", strategy]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"L'installation a échoué : {exc}") from exc
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        raise ValueError(
            f"L'installation a échoué (code {completed.returncode}) : {output}"
        )
    _invalidate_automation_cache()
    return {
        "ok": True,
        "message": f"Pilotage automatique activé avec la stratégie {strategy}.",
        "output": output,
        "automation": _automation_view(),
    }


def _env_mtime(path: Path) -> int | None:
    """Empreinte temporelle du fichier, ou ``None`` s'il est illisible.

    Résolution nanoseconde (``st_mtime_ns``) : deux écritures rapprochées
    doivent être distinguées, y compris sur un système de fichiers à
    granularité grossière.
    """
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _refresh_settings(context: WebContext) -> Settings:
    """Recharge le ``.env`` s'il a changé depuis le dernier chargement.

    Sans cette relecture, une modification du fichier faite hors de
    l'interface (éditeur de texte, script) laisserait la page afficher
    l'ancienne liste d'actions alors que le bot, lui, lit déjà la nouvelle :
    l'utilisateur croirait piloter un univers que le robot n'utilise plus.

    Args:
        context: Contexte du serveur, mis à jour sur place.

    Returns:
        La configuration effective (rechargée si nécessaire).
    """
    with context.config_lock:
        mtime = _env_mtime(context.env_path)
        if mtime is None:
            # Aucun fichier à relire : on garde la configuration en mémoire
            # (c'est aussi celle injectée par les tests).
            return context.settings
        if context.env_mtime != mtime:
            try:
                # Relit strictement le .env du projet, pas un autre fichier.
                fresh = Settings(_env_file=str(context.env_path))
            except Exception:  # noqa: BLE001 - fichier tronqué en cours d'écriture
                return context.settings
            context.settings = fresh
            context.env_mtime = mtime
        return context.settings


def _status_payload(context: WebContext) -> dict[str, Any]:
    """Toutes les informations du tableau de bord, en un appel."""
    settings = _refresh_settings(context)
    webhook_val = (
        settings.webhook_url.get_secret_value().strip()
        if settings.webhook_url is not None
        else ""
    )
    token_set = bool(
        settings.webhook_token and settings.webhook_token.get_secret_value().strip()
    )
    payload: dict[str, Any] = {
        "positions": [],
        "market_open": None,
        "online_time": get_online_now().isoformat(),
        "clock_offset_seconds": round(clock_offset_seconds(), 3),
        "automation": _automation_view(),
        "settings": {
            "symbols": settings.symbols,
            "benchmark": settings.benchmark_symbol,
            "dry_run": settings.dry_run,
            "webhook_enabled": bool(webhook_val),
            "webhook_token_set": token_set,
            "webhook_format": settings.webhook_format,
            "webhook_chat_id": settings.webhook_chat_id or "",
            "daily_loss_limit_pct": settings.risk.max_daily_loss_pct,
            "drawdown_limit_pct": settings.risk.max_drawdown_pct,
            "withdrawable_cash_floor_usd": settings.risk.withdrawable_cash_floor_usd,
            "equity_hard_floor_usd": settings.risk.equity_hard_floor_usd,
            "partial_take_profit_pct": settings.risk.partial_take_profit_pct,
            "partial_take_profit_sell_ratio": settings.risk.partial_take_profit_sell_ratio,
            "max_single_position_weight": settings.risk.max_single_position_weight,
            "safe_haven_symbols": settings.safe_haven_symbols,
            "shield_symbol": settings.risk.shield_symbol,
            "shield_allocation_pct": settings.shield_allocation_pct,
        },
    }
    try:
        broker = context.broker_factory(settings)
        account = broker.get_account()
        payload["account"] = {
            "equity": account.equity,
            "cash": account.cash,
            "buying_power": account.buying_power,
        }
        payload["market_open"] = broker.is_market_open()
        payload["positions"] = _positions_view(broker, settings)
    except Exception as exc:  # noqa: BLE001 - le tableau de bord reste affichable
        payload["error"] = _friendly(exc)

    state = StateStore(settings.state_dir / "state.json").load()
    payload["state"] = {
        "trading_day": state.trading_day,
        "halted": state.halted,
        "halt_scope": state.halt_scope,
        "halt_reason": state.halt_reason,
        "day_start_equity": state.day_start_equity,
        "peak_equity": state.peak_equity,
    }
    # Dernier cycle connu : permet à la synthèse du tableau de bord de dater
    # l'activité du robot sans imposer un second appel à /api/history.
    latest = _history_entries(context, limit=1)
    payload["last_cycle"] = latest[0] if latest else None
    return payload


def _history_entries(context: WebContext, limit: int = 60) -> list[dict[str, Any]]:
    """Derniers cycles consignés, du plus récent au plus ancien.

    L'historique est écrit par le bot lui-même (service systemd comme cycles
    lancés depuis la page) : un enregistrement par séance de marché.
    """
    store = HistoryStore(context.settings.state_dir / "history.jsonl")
    try:
        return store.entries(limit=limit)
    except OSError:
        return []


def _env_file_path(settings: Settings) -> Path:
    """Chemin du ``.env`` que Settings charge réellement."""
    candidate = settings.model_config.get("env_file")
    if isinstance(candidate, Path):
        return candidate
    return Path(".env")


def build_handler(context: WebContext) -> type[BaseHTTPRequestHandler]:
    """Construit la classe de gestion des requêtes HTTP."""

    class GuiHandler(BaseHTTPRequestHandler):
        """Routes de l'API + page unique, avec garde-fous croix-site."""

        server_version = "TradingBotGUI/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            """Silence : le serveur n'émet pas de log par requête."""

        def _send(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _guarded(self) -> bool:
            """Refuse les requêtes croix-site et les hôtes étrangers.

            Un ``Origin`` local est légitime : le navigateur l'envoie même sur
            ses propres requêtes ``fetch`` en POST (donc pour chaque bouton de
            la page). Seule une origine distante révèle un appel croix-site.
            """
            origin = self.headers.get("Origin")
            if origin and not is_local_origin(origin):
                self._send(403, {"ok": False, "error": "Requête croix-site refusée."})
                return False
            host = _extract_host(self.headers.get("Host"))
            if host not in _LOCAL_HOSTS:
                self._send(403, {"ok": False, "error": "Hôte non autorisé."})
                return False
            return True

        def do_GET(self) -> None:
            if not self._guarded():
                return
            path = urlparse(self.path).path
            if path == "/":
                page = (Path(__file__).parent / "static" / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            static_assets: dict[str, tuple[str, str]] = {
                # Interface (cache court : une mise à jour doit être visible).
                "/app.css": ("text/css; charset=utf-8", "app.css"),
                "/app.js": ("text/javascript; charset=utf-8", "app.js"),
                "/favicon.svg": ("image/svg+xml", "favicon.svg"),
                "/favicon.ico": ("image/x-icon", "favicon.ico"),
                "/apple-touch-icon.png": ("image/png", "apple-touch-icon.png"),
                "/icon-192.png": ("image/png", "icon-192.png"),
                "/icon-512.png": ("image/png", "icon-512.png"),
                "/icon-maskable-512.png": ("image/png", "icon-maskable-512.png"),
                "/manifest.webmanifest": (
                    "application/manifest+json; charset=utf-8",
                    "manifest.webmanifest",
                ),
            }
            if path in static_assets:
                mime_type, filename = static_assets[path]
                asset_file = Path(__file__).parent / "static" / filename
                if asset_file.is_file():
                    data = asset_file.read_bytes()
                    cache_ctl = (
                        "no-cache"
                        if path in {"/app.css", "/app.js"}
                        else "public, max-age=86400"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", mime_type)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", cache_ctl)
                    self.end_headers()
                    self.wfile.write(data)
                    return
            match = re.fullmatch(r"/api/job/([0-9a-f]+)", path)
            if match:
                job = context.jobs.get(match.group(1))
                if job is None:
                    self._send(404, {"ok": False, "error": "Tâche inconnue ou expirée."})
                    return
                snapshot = job.snapshot()
                snapshot["ok"] = job.status != "failed"
                self._send(200, snapshot)
                return
            if path == "/api/status":
                self._send(200, {"ok": True, **_status_payload(context)})
                return
            if path == "/api/history":
                self._send(200, {"ok": True, "entries": _history_entries(context)})
                return
            self._send(404, {"ok": False, "error": "Page inconnue."})

        def do_POST(self) -> None:
            if not self._guarded():
                return
            path = urlparse(self.path).path
            try:
                request = _parse_body(self)
            except (ValueError, json.JSONDecodeError) as exc:
                self._send(400, {"ok": False, "error": str(exc)})
                return
            try:
                payload = self._dispatch_post(path, request)
                self._send(200, payload)
            except Exception as exc:  # noqa: BLE001 - l'interface affiche le message
                self._send(200, {"ok": False, "error": _friendly(exc)})

        def _dispatch_post(self, path: str, request: dict[str, Any]) -> dict[str, Any]:
            """Aiguille les actions ; toute erreur remonte en français."""
            settings = _refresh_settings(context)
            if path == "/api/check":
                strategy = str(request.get("strategy") or "ema_cross")
                symbols = request.get("symbols") or settings.symbols
                job = context.jobs.submit(
                    "check",
                    "Diagnostic du jour",
                    lambda: _run_check(
                        settings, context.broker_factory, strategy, list(symbols)
                    ),
                )
                return {"ok": True, "job": job.snapshot()}
            if path == "/api/backtest":
                strategy = str(request.get("strategy") or "ema_cross")
                symbols = [
                    symbol.strip().upper()
                    for symbol in (request.get("symbols") or settings.symbols)
                    if symbol
                ]
                start = str(request.get("start") or "2022-01-01")
                job = context.jobs.submit(
                    "backtest",
                    f"Backtest {strategy}",
                    lambda: _run_backtest(
                        settings, strategy, symbols, start, request.get("end")
                    ),
                    lock=context.backtest_lock,
                )
                return {"ok": True, "job": job.snapshot()}
            if path == "/api/cycle":
                if (
                    "cycle" in context.jobs.running_kinds
                    or "withdraw" in context.jobs.running_kinds
                ):
                    return {"ok": False, "error": "Un cycle est déjà en cours d'exécution."}
                strategy = str(request.get("strategy") or "ema_cross")
                dry_run = bool(request.get("dry_run", False))
                watch_only = bool(request.get("watch_only", False))
                label = "Vigie horaire paper" if watch_only else "Cycle complet paper"
                job = context.jobs.submit(
                    "cycle",
                    label,
                    lambda: _run_cycle(
                        settings,
                        context.broker_factory,
                        strategy,
                        dry_run=dry_run,
                        watch_only=watch_only,
                    ),
                    lock=context.cycle_lock,
                )
                return {"ok": True, "job": job.snapshot()}
            if path == "/api/automation":
                return _install_automation(context, request)
            if path == "/api/halt/reset":
                return _reset_permanent_halt(context, request)
            if path == "/api/rebase":
                return _rebase(context, request)
            if path == "/api/config":
                return _apply_config(context, request)
            if path == "/api/withdraw":
                if (
                    "cycle" in context.jobs.running_kinds
                    or "withdraw" in context.jobs.running_kinds
                ):
                    return {
                        "ok": False,
                        "error": "Un cycle ou un retrait est déjà en cours d'exécution.",
                    }
                if request.get("async") is True:
                    job = context.jobs.submit(
                        "withdraw",
                        "Retrait et sanctuarisation",
                        lambda: _run_withdraw(context, request),
                    )
                    return {"ok": True, "job": job.snapshot()}
                return _run_withdraw(context, request)
            raise ValueError("Action inconnue.")

    return GuiHandler


def serve(
    settings: Settings,
    *,
    host: str = "127.0.0.1",
    port: int = 8484,
    broker_factory: Callable[[Settings], AlpacaPaperClient] | None = None,
) -> None:
    """Démarre l'interface web (bloquant jusqu'à Ctrl+C).

    Raises:
        ValueError: si l'hôte demandé n'est pas local.
    """
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("l'interface web n'écoute que sur localhost (sécurité)")
    context = WebContext(
        settings=settings,
        broker_factory=broker_factory or _default_broker_factory,
        env_path=_env_file_path(settings),
        install_script=Path(__file__).resolve().parent.parent.parent
        / "deploy"
        / "install.sh",
        jobs=JobManager(),
        backtest_lock=threading.Lock(),
        cycle_lock=threading.Lock(),
    )
    httpd = ThreadingHTTPServer((host, port), build_handler(context))
    configure_logging(settings)
    print(f"Interface du TradingBot : http://{host}:{port}  (Ctrl+C pour arrêter)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Interface arrêtée.")
    finally:
        httpd.server_close()


__all__ = ["WebContext", "build_handler", "serve"]
