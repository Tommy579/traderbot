"""Point d'entrée : ligne de commande du paper trading.

Trois familles de commandes :

* ``live`` — boucle paper trading (ou ``--once``, ``--watch-only``) ;
* ``backtest`` — backtest déterministe yfinance ;
* ``check``, ``news``, ``web``, ``withdraw``, ``rebase``, ``publish-stats`` —
  diagnostic, radar d'actualités, interface locale, retraits et publication.

L'orchestrateur vit dans :mod:`trading_bot.bot` et la persistance dans
:mod:`trading_bot.state` ; ce module ne porte que les arguments, le formatage
et les codes de sortie. Les noms publics restent ré-exportés ici pour que
``python -m trading_bot.main`` et les appelants existants ne changent pas.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import Any

from trading_bot.backtest.runner import BacktestResult, BacktestRunner
from trading_bot.bot import (
    STRATEGY_NAMES,
    TradingBot,
    _assess_news,
    _build_news_analyzer,
    _build_notifier,
    _build_stats_publisher,
    _inject_benchmark,
    build_strategy,
    market_closed_message,
)
from trading_bot.clock import get_online_now
from trading_bot.config import (
    PROJECT_ROOT,
    BacktestConfig,
    ConfigurationError,
    Settings,
    configure_logging,
)
from trading_bot.data.loader import (
    BarFrame,
    CachedProvider,
    DataError,
    MarketDataProvider,
    YFinanceProvider,
    drop_incomplete_bar,
)
from trading_bot.data.news_sentiment import NewsAssessment
from trading_bot.execution.alpaca_client import (
    AccountSnapshot,
    AlpacaPaperClient,
    BrokerError,
)
from trading_bot.history import HistoryStore
from trading_bot.risk.engine import (
    AccountState,
    HaltScope,
    RiskEngine,
    RiskError,
)
from trading_bot.state import AuditLog, PersistedState, StateStore
from trading_bot.state import utc_now as _utc_now
from trading_bot.strategies.base import SignalType, StrategyError
from trading_bot.withdrawal import WithdrawalResult, WithdrawalService

__all__ = [
    "STRATEGY_NAMES",
    "AuditLog",
    "PersistedState",
    "StateStore",
    "TradingBot",
    "apply_equity_rebase",
    "build_parser",
    "build_strategy",
    "cli",
    "cmd_backtest",
    "cmd_check",
    "cmd_live",
    "cmd_news",
    "cmd_publish_stats",
    "cmd_rebase",
    "cmd_web",
    "cmd_withdraw",
    "main",
    "market_closed_message",
    "news_effect_label",
    "print_backtest_summary",
]


def _format_pct(value: float | int | None) -> str:
    """Formate un ratio en pourcentage lisible."""
    if value is None:
        return "n/a"
    return f"{float(value):.2%}"


def _format_number(value: float | int | None, digits: int = 2) -> str:
    """Formate un nombre avec un nombre fixe de décimales."""
    if value is None:
        return "n/a"
    return f"{float(value):,.{digits}f}"


def print_backtest_summary(result: BacktestResult) -> None:
    """Affiche les métriques principales d'un backtest."""
    metrics = result.metrics
    print("")
    print(f"Backtest {result.strategy_name} | {', '.join(result.symbols)}")
    print(f"  équity initiale      : {_format_number(metrics['initial_equity'])}")
    print(f"  équity finale        : {_format_number(metrics['final_equity'])}")
    print(f"  rendement total      : {_format_pct(metrics['total_return'])}")
    print(f"  CAGR                 : {_format_pct(metrics['cagr'])}")
    print(f"  Sharpe               : {_format_number(metrics['sharpe'])}")
    print(f"  Sortino              : {_format_number(metrics['sortino'])}")
    print(f"  max drawdown         : {_format_pct(metrics['max_drawdown'])}")
    print(f"  Calmar               : {_format_number(metrics['calmar'])}")
    print(f"  trades               : {metrics['trades']}")
    print(f"  taux de réussite     : {_format_pct(metrics['win_rate'])}")
    print(f"  profit factor        : {_format_number(metrics['profit_factor'])}")
    print(f"  espérance / trade    : {_format_number(metrics['expectancy'])}")
    print(f"  frais totaux         : {_format_number(metrics['total_fees'])}")
    print(f"  exposition moyenne   : {_format_pct(metrics['exposure_pct'])}")
    print(f"  circuit breakers     : {metrics['halt_events']}")
    print(f"  entrées refusées     : {metrics['entry_rejections']}")
    print(f"  positions ouvertes   : {metrics['open_positions_at_end']}")
    print(f"  buy & hold indice    : {_format_pct(metrics['benchmark_buy_hold'])}")
    print(f"  buy & hold panier    : {_format_pct(metrics['basket_buy_hold'])}")
    if result.halts:
        print("  halts:")
        for halt in result.halts:
            print(
                f"    - {halt.session_date.isoformat()} [{halt.scope.value}] {halt.reason}"
            )
    print("")


def _settings_from_args(args: argparse.Namespace) -> Settings:
    """Construit les Settings en appliquant les surcharges CLI."""
    overrides: dict[str, Any] = {}
    if getattr(args, "symbols", None):
        overrides["symbols"] = args.symbols
    if getattr(args, "dry_run", False):
        overrides["dry_run"] = True
    if getattr(args, "log_level", None):
        overrides["log_level"] = args.log_level
    return Settings(**overrides)


def cmd_live(args: argparse.Namespace) -> int:
    """Commande ``live`` : boucle paper trading."""
    settings = _settings_from_args(args)
    logger = configure_logging(settings)
    try:
        strategy = build_strategy(args.strategy, settings=settings)
    except (ConfigurationError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    try:
        broker = AlpacaPaperClient(settings)
    except (ConfigurationError, BrokerError) as exc:
        logger.error("%s", exc)
        return 2
    bot = TradingBot(
        settings,
        strategy,
        broker,
        notifier=_build_notifier(settings),
        news_analyzer=_build_news_analyzer(settings),
    )
    if getattr(args, "watch_only", False):
        report = bot.run_watch_only()
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return 0 if not report.errors else 1
    if args.once:
        report = bot.run_once()
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return 0 if not report.errors else 1
    try:
        bot.run_forever()
    except KeyboardInterrupt:
        logger.info("interruption clavier")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Commande ``check`` : diagnostic lecture seule, sans ordre."""
    settings = _settings_from_args(args)
    logger = configure_logging(settings)
    try:
        strategy = build_strategy(args.strategy, settings=settings)
    except (ConfigurationError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    try:
        broker = AlpacaPaperClient(settings)
        account = broker.get_account()
        positions = broker.get_positions()
        market_open = broker.is_market_open()
    except (ConfigurationError, BrokerError) as exc:
        logger.error("%s", exc)
        return 2

    symbol = settings.symbols[0]
    benchmark_symbol = settings.benchmark_symbol
    print("")
    print("Diagnostic paper trading Alpaca")
    print(f"  endpoint        : {settings.alpaca_trading_url}")
    print(f"  paper           : {settings.is_paper}")
    print(f"  marché ouvert   : {market_open}")
    print(f"  équité          : {_format_number(account.equity)}")
    print(f"  cash            : {_format_number(account.cash)}")
    print(f"  buying power    : {_format_number(account.buying_power)}")
    print(f"  univers         : {', '.join(settings.symbols)}")
    print(f"  stratégie       : {strategy.name}")
    print(f"  benchmark       : {benchmark_symbol}")
    print(f"  timeframe       : {settings.timeframe}")

    benchmark_complete: BarFrame | None = None
    try:
        benchmark_raw = broker.fetch_bars(
            benchmark_symbol,
            timeframe=settings.timeframe,
            lookback_days=settings.lookback_days,
            now=_utc_now(),
        )
        benchmark_complete = drop_incomplete_bar(
            benchmark_raw,
            settings.timeframe,
            _utc_now(),
            market_timezone=settings.market_timezone,
        )
        if getattr(strategy, "benchmark_symbol", None) is not None:
            _inject_benchmark(strategy, benchmark_complete, benchmark_symbol)
    except (BrokerError, DataError, StrategyError) as exc:
        logger.warning("benchmark %s indisponible: %s", benchmark_symbol, exc)

    sma_period = settings.risk.regime_sma_period
    if benchmark_complete is not None and len(benchmark_complete) >= sma_period:
        b_close = benchmark_complete["close"].astype(float)
        b_sma = float(b_close.rolling(sma_period).mean().iloc[-1])
        b_last = float(b_close.iloc[-1])
        regime_label = "BULL" if b_last > b_sma else "BEAR"
        extra = "" if regime_label == "BULL" else " [BEAR_REGIME_VETO actif sur offensifs]"
        print(
            f"  régime {benchmark_symbol} (SMA {sma_period}) : {regime_label} "
            f"(cours {_format_number(b_last)} vs SMA {_format_number(b_sma)}){extra}"
        )
    elif benchmark_complete is not None:
        print(
            f"  régime {benchmark_symbol} (SMA {sma_period}) : INCONNU "
            f"({len(benchmark_complete)}/{sma_period} barres requises)"
        )
    else:
        print(f"  régime {benchmark_symbol} (SMA {sma_period}) : INDISPONIBLE")

    if settings.safe_haven_enabled:
        safe_symbols_str = ", ".join(settings.safe_haven_symbols)
        cash_buf = account.equity * settings.risk.cash_buffer_pct
        max_sh_notional = account.equity * settings.risk.max_safe_haven_total_pct
        sh_held = {
            sym: pos
            for sym, pos in positions.items()
            if sym in settings.safe_haven_symbols and pos.qty > 0
        }
        sh_total_val = sum(p.market_value for p in sh_held.values())
        sh_pct = (sh_total_val / account.equity) if account.equity > 0 else 0.0
        print(
            f"  poche de repli  : activée (symboles: {safe_symbols_str} | "
            f"buffer cash min: {_format_pct(settings.risk.cash_buffer_pct)} [{_format_number(cash_buf)}] | "
            f"plafond: {_format_pct(settings.risk.max_safe_haven_total_pct)} [{_format_number(max_sh_notional)}])"
        )
        if sh_held:
            sh_details = ", ".join(
                f"{sym}: {p.qty:.2f} parts ({_format_number(p.market_value)})"
                for sym, p in sh_held.items()
            )
            print(
                f"  repli détenu    : {_format_number(sh_total_val)} "
                f"({_format_pct(sh_pct)}) [{sh_details}]"
            )
        else:
            print(f"  repli détenu    : {_format_number(0.0)} (0.00%) [aucune position]")
    else:
        print("  poche de repli  : désactivée")

    if strategy.name == "shield_sword":
        shield_weight = getattr(
            getattr(strategy, "params", None), "shield_allocation_pct", 0.0
        )
        print(
            f"  bouclier        : {settings.risk.shield_symbol} "
            f"({_format_pct(shield_weight)} de l'équité, sans stop)"
        )

    try:
        bars = broker.fetch_bars(
            symbol,
            timeframe=settings.timeframe,
            lookback_days=settings.lookback_days,
            now=_utc_now(),
        )
        complete = drop_incomplete_bar(
            bars,
            settings.timeframe,
            _utc_now(),
            market_timezone=settings.market_timezone,
        )
        if hasattr(strategy, "set_universe"):
            strategy.set_universe({symbol: complete})
        signal = strategy.latest_signal(complete, symbol=symbol)
    except (BrokerError, DataError, StrategyError) as exc:
        logger.error("données indisponibles: %s", exc)
        return 1

    atr = signal.indicators.get("atr", 0.0)
    print(f"  barres          : {len(complete)} (dernière {complete.index[-1]})")
    print(f"  prix            : {_format_number(signal.price)}")
    print(f"  signal          : {signal.action.value} — {signal.reason}")
    print(f"  ATR             : {_format_number(atr)}")
    if atr > 0:
        stop = signal.price - atr * settings.risk.atr_stop_multiple
        print(f"  stop ATR proposé: {_format_number(stop)}")

    # Le diagnostic doit annoncer la même chose que le cycle : un signal BUY
    # peut être refusé la nuit venue pour cause d'actualité, et c'est ici qu'on
    # le voit. Sans ce bloc, le diagnostic promettrait un achat qui n'aura pas
    # lieu — précisément le genre d'écart qui rend un outil trompeur.
    engine = RiskEngine(settings.risk)
    panic = settings.risk.news_panic_exit_threshold
    assessment = _assess_news(settings, [symbol]).get(symbol)
    print("")
    if assessment is None:
        print("  actualités      : filtre inactif (aucune analyse)")
    else:
        print(
            f"  actualités      : score {assessment.sentiment_score:+.2f} "
            f"(confiance {assessment.confidence:.2f}, "
            f"{assessment.articles_count} article(s))"
        )
        print(
            f"  effet           : {news_effect_label(assessment, engine=engine, panic=panic)}"
        )
        if assessment.is_critical_threat:
            print("  menace critique : oui")
        print(f"  motif           : {assessment.rationale}")
        if signal.action is SignalType.BUY and engine.news_vetoes_entry(assessment):
            print(
                "  ATTENTION       : ce signal d'achat serait REFUSÉ ([NEWS_VETO]) "
                "lors du prochain cycle."
            )
    print("")
    return 0


def news_effect_label(
    assessment: NewsAssessment, *, engine: RiskEngine, panic: float
) -> str:
    """Qualifie l'effet d'une note d'actualités sur la décision d'achat.

    Source unique de vérité pour l'affichage : le radar, le diagnostic en
    ligne de commande et le diagnostic web doivent annoncer la même chose que
    ce que le moteur de risque appliquera réellement.

    Args:
        assessment: Note d'actualités du symbole.
        engine: Moteur portant les seuils configurés.
        panic: Seuil de sortie d'urgence.

    Returns:
        ``"SORTIE URGENCE"``, ``"VETO ENTRÉE"`` ou ``"—"``.
    """
    if assessment.is_critical_threat and assessment.sentiment_score <= panic:
        return "SORTIE URGENCE"
    if engine.news_vetoes_entry(assessment):
        return "VETO ENTRÉE"
    return "—"


def _format_news_row(
    symbol: str,
    assessment: NewsAssessment,
    *,
    panic: float,
    engine: RiskEngine,
) -> str:
    """Met en forme une ligne du radar d'actualités."""
    effect = news_effect_label(assessment, engine=engine, panic=panic)
    rationale = " ".join(assessment.rationale.split())
    return (
        f"{symbol:<9}{assessment.sentiment_score:>+8.2f}"
        f"{assessment.confidence:>8.2f}"
        f"{'oui' if assessment.is_critical_threat else 'non':>9}"
        f"{assessment.articles_count:>5}  {effect:<15}{rationale[:44]}"
    )


def cmd_news(args: argparse.Namespace) -> int:
    """Commande ``news`` : radar de sentiment sur les actualités récentes.

    Affiche, pour chaque symbole, la note du modèle et l'effet qu'elle
    produirait sur le moteur de risque. Utile pour comprendre *pourquoi* un
    achat a été refusé, sans fouiller ``audit.jsonl``.
    """
    settings = _settings_from_args(args)
    configure_logging(settings)
    analyzer = _build_news_analyzer(settings)
    if analyzer is None:
        if not settings.news_filter_enabled:
            print("Filtre d'actualités désactivé (NEWS_FILTER_ENABLED=false).")
        else:
            print("Identifiants Alpaca absents : actualités illisibles.")
        return 2

    symbols = list(settings.symbols)
    veto = settings.risk.news_veto_threshold
    panic = settings.risk.news_panic_exit_threshold
    engine = RiskEngine(settings.risk)

    # Tout est écrit avec flush : un appel au modèle prend plusieurs secondes
    # et l'absence de retour immédiat donne l'impression que rien ne se passe.
    print(
        f"Radar d'actualités | fenêtre {settings.news_lookback_hours} h | "
        f"modèle {settings.openrouter_model}",
        flush=True,
    )
    print(
        f"seuils : veto d'entrée ≤ {veto:+.2f} | sortie d'urgence ≤ {panic:+.2f}",
        flush=True,
    )
    print(
        f"{len(symbols)} symbole(s). Un appel au modèle par symbole ayant des "
        "articles inédits ; les autres sont servis par le cache.",
        flush=True,
    )

    header = (
        f"{'Symbole':<9}{'Score':>8}{'Conf.':>8}{'Critique':>9}"
        f"{'Art.':>5}  {'Effet':<15}Motif"
    )
    print(header, flush=True)
    print("-" * 78, flush=True)

    def emit(symbol: str, assessment: NewsAssessment) -> None:
        """Affiche une ligne sitôt calculée, sans attendre les suivantes."""
        print(
            _format_news_row(symbol, assessment, panic=panic, engine=engine),
            flush=True,
        )

    analyzer.assess(symbols, now=get_online_now(), on_progress=emit)

    if not analyzer.llm_enabled:
        print("\n(Attention : aucune clé OpenRouter, les scores sont neutres.)")
    return 0


def cmd_publish_stats(args: argparse.Namespace) -> int:
    """Commande ``publish-stats`` : republie l'historique local vers GitHub.

    Sert à rattraper les cycles antérieurs à la mise en place de la
    publication, ou ceux dont l'envoi a échoué (PC éteint, réseau coupé, jeton
    posé après coup). La publication normale, elle, a lieu à la fin de chaque
    cycle sans rien demander.
    """
    settings = _settings_from_args(args)
    logger = configure_logging(settings)
    publisher = _build_stats_publisher(settings)
    if publisher is None:
        logger.error(
            "publication non configurée : renseignez STATS_GITHUB_REPO et "
            "STATS_GITHUB_TOKEN (voir .env.example)"
        )
        return 2

    store = HistoryStore(settings.state_dir / "history.jsonl")
    rows = store.entries(limit=-1)
    if not rows:
        logger.error("aucun cycle enregistré dans %s", store.path)
        return 1

    published = 0
    # Du plus ancien au plus récent : le fichier distant se construit dans
    # l'ordre, et la règle « une entrée par séance » s'applique correctement.
    for row in reversed(rows):
        report = {
            key: value
            for key, value in row.items()
            if key not in {"recorded_at", "source", "recovered_from"}
        }
        session = str(report.get("session_date", "?"))
        if publisher.send("cycle_trading", report):
            published += 1
            print(f"  publié : {session}")
        else:
            logger.warning("cycle %s non publié", session)

    total = len(rows)
    print(f"\n{published}/{total} cycle(s) publié(s) dans {settings.stats_github_repo}")
    return 0 if published == total else 1


def cmd_backtest(args: argparse.Namespace) -> int:
    """Commande ``backtest`` : télécharge les données et exécute la simulation."""
    settings = Settings()
    logger = configure_logging(settings)
    provider: MarketDataProvider = YFinanceProvider()
    if not args.no_cache:
        provider = CachedProvider(
            provider,
            settings.state_dir / "cache",
            refresh=args.refresh,
        )

    try:
        start = datetime.combine(date.fromisoformat(args.start), dt_time.min, tzinfo=UTC)
        end = datetime.combine(
            date.fromisoformat(args.end), dt_time.min, tzinfo=UTC
        ) + timedelta(days=1)
    except ValueError as exc:
        logger.error("dates invalides: %s", exc)
        return 2
    if end <= start:
        logger.error("la date de fin doit être postérieure à la date de début")
        return 2

    try:
        strategy = build_strategy(args.strategy, args, settings=settings)
    except ValueError as exc:
        logger.error("paramètres de stratégie invalides: %s", exc)
        return 2
    except ConfigurationError as exc:
        logger.error("%s", exc)
        return 2

    # Période de warm-up (200 barres journalières ~ 320 jours calendaires) pour
    # que la SMA(200) et les calculs de momentum soient actifs dès le jour 1.
    warmup_days = 320 if args.interval in {"1d", "1Day"} else 30
    fetch_start = start - timedelta(days=warmup_days)

    data: dict[str, BarFrame] = {}
    for symbol in Settings(symbols=args.symbols).symbols:
        try:
            data[symbol] = provider.fetch(symbol, fetch_start, end, interval=args.interval)
        except DataError as exc:
            logger.error("%s", exc)
            return 1
        logger.info(
            "%s: %d barres (%s → %s)",
            symbol,
            len(data[symbol]),
            data[symbol].index[0].date(),
            data[symbol].index[-1].date(),
        )

    # Le benchmark n'est PAS ajouté à ``data`` : le moteur ne doit pas le
    # trader. Il pilote le régime de marché (SMA longue) et sert de référence
    # de force relative aux stratégies qui en demandent un.
    benchmark_symbol = settings.benchmark_symbol
    benchmark_bars: BarFrame | None = None
    try:
        benchmark_bars = provider.fetch(
            benchmark_symbol, fetch_start, end, interval=args.interval
        )
        _inject_benchmark(strategy, benchmark_bars, benchmark_symbol)
    except (DataError, StrategyError) as exc:
        logger.error("benchmark %s indisponible: %s", benchmark_symbol, exc)
        return 1
    logger.info(
        "%s (benchmark): %d barres (%s → %s)",
        benchmark_symbol,
        len(benchmark_bars),
        benchmark_bars.index[0].date(),
        benchmark_bars.index[-1].date(),
    )

    config = BacktestConfig(
        initial_equity=args.initial_equity,
        commission_pct=args.commission,
        slippage_pct=args.slippage,
    )
    runner = BacktestRunner(
        strategy,
        config=config,
        risk_config=settings.risk,
    )
    result = runner.run(data, benchmark=benchmark_bars, start_date=start)
    print_backtest_summary(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Résultats JSON : {args.output}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        result.equity_curve.to_csv(args.csv)
        print(f"Courbe d'équity : {args.csv}")
    if args.trades:
        args.trades.parent.mkdir(parents=True, exist_ok=True)
        result.trades_frame().to_csv(args.trades, index=False)
        print(f"Trades CSV : {args.trades}")
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    """Commande ``web`` : interface locale, pensée pour les non-financiers."""
    settings = _settings_from_args(args)
    configure_logging(settings)
    from trading_bot.web.app import serve

    try:
        serve(settings, host=args.host, port=args.port)
    except (ConfigurationError, ValueError) as exc:
        logging.getLogger("trading_bot").error("%s", exc)
        return 2
    return 0


def _withdrawal_payload(kind: str, result: WithdrawalResult) -> dict[str, Any]:
    """Résumé JSON d'un retrait, identique en CLI et dans l'interface."""
    return {
        "ok": not result.errors,
        "kind": kind,
        "fills": [
            {
                "symbol": fill.symbol,
                "qty": fill.qty,
                "price": fill.price,
                "cash_freed": round(fill.cash_freed, 2),
                "reason": fill.reason,
            }
            for fill in result.fills
        ],
        "cash": round(result.cash, 2),
        "floor": result.floor,
        "errors": list(result.errors),
    }


def cmd_withdraw(args: argparse.Namespace) -> int:
    """Commande ``withdraw`` : retrait manuel ou sanctuarisation de cash."""
    settings = _settings_from_args(args)
    logger = configure_logging(settings)
    try:
        broker = AlpacaPaperClient(settings)
    except (ConfigurationError, BrokerError) as exc:
        logger.error("%s", exc)
        return 2

    service = WithdrawalService(
        settings,
        broker,
        state_store=StateStore(settings.state_dir / "state.json"),
        audit=AuditLog(settings.state_dir / "audit.jsonl"),
        env_path=PROJECT_ROOT / ".env",
    )

    try:
        if getattr(args, "free_cash", None) is not None:
            result = service.withdraw_free_cash(float(args.free_cash))
            print(
                json.dumps(
                    _withdrawal_payload("free_cash", result), indent=2, ensure_ascii=False
                )
            )
            return 0 if not result.errors else 1

        symbol = getattr(args, "symbol", None)
        if not symbol:
            logger.error(
                "veuillez spécifier soit --symbol avec (--amount ou --pct), "
                "soit --free-cash"
            )
            return 2
        result = service.withdraw_symbol(
            str(symbol),
            amount=getattr(args, "amount", None),
            pct=getattr(args, "pct", None),
        )
        print(
            json.dumps(_withdrawal_payload("symbol", result), indent=2, ensure_ascii=False)
        )
        return 0 if not result.errors else 1
    except (BrokerError, ValueError) as exc:
        logger.error("%s", exc)
        return 2


def apply_equity_rebase(
    settings: Settings,
    snapshot: AccountSnapshot,
    *,
    reason: str,
    state_store: StateStore,
    audit: AuditLog,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Ré-ancre le pic d'équité sur la valeur courante et rend un résumé.

    Partagé par la commande ``rebase`` et l'interface web : le raisonnement
    (pourquoi, avec quelle justification, ce qui est levé) ne doit exister
    qu'une fois.

    Args:
        settings: Configuration effective.
        snapshot: Compte lu chez le courtier.
        reason: Justification en clair, recopiée dans l'audit.
        state_store: Persistance de l'état de risque.
        audit: Journal d'audit.
        dry_run: Si ``True``, n'écrit rien.

    Returns:
        ``{"previous_peak", "new_peak", "halt_lifted", "dry_run"}``.

    Raises:
        RiskError: justification vide ou équité invalide.
    """
    state = state_store.load()
    account = AccountState(
        equity=snapshot.equity,
        cash=snapshot.cash,
        buying_power=snapshot.buying_power,
        peak_equity=state.peak_equity,
        day_start_equity=state.day_start_equity,
        halted=state.halted,
        halt_scope=HaltScope(state.halt_scope) if state.halt_scope else None,
        halt_reason=state.halt_reason,
    )
    previous_peak, lifted = RiskEngine(settings.risk).rebase_equity_reference(
        account, reason=reason
    )
    payload: dict[str, Any] = {
        "previous_peak": previous_peak,
        "new_peak": account.equity,
        "halt_lifted": lifted,
        "dry_run": dry_run,
    }
    if dry_run:
        return payload

    state.peak_equity = account.peak_equity
    state.day_start_equity = account.day_start_equity
    state.halted = account.halted
    state.halt_scope = account.halt_scope.value if account.halt_scope else None
    state.halt_reason = account.halt_reason
    state_store.save(state)
    audit.record(
        "equity_rebase",
        {
            "reason": reason,
            "previous_peak": previous_peak,
            "new_peak": account.peak_equity,
            "halt_lifted": lifted,
            "equity": account.equity,
        },
    )
    return payload


def cmd_rebase(args: argparse.Namespace) -> int:
    """Commande ``rebase`` : ré-ancre le pic d'équité après un mouvement de cash.

    Un dépôt ou un retrait fait chez le courtier modifie l'équité sans qu'aucun
    trade n'ait eu lieu. Le drawdown mesuré depuis l'ancien pic devient alors
    fictif et déclenche l'arrêt permanent. Cette commande, explicitement
    demandée avec sa justification, recale la référence et lève l'arrêt.
    """
    settings = _settings_from_args(args)
    logger = configure_logging(settings)
    reason = str(getattr(args, "reason", "") or "").strip()
    if not reason:
        logger.error("une justification est requise : --reason « retrait de X $ »")
        return 2

    try:
        broker = AlpacaPaperClient(settings)
        snapshot = broker.get_account()
    except (ConfigurationError, BrokerError) as exc:
        logger.error("%s", exc)
        return 2

    try:
        payload = apply_equity_rebase(
            settings,
            snapshot,
            reason=reason,
            state_store=StateStore(settings.state_dir / "state.json"),
            audit=AuditLog(settings.state_dir / "audit.jsonl"),
            dry_run=bool(getattr(args, "dry_run", False)),
        )
    except RiskError as exc:
        logger.error("%s", exc)
        return 2

    logger.info(
        "pic d'équité recalé : %s -> %.2f (%s)",
        f"{payload['previous_peak']:.2f}"
        if payload["previous_peak"] is not None
        else "n/a",
        payload["new_peak"],
        reason,
    )
    print(json.dumps({"ok": True, **payload}, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construit le parseur de commandes."""
    parser = argparse.ArgumentParser(
        prog="trading-bot",
        description=(
            "Bot de trading systématique restreint au paper trading Alpaca. "
            "Aucun ordre réel ne peut être transmis."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    live = subparsers.add_parser("live", help="boucle paper trading")
    live.add_argument("--once", action="store_true", help="exécuter un seul cycle")
    live.add_argument(
        "--watch-only",
        action="store_true",
        help="mode vigie horaire : surveillance temps réel des stops et news sans nouveaux achats",
    )
    live.add_argument(
        "--dry-run",
        action="store_true",
        help="analyser et journaliser sans transmettre d'ordre",
    )
    live.add_argument("--symbols", help="univers CSV (ex: AAPL,MSFT)")
    live.add_argument(
        "--strategy",
        choices=STRATEGY_NAMES,
        default="ema_cross",
        help="moteur de signaux (défaut: ema_cross)",
    )
    live.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    live.set_defaults(func=cmd_live)

    check = subparsers.add_parser(
        "check", help="diagnostic lecture seule (compte, données, signal)"
    )
    check.add_argument("--symbols", help="univers CSV")
    check.add_argument(
        "--strategy",
        choices=STRATEGY_NAMES,
        default="ema_cross",
        help="moteur de signaux (défaut: ema_cross)",
    )
    check.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    check.set_defaults(func=cmd_check)

    web = subparsers.add_parser(
        "web",
        help="interface web locale (tableau de bord, backtests, pilotage)",
    )
    web.add_argument(
        "--host",
        default="127.0.0.1",
        help="hôte d'écoute (localhost uniquement, par sécurité)",
    )
    web.add_argument("--port", type=int, default=8484)
    web.add_argument("--symbols", help="univers CSV")
    web.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    web.set_defaults(func=cmd_web)

    news = subparsers.add_parser(
        "news",
        help="radar de sentiment des actualités récentes",
    )
    news.add_argument(
        "--symbols",
        help="tickers à noter, séparés par des virgules (défaut : l'univers)",
    )
    news.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    news.set_defaults(func=cmd_news)

    publish = subparsers.add_parser(
        "publish-stats",
        help="republier l'historique local vers GitHub (rattrapage)",
    )
    publish.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    publish.set_defaults(func=cmd_publish_stats)

    backtest = subparsers.add_parser(
        "backtest", help="backtest déterministe sur données yfinance"
    )
    backtest.add_argument("--symbols", default="AAPL,MSFT")
    backtest.add_argument("--start", default="2022-01-01")
    backtest.add_argument("--end", default=date.today().isoformat())
    backtest.add_argument("--interval", default="1d")
    backtest.add_argument("--initial-equity", type=float, default=100_000.0)
    backtest.add_argument("--commission", type=float, default=0.001)
    backtest.add_argument("--slippage", type=float, default=0.001)
    backtest.add_argument(
        "--strategy",
        choices=STRATEGY_NAMES,
        default="ema_cross",
        help="moteur de signaux (défaut: ema_cross)",
    )
    backtest.add_argument("--fast", type=int, default=20, help="EMA rapide (ema_cross)")
    backtest.add_argument("--slow", type=int, default=50, help="EMA lente (ema_cross)")
    backtest.add_argument(
        "--entry", type=int, default=20, help="canal d'entrée (donchian_breakout)"
    )
    backtest.add_argument(
        "--exit-channel",
        type=int,
        default=10,
        help="canal de sortie (donchian_breakout)",
    )
    backtest.add_argument(
        "--sma-long", type=int, default=200, help="filtre de tendance (donchian_breakout)"
    )
    backtest.add_argument(
        "--mom-fast", type=int, default=60, help="momentum rapide (relative_momentum)"
    )
    backtest.add_argument(
        "--mom-slow", type=int, default=120, help="momentum lent (relative_momentum)"
    )
    backtest.add_argument(
        "--shield-pct",
        type=float,
        default=None,
        help="allocation bouclier (shield_sword, défaut: SHIELD_ALLOCATION_PCT)",
    )
    backtest.add_argument(
        "--sword-pct",
        type=float,
        default=None,
        help="allocation épée (shield_sword, défaut: SWORD_ALLOCATION_PCT)",
    )
    backtest.add_argument(
        "--sword-top-n",
        type=int,
        default=None,
        help="nombre d'actions dans l'épée (shield_sword, défaut: SWORD_TOP_N)",
    )
    backtest.add_argument(
        "--sword-rebalance-days",
        type=int,
        default=None,
        help="jours entre rebalancements (shield_sword, défaut: SWORD_REBALANCE_DAYS)",
    )
    backtest.add_argument(
        "--sword-trend-sma",
        type=int,
        default=None,
        help="SMA de tendance (shield_sword, défaut: SWORD_TREND_SMA)",
    )
    backtest.add_argument("--output", type=Path, help="fichier JSON de résultats")
    backtest.add_argument("--csv", type=Path, help="fichier CSV de la courbe d'équity")
    backtest.add_argument("--trades", type=Path, help="fichier CSV des trades")
    backtest.add_argument(
        "--no-cache",
        action="store_true",
        help="ignorer le cache disque et retélécharger les barres",
    )
    backtest.add_argument(
        "--refresh",
        action="store_true",
        help="retélécharger les barres puis réécrire le cache",
    )
    backtest.set_defaults(func=cmd_backtest)

    withdraw = subparsers.add_parser(
        "withdraw", help="retrait de cash manuel ou sanctuarisation de plancher"
    )
    withdraw.add_argument("--symbol", help="symbole à vendre (ex: NVDA)")
    withdraw.add_argument("--amount", type=float, help="montant en dollars à liquider")
    withdraw.add_argument(
        "--pct", type=float, help="pourcentage de la ligne à liquider (0-100)"
    )
    withdraw.add_argument(
        "--free-cash", type=float, help="montant de cash cible à sanctuariser"
    )
    withdraw.add_argument("--dry-run", action="store_true", help="simuler sans transmettre")
    withdraw.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    withdraw.set_defaults(func=cmd_withdraw)

    rebase = subparsers.add_parser(
        "rebase",
        help="ré-ancrer le pic d'équité après un dépôt ou un retrait chez le courtier",
    )
    rebase.add_argument(
        "--reason",
        required=True,
        help="justification en clair, recopiée dans audit.jsonl",
    )
    rebase.add_argument(
        "--dry-run", action="store_true", help="afficher l'effet sans rien enregistrer"
    )
    rebase.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    rebase.set_defaults(func=cmd_rebase)

    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée CLI, retourne un code de sortie."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigurationError as exc:
        logging.getLogger("trading_bot").error("%s", exc)
        return 2
    except KeyboardInterrupt:
        return 130


def main() -> None:
    """Entrée console ``python -m trading_bot.main``."""
    raise SystemExit(cli(sys.argv[1:]))


if __name__ == "__main__":
    main()
