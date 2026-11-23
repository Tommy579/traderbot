"""Moteur de prédiction multi-horizons purement informatif (lecture seule).

Ce module alimente exclusivement l'onglet « Prédiction » de l'interface web.
Il ne modifie aucun fichier d'état (``state.json``, ``history.jsonl``,
``audit.jsonl``), ne passe aucun ordre et n'influence jamais les décisions
réelles ou paper du bot.

Sources croisées :
1. Le LLM d'actualités (:class:`~trading_bot.data.news_sentiment.NewsAssessment`),
   y compris l'actualité macro du benchmark (SPY) ;
2. La mémoire longue des marchés passés et les vrais épisodes de krach détectés
   sur l'historique du benchmark (10+ ans lorsque disponible) ;
3. Un bootstrap Monte Carlo par blocs historiques (déterministe, préservant les
   corrélations inter-actifs et les queues épaisses) croisé avec la simulation
   des 4 stratégies du bot depuis le portefeuille réel détenu.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import numpy as np
import pandas as pd

from trading_bot.backtest.runner import BacktestError, BacktestResult, BacktestRunner
from trading_bot.bot import STRATEGY_NAMES, _inject_benchmark, build_strategy
from trading_bot.config import BacktestConfig, Settings
from trading_bot.data.loader import OHLCV_COLUMNS, compute_atr, momentum_zscores
from trading_bot.data.news_sentiment import NewsAssessment
from trading_bot.main import news_effect_label
from trading_bot.risk.engine import Position, RiskEngine
from trading_bot.strategies.base import SignalType, StrategyError
from trading_bot.strategies.shield_sword import ShieldSwordStrategy

__all__ = [
    "HISTORICAL_CRASH_ANALOGUES",
    "build_market_prediction",
]

_DISCLAIMER: Final[str] = (
    "Projection indicative uniquement : cette simulation croise les actualités "
    "notées par le LLM (titres et macro), un bootstrap Monte Carlo de blocs "
    "historiques incluant les krachs passés, ainsi que les règles des 4 "
    "stratégies du bot. Elle n'est jamais utilisée par le bot pour trader."
)

_MC_SIMULATIONS: Final[int] = 200
_MC_BLOCK_SIZE: Final[int] = 5

_STRATEGY_LABELS: Final[dict[str, str]] = {
    "donchian_breakout": "Cassure de tendance",
    "relative_momentum": "Force relative",
    "ema_cross": "Croisement de moyennes",
    "shield_sword": "Bouclier + Épée",
}

_EXIT_REASON_LABELS: Final[dict[str, str]] = {
    "stop_loss": "Sortie sur filet de sécurité (stop ATR)",
    "trailing_stop": "Sortie sur filet remonté (trailing stop)",
    "stop_gap": "Sortie sur trou d'ouverture sous le stop",
    "strategy_exit": "Signal de sortie de la stratégie",
    "circuit_breaker": "Liquidation coupe-circuit (protection capital)",
    "partial_take_profit": "Prise de profit partielle (+20 %)",
    "position_trim_overweight": "Écrémage automatique (poids > 25 %)",
    "news_panic_exit": "Sortie d'urgence déclenchée par le LLM d'actualités",
    "safe_haven_unwind": "Allègement défensif pour financer un achat offensif",
}


@dataclass(frozen=True, slots=True)
class CrashAnalogue:
    """Profil type d'un grand krach historique et réaction prévue du bot."""

    key: str
    title: str
    period: str
    drop_pct: float
    duration_days: int
    summary: str
    bot_defense: str

    def to_dict(
        self, *, similarity: float = 0.0, is_closest: bool = False
    ) -> dict[str, Any]:
        """Sérialise l'analogie pour l'API JSON."""
        return {
            "key": self.key,
            "title": self.title,
            "period": self.period,
            "drop_pct": self.drop_pct,
            "duration_days": self.duration_days,
            "summary": self.summary,
            "bot_defense": self.bot_defense,
            "similarity": round(similarity, 3),
            "is_closest": is_closest,
        }


HISTORICAL_CRASH_ANALOGUES: Final[tuple[CrashAnalogue, ...]] = (
    CrashAnalogue(
        key="covid_2020",
        title="Krach éclair (type Mars 2020 — Choc pandémique)",
        period="Fév.-Mars 2020",
        drop_pct=-0.34,
        duration_days=23,
        summary=(
            "Chute brutale et rapide (-34 % en ~1 mois) avec explosion de la "
            "volatilité, suivie d'un rebond technique violent."
        ),
        bot_defense=(
            "Les stops ATR et le coupe-circuit journalier (-3 %) coupent les "
            "pertes dès les premières séances ; le Bouclier SGOV préserve ~70 % "
            "à 90 % du capital avant le rebond."
        ),
    ),
    CrashAnalogue(
        key="inflation_2022",
        title="Marché baissier prolongé (type 2022 — Choc taux & inflation)",
        period="Jan.-Oct. 2022",
        drop_pct=-0.25,
        duration_days=190,
        summary=(
            "Baisse graduelle sur plusieurs mois sous la moyenne mobile 200 jours "
            "avec faux rebonds réguliers."
        ),
        bot_defense=(
            "Le filtre de régime (SPY < SMA 200) bloque les nouveaux achats "
            "offensifs et bascule Bouclier + Épée sur le Bouclier défensif "
            "dès le rebalancement mensuel."
        ),
    ),
    CrashAnalogue(
        key="gfc_2008",
        title="Crise systémique (type 2008 — Krach financier)",
        period="2008-2009",
        drop_pct=-0.52,
        duration_days=250,
        summary=(
            "Effondrement prolongé avec chocs journaliers répétés et corrélation "
            "élevée entre presque toutes les actions."
        ),
        bot_defense=(
            "Le coupe-circuit permanent à -15 % depuis le sommet et le plancher "
            "dur d'équité liquident l'exposition offensive pour éviter la ruine."
        ),
    ),
    CrashAnalogue(
        key="tech_correction",
        title="Correction de valorisation / rotation (type 2000 / 2025)",
        period="Correction sectorielle",
        drop_pct=-0.18,
        duration_days=45,
        summary=(
            "Repli ciblé de 15 à 20 % sur les leaders de momentum après une "
            "phase d'euphorie et de forte extension."
        ),
        bot_defense=(
            "La prise de profit partielle à +20 %, l'écrémage à 25 % et la "
            "rotation momentum sécurisent une part des gains avant le repli."
        ),
    ),
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _horizon_label(days: int) -> str:
    weeks = max(1, round(days / 5.0))
    if days >= 42:
        months = round(days / 21.0)
        return f"+{months} mois (~{weeks} sem.)"
    if days >= 18:
        return f"+1 mois (~{weeks} sem.)"
    return f"+{weeks} sem."


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("index DatetimeIndex requis")
    if idx.tz is None:
        out.index = idx.tz_localize("UTC")
    else:
        out.index = idx.tz_convert("UTC")
    return out.sort_index()


def _future_business_dates(last_timestamp: pd.Timestamp, count: int) -> pd.DatetimeIndex:
    """Génère ``count`` séances futures (lundi-vendredi, UTC) après ``last_timestamp``."""
    base = (
        last_timestamp.tz_convert("UTC")
        if last_timestamp.tzinfo
        else last_timestamp.tz_localize("UTC")
    )
    cursor = base.normalize() + timedelta(days=1)
    if cursor <= base:
        cursor = cursor + timedelta(days=1)
    dates: list[pd.Timestamp] = []
    while len(dates) < count:
        if cursor.weekday() < 5:
            dates.append(cursor)
        cursor = cursor + timedelta(days=1)
    return pd.DatetimeIndex(dates)


def _is_defensive(symbol: str, settings: Settings) -> bool:
    return symbol == settings.risk.shield_symbol or (
        settings.safe_haven_enabled and symbol in settings.safe_haven_symbols
    )


def _extract_crash_episodes(bars: pd.DataFrame, *, min_drop: float = -0.10) -> list[dict[str, Any]]:
    """Détecte les principaux épisodes de krach / correction réels dans l'historique."""
    close = bars["close"].astype(float)
    if len(close) < 10:
        return []
    running_max = close.cummax()
    dd = close / running_max - 1.0

    episodes: list[dict[str, Any]] = []
    in_dd = False
    peak_idx = close.index[0]
    trough_idx = close.index[0]
    trough_val = 0.0

    for i, ts in enumerate(close.index):
        val = float(dd.iloc[i])
        if val < 0.0:
            if not in_dd:
                in_dd = True
                peak_idx = close.index[max(0, i - 1)]
                trough_idx = ts
                trough_val = val
            elif val < trough_val:
                trough_idx = ts
                trough_val = val
        elif in_dd:
            if trough_val <= min_drop:
                dur = len(close.loc[peak_idx:trough_idx]) - 1
                episodes.append(
                    {
                        "peak_date": pd.Timestamp(peak_idx).date().isoformat(),
                        "trough_date": pd.Timestamp(trough_idx).date().isoformat(),
                        "drop_pct": round(trough_val, 4),
                        "duration_bars": max(1, dur),
                        "recovered": True,
                    }
                )
            in_dd = False
            trough_val = 0.0

    if in_dd and trough_val <= min_drop:
        dur = len(close.loc[peak_idx:trough_idx]) - 1
        episodes.append(
            {
                "peak_date": pd.Timestamp(peak_idx).date().isoformat(),
                "trough_date": pd.Timestamp(trough_idx).date().isoformat(),
                "drop_pct": round(trough_val, 4),
                "duration_bars": max(1, dur),
                "recovered": False,
            }
        )

    episodes.sort(key=lambda ep: ep["drop_pct"])
    return episodes[:4]


def _compute_observed_events(bars: pd.DataFrame) -> dict[str, Any]:
    """Détecte les chocs et plus fortes chutes réellement observés dans le passé."""
    close = bars["close"].astype(float)
    returns = close.pct_change().dropna()
    running_max = close.cummax()
    drawdown = close / running_max - 1.0

    trough_ts = drawdown.idxmin()
    max_dd = float(drawdown.loc[trough_ts]) if not drawdown.empty else 0.0
    before_trough = close.loc[:trough_ts]
    peak_ts = before_trough.idxmax() if not before_trough.empty else trough_ts
    duration_bars = len(close.loc[peak_ts:trough_ts]) - 1 if peak_ts <= trough_ts else 0

    worst_day_ret = float(returns.min()) if not returns.empty else 0.0
    worst_day_ts = returns.idxmin() if not returns.empty else close.index[-1]
    best_day_ret = float(returns.max()) if not returns.empty else 0.0
    best_day_ts = returns.idxmax() if not returns.empty else close.index[-1]

    roll_20 = close.pct_change(20).dropna()
    worst_20d = float(roll_20.min()) if not roll_20.empty else 0.0
    worst_20d_ts = roll_20.idxmin() if not roll_20.empty else close.index[-1]
    best_20d = float(roll_20.max()) if not roll_20.empty else 0.0
    best_20d_ts = roll_20.idxmax() if not roll_20.empty else close.index[-1]

    shock_days = int((returns.abs() >= 0.025).sum()) if not returns.empty else 0
    recent_shock_days = (
        int((returns.tail(20).abs() >= 0.02).sum()) if not returns.empty else 0
    )
    crash_episodes = _extract_crash_episodes(bars, min_drop=-0.10)

    return {
        "bars_analyzed": len(bars),
        "history_start": close.index[0].date().isoformat(),
        "history_end": close.index[-1].date().isoformat(),
        "max_drawdown_pct": round(max_dd, 4),
        "max_drawdown_peak_date": pd.Timestamp(peak_ts).date().isoformat(),
        "max_drawdown_trough_date": pd.Timestamp(trough_ts).date().isoformat(),
        "max_drawdown_duration_bars": max(0, duration_bars),
        "worst_daily_return_pct": round(worst_day_ret, 4),
        "worst_daily_date": pd.Timestamp(worst_day_ts).date().isoformat(),
        "best_daily_return_pct": round(best_day_ret, 4),
        "best_daily_date": pd.Timestamp(best_day_ts).date().isoformat(),
        "worst_20d_return_pct": round(worst_20d, 4),
        "worst_20d_date": pd.Timestamp(worst_20d_ts).date().isoformat(),
        "best_20d_return_pct": round(best_20d, 4),
        "best_20d_date": pd.Timestamp(best_20d_ts).date().isoformat(),
        "shock_days_count": shock_days,
        "recent_shock_days_20d": recent_shock_days,
        "crash_episodes": crash_episodes,
    }


def _build_synthetic_benchmark(
    bars_by_symbol: Mapping[str, pd.DataFrame], settings: Settings
) -> pd.DataFrame | None:
    """Construit un indice équipondéré si le benchmark n'a pas pu être chargé."""
    candidates = [
        df
        for sym, df in bars_by_symbol.items()
        if not df.empty and not _is_defensive(sym, settings)
    ]
    if not candidates:
        candidates = [df for df in bars_by_symbol.values() if not df.empty]
    if not candidates:
        return None
    aligned_closes = (
        pd.concat([df["close"].astype(float) for df in candidates], axis=1).ffill().bfill()
    )
    mean_close = aligned_closes.mean(axis=1)
    out = pd.DataFrame(index=mean_close.index)
    out["open"] = mean_close
    out["high"] = mean_close * 1.005
    out["low"] = mean_close * 0.995
    out["close"] = mean_close
    out["volume"] = 1_000_000.0
    return out


def _summarize_news_impact(
    settings: Settings,
    symbols: Sequence[str],
    news_assessments: Mapping[str, NewsAssessment] | None,
) -> dict[str, Any]:
    """Synthétise l'influence des notes du LLM d'actualités (titres + macro SPY)."""
    assessments = dict(news_assessments or {})
    engine = RiskEngine(settings.risk)
    panic_threshold = settings.risk.news_panic_exit_threshold
    bench_sym = settings.benchmark_symbol

    if not assessments:
        return {
            "consulted": False,
            "model": settings.openrouter_model,
            "average_score": 0.0,
            "macro_score": None,
            "macro_rationale": None,
            "mood_label": "Neutre (aucune actualité analysée)",
            "critical_symbols": [],
            "vetoed_symbols": [],
            "panic_symbols": [],
            "summary_text": (
                "Aucune note d'actualité récente disponible : le module de "
                "prédiction s'appuie uniquement sur l'historique des prix et "
                "les stratégies."
            ),
            "per_symbol": {},
        }

    per_symbol: dict[str, dict[str, Any]] = {}
    scores: list[float] = []
    critical_symbols: list[str] = []
    vetoed_symbols: list[str] = []
    panic_symbols: list[str] = []

    target_symbols = list(dict.fromkeys([bench_sym, *symbols, *assessments.keys()]))
    for sym in target_symbols:
        note = assessments.get(sym)
        if note is None:
            continue
        scores.append(note.sentiment_score)
        veto = engine.news_vetoes_entry(note)
        panic = note.is_critical_threat and note.sentiment_score <= panic_threshold
        if note.is_critical_threat:
            critical_symbols.append(sym)
        if veto and sym != bench_sym:
            vetoed_symbols.append(sym)
        if panic and sym != bench_sym:
            panic_symbols.append(sym)
        impulse = note.sentiment_score * max(0.35, note.confidence)
        if note.is_critical_threat:
            impulse -= 0.25
        per_symbol[sym] = {
            "score": round(note.sentiment_score, 3),
            "confidence": round(note.confidence, 3),
            "critical": note.is_critical_threat,
            "articles": note.articles_count,
            "rationale": note.rationale,
            "effect": news_effect_label(note, engine=engine, panic=panic_threshold),
            "blocks_entry": veto,
            "triggers_panic": panic,
            "impulse": round(_clamp(impulse, -1.25, 1.0), 3),
        }

    macro_note = per_symbol.get(bench_sym)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    if macro_note is not None:
        # L'actualité macro du benchmark pèse double sur la tonalité globale.
        avg_score = (avg_score * len(scores) + float(macro_note["score"])) / (
            len(scores) + 1
        )

    if panic_symbols or len(critical_symbols) >= 2 or (macro_note and macro_note["critical"]):
        mood = "Alerte critique"
    elif avg_score <= -0.35 or vetoed_symbols:
        mood = "Défensif / Négatif"
    elif avg_score <= -0.10:
        mood = "Prudent"
    elif avg_score >= 0.25:
        mood = "Optimiste"
    else:
        mood = "Neutre à modéré"

    parts: list[str] = [
        f"Score moyen du LLM ({settings.openrouter_model}) : {avg_score:+.2f} ({mood.lower()})."
    ]
    if macro_note is not None:
        parts.append(
            f"Macro {bench_sym} : {macro_note['score']:+.2f} ({macro_note['rationale']})."
        )
    if critical_symbols:
        parts.append(f"Menace critique détectée sur : {', '.join(critical_symbols)}.")
    if vetoed_symbols:
        parts.append(
            f"Achats bloqués par le filtre d'actualités sur : {', '.join(vetoed_symbols)}."
        )
    if not critical_symbols and not vetoed_symbols:
        parts.append("Aucun veto d'achat ni menace critique active.")

    return {
        "consulted": True,
        "model": settings.openrouter_model,
        "average_score": round(avg_score, 3),
        "macro_score": macro_note["score"] if macro_note else None,
        "macro_rationale": macro_note["rationale"] if macro_note else None,
        "mood_label": mood,
        "critical_symbols": critical_symbols,
        "vetoed_symbols": vetoed_symbols,
        "panic_symbols": panic_symbols,
        "summary_text": " ".join(parts),
        "per_symbol": per_symbol,
    }


def _analyze_past_market(
    benchmark_bars: pd.DataFrame,
    news_outlook: Mapping[str, Any],
    settings: Settings,
    *,
    benchmark_long_bars: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Analyse la mémoire des marchés passés, la volatilité et le risque de krach."""
    close = benchmark_bars["close"].astype(float)
    returns = close.pct_change().dropna()
    last_close = float(close.iloc[-1])

    sma_50 = float(close.rolling(min(50, len(close))).mean().iloc[-1])
    sma_200 = float(close.rolling(min(200, len(close))).mean().iloc[-1])
    ret_20d = (
        float(close.iloc[-1] / close.iloc[max(0, len(close) - 21)] - 1.0)
        if len(close) >= 2
        else 0.0
    )
    ret_60d = (
        float(close.iloc[-1] / close.iloc[max(0, len(close) - 61)] - 1.0)
        if len(close) >= 2
        else 0.0
    )
    ret_120d = (
        float(close.iloc[-1] / close.iloc[max(0, len(close) - 121)] - 1.0)
        if len(close) >= 2
        else 0.0
    )

    vol_20d = (
        float(returns.tail(20).std(ddof=1) * math.sqrt(252))
        if len(returns.tail(20)) >= 2
        else 0.15
    )
    vol_60d = (
        float(returns.tail(60).std(ddof=1) * math.sqrt(252))
        if len(returns.tail(60)) >= 2
        else vol_20d
    )
    vol_20d = vol_20d if math.isfinite(vol_20d) and vol_20d > 0 else 0.15
    vol_60d = vol_60d if math.isfinite(vol_60d) and vol_60d > 0 else vol_20d

    running_max = close.cummax()
    current_dd = (
        float(last_close / float(running_max.iloc[-1]) - 1.0)
        if float(running_max.iloc[-1]) > 0
        else 0.0
    )

    history_for_events = (
        benchmark_long_bars
        if benchmark_long_bars is not None and len(benchmark_long_bars) > len(benchmark_bars)
        else benchmark_bars
    )
    observed = _compute_observed_events(history_for_events)
    recent_events = _compute_observed_events(benchmark_bars)
    above_sma200 = last_close > sma_200
    above_sma50 = last_close > sma_50

    score = 0.15
    drivers: list[str] = []

    vol_ratio = vol_20d / max(vol_60d, 0.05)
    if vol_ratio >= 1.35 or vol_20d >= 0.28:
        score += 0.22
        drivers.append(
            f"Volatilité en hausse ({vol_20d * 100:.1f} % annualisée sur 20j vs {vol_60d * 100:.1f} % sur 60j)"
        )
    elif vol_ratio <= 0.90 and vol_20d < 0.16:
        score -= 0.05
        drivers.append(f"Volatilité contenue ({vol_20d * 100:.1f} % annualisée)")

    if not above_sma200:
        score += 0.20
        drivers.append(
            f"Indice {settings.benchmark_symbol} sous sa moyenne 200j (régime prudent/baissier)"
        )
    else:
        stretch = last_close / sma_200 - 1.0
        if stretch > 0.18:
            score += 0.10
            drivers.append(
                f"Marché étiré (+{stretch * 100:.1f} % au-dessus de la SMA 200j, risque de consolidation)"
            )
        else:
            drivers.append(
                f"Indice {settings.benchmark_symbol} au-dessus de sa SMA 200j (+{stretch * 100:.1f} %)"
            )

    if current_dd <= -0.10:
        score += 0.18
        drivers.append(f"Correction déjà engagée ({current_dd * 100:.1f} % sous le sommet)")
    elif current_dd <= -0.05:
        score += 0.08
        drivers.append(f"Repli modéré depuis le sommet ({current_dd * 100:.1f} %)")

    if recent_events["recent_shock_days_20d"] >= 2:
        score += 0.10
        drivers.append(
            f"{recent_events['recent_shock_days_20d']} séances de forte amplitude (≥ 2 %) sur les 20 derniers jours"
        )

    if observed["max_drawdown_pct"] <= -0.20:
        score += 0.05
        drivers.append(
            f"Mémoire historique marquée par un krach de {observed['max_drawdown_pct'] * 100:.1f} % "
            f"({observed['max_drawdown_peak_date']} -> {observed['max_drawdown_trough_date']})"
        )

    avg_news = float(news_outlook.get("average_score", 0.0))
    macro_score = news_outlook.get("macro_score")
    critical_syms = news_outlook.get("critical_symbols") or []
    if critical_syms:
        score += 0.22
        drivers.append(f"Alerte critique du LLM d'actualités ({', '.join(critical_syms)})")
    elif macro_score is not None and float(macro_score) <= -0.40:
        score += 0.16
        drivers.append(
            f"Actualités macro {settings.benchmark_symbol} négatives ({float(macro_score):+.2f})"
        )
    elif avg_news <= -0.35:
        score += 0.14
        drivers.append(f"Sentiment d'actualités dégradé ({avg_news:+.2f})")
    elif avg_news >= 0.25:
        score -= 0.06
        drivers.append(f"Actualités porteuses selon le LLM ({avg_news:+.2f})")

    crash_score = round(_clamp(score, 0.05, 0.95), 2)
    if crash_score >= 0.70:
        level, label = "critique", "Risque de krach critique"
    elif crash_score >= 0.48:
        level, label = "élevé", "Risque de krach élevé"
    elif crash_score >= 0.28:
        level, label = "modéré", "Risque modéré / vigilance"
    else:
        level, label = "faible", "Risque de krach faible"

    similarities: dict[str, float] = {}
    for analogue in HISTORICAL_CRASH_ANALOGUES:
        if analogue.key == "covid_2020":
            sim = (
                0.4
                + 0.4 * _clamp((vol_ratio - 1.0) / 0.8, 0.0, 1.0)
                + (0.2 if critical_syms else 0.0)
            )
        elif analogue.key == "inflation_2022":
            sim = (
                0.35
                + (0.45 if not above_sma200 else 0.0)
                + 0.2 * _clamp(-ret_60d / 0.15, 0.0, 1.0)
            )
        elif analogue.key == "gfc_2008":
            sim = (
                0.20
                + 0.50 * _clamp(-current_dd / 0.25, 0.0, 1.0)
                + 0.30 * _clamp(vol_20d / 0.40, 0.0, 1.0)
            )
        else:
            stretch = max(0.0, last_close / sma_200 - 1.0)
            sim = (
                0.45
                + 0.35 * _clamp(stretch / 0.20, 0.0, 1.0)
                + 0.20 * _clamp(vol_20d / 0.25, 0.0, 1.0)
            )
        similarities[analogue.key] = _clamp(sim, 0.05, 0.98)

    closest_key = max(similarities, key=lambda k: similarities[k])
    analogues_payload = [
        analogue.to_dict(
            similarity=similarities[analogue.key],
            is_closest=(analogue.key == closest_key),
        )
        for analogue in HISTORICAL_CRASH_ANALOGUES
    ]

    regime = (
        "haussier"
        if above_sma200 and above_sma50
        else ("prudent" if above_sma200 else "baissier")
    )
    return {
        "benchmark": settings.benchmark_symbol,
        "last_close": round(last_close, 2),
        "sma_50": round(sma_50, 2),
        "sma_200": round(sma_200, 2),
        "above_sma200": above_sma200,
        "regime": regime,
        "ret_20d": round(ret_20d, 4),
        "ret_60d": round(ret_60d, 4),
        "ret_120d": round(ret_120d, 4),
        "vol_20d": round(vol_20d, 4),
        "vol_60d": round(vol_60d, 4),
        "current_drawdown_pct": round(current_dd, 4),
        "crash_risk": {
            "score": crash_score,
            "level": level,
            "label": label,
            "drivers": drivers,
        },
        "observed_events": observed,
        "crash_analogues": analogues_payload,
        "closest_analogue": closest_key,
    }


def _deterministic_seed(symbols: Sequence[str], last_ts: pd.Timestamp, horizon: int) -> int:
    key = f"{','.join(sorted(symbols))}|{last_ts.date().isoformat()}|{horizon}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def _returns_to_ohlcv_bars(
    hist_bars: pd.DataFrame,
    daily_returns: np.ndarray,
    future_dates: pd.DatetimeIndex,
    *,
    spread_mult: float = 1.0,
    vol_mult: float = 1.0,
) -> pd.DataFrame:
    """Convertit une séquence de rendements journaliers en barres OHLCV concaténées."""
    prev_close = float(hist_bars["close"].iloc[-1])
    med_vol = (
        float(hist_bars["volume"].tail(30).median())
        if "volume" in hist_bars.columns
        else 1_000_000.0
    )
    if not math.isfinite(med_vol) or med_vol <= 0:
        med_vol = 1_000_000.0
    hl_spread = float(
        ((hist_bars["high"] - hist_bars["low"]) / hist_bars["close"]).tail(20).median()
    )
    if not math.isfinite(hl_spread) or hl_spread <= 0:
        hl_spread = 0.012

    rows: list[dict[str, float]] = []
    for raw_r in daily_returns:
        daily_ret = _clamp(float(raw_r), -0.18, 0.15)
        next_close = max(0.01, prev_close * (1.0 + daily_ret))
        next_open = max(0.01, prev_close * (1.0 + 0.15 * daily_ret))
        half_range = max(0.001, 0.5 * hl_spread * spread_mult)
        next_high = max(next_open, next_close) * (1.0 + half_range)
        next_low = max(0.005, min(next_open, next_close) * (1.0 - half_range))
        rows.append(
            {
                "open": float(next_open),
                "high": float(next_high),
                "low": float(next_low),
                "close": float(next_close),
                "volume": float(med_vol * vol_mult),
            }
        )
        prev_close = next_close

    future_df = pd.DataFrame(rows, index=future_dates)
    hist_tail = hist_bars[list(OHLCV_COLUMNS)].astype(float)
    combined = pd.concat([hist_tail, future_df], axis=0)
    return combined[~combined.index.duplicated(keep="last")].sort_index()


def _project_future_bars_monte_carlo(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    benchmark_bars: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    *,
    news_outlook: Mapping[str, Any],
    crash_risk_score: float,
    settings: Settings,
    benchmark_long_bars: pd.DataFrame | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Bootstrap Monte Carlo par blocs historiques (préserve corrélations et queues épaisses).

    Produit 3 trajectoires jointes cohérentes (``central`` = P50, ``bull`` = P85,
    ``crash`` = P10 stressé par les krachs passés) ainsi que les statistiques
    probabilistes de la distribution.
    """
    horizon = len(future_dates)
    bench_key = "__BENCHMARK__"
    ordered_syms = sorted(bars_by_symbol)
    all_keys = [bench_key, *ordered_syms]

    series_map: dict[str, pd.Series] = {
        bench_key: benchmark_bars["close"].astype(float).pct_change(),
        **{
            sym: bars_by_symbol[sym]["close"].astype(float).pct_change()
            for sym in ordered_syms
        },
    }
    ret_df = pd.DataFrame(series_map).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(ret_df) > 1:
        ret_df = ret_df.iloc[1:]
    ret_matrix = ret_df[all_keys].to_numpy(dtype=float)
    n_hist = ret_matrix.shape[0]

    # Décomposition dérive/volatilité + impulsion LLM décroissante
    per_news = news_outlook.get("per_symbol", {})
    macro_note = per_news.get(settings.benchmark_symbol)
    macro_impulse = (
        float(macro_note["impulse"])
        if macro_note is not None
        else float(news_outlook.get("average_score", 0.0)) * 0.6
    )

    impulses = np.zeros(len(all_keys), dtype=float)
    defensive_mask = np.zeros(len(all_keys), dtype=bool)
    impulses[0] = macro_impulse
    for idx, sym in enumerate(ordered_syms, start=1):
        is_def = _is_defensive(sym, settings)
        defensive_mask[idx] = is_def
        sym_imp = float(per_news.get(sym, {}).get("impulse", macro_impulse * 0.45))
        impulses[idx] = 0.05 * sym_imp if is_def else sym_imp

    # Amortissement vers une dérive réaliste pour éviter d'extrapoler à l'infini
    hist_means = ret_matrix.mean(axis=0) if n_hist > 0 else np.zeros(len(all_keys))
    hist_stds = (
        ret_matrix.std(axis=0, ddof=1)
        if n_hist > 1
        else np.full(len(all_keys), 0.012)
    )
    hist_stds = np.where(np.isfinite(hist_stds) & (hist_stds > 1e-6), hist_stds, 0.012)
    baseline = np.where(defensive_mask, 0.00018, 0.00030)
    target_drift = np.clip(0.50 * hist_means + 0.50 * baseline, -0.0025, 0.0025)
    drift_shift = target_drift - hist_means

    # Pondération des blocs de départ selon le crash_risk (légère sur-pondération des blocs négatifs si risque élevé)
    block = min(_MC_BLOCK_SIZE, max(1, n_hist))
    max_start = max(1, n_hist - block + 1)
    block_bench_rets = np.array(
        [ret_matrix[s : s + block, 0].sum() for s in range(max_start)], dtype=float
    )
    stress_bias = max(0.0, crash_risk_score - 0.35)
    weights = np.where(block_bench_rets < 0.0, 1.0 + 1.8 * stress_bias, 1.0)
    weights = weights / weights.sum()

    seed = _deterministic_seed(ordered_syms, pd.Timestamp(benchmark_bars.index[-1]), horizon)
    rng = np.random.default_rng(seed)

    decay = np.exp(-np.arange(horizon, dtype=float) / 15.0)[:, None]
    news_tilt = 0.0045 * impulses[None, :] * decay
    risk_drag = np.where(
        defensive_mask, 0.0, -0.0006 * max(0.0, crash_risk_score - 0.45)
    )[None, :]

    # Tirage de _MC_SIMULATIONS trajectoires jointes par bootstrap de blocs
    paths = np.zeros((_MC_SIMULATIONS, horizon, len(all_keys)), dtype=float)
    n_blocks = math.ceil(horizon / block)
    for sim_idx in range(_MC_SIMULATIONS):
        if n_hist == 0:
            sampled = np.zeros((horizon, len(all_keys)), dtype=float)
        else:
            starts = rng.choice(max_start, size=n_blocks, p=weights)
            chunks = [ret_matrix[s : s + block] for s in starts]
            sampled = np.vstack(chunks)[:horizon]
        # Si la série historique est quasi-constante (tests synthétiques), ajoute
        # une dispersion déterministe minime proportionnelle à la volatilité pour
        # que la distribution P10/P50/P90 reste informative.
        jitter_scale = np.where(defensive_mask, 0.0002, np.maximum(0.004, hist_stds * 0.25))
        jitter = rng.normal(0.0, jitter_scale, size=(horizon, len(all_keys)))
        paths[sim_idx] = np.clip(
            sampled + drift_shift[None, :] + news_tilt + risk_drag + jitter,
            -0.18,
            0.15,
        )

    # Rendement cumulé du benchmark sur chaque trajectoire simulée
    bench_cum = np.prod(1.0 + paths[:, :, 0], axis=1)
    bench_final_rets = bench_cum - 1.0
    bench_equities = np.cumprod(1.0 + paths[:, :, 0], axis=1)
    bench_peaks = np.maximum.accumulate(bench_equities, axis=1)
    bench_max_dds = (bench_equities / bench_peaks - 1.0).min(axis=1)

    order = np.argsort(bench_final_rets)
    idx_p10 = int(order[max(0, round(0.10 * (_MC_SIMULATIONS - 1)))])
    idx_p50 = int(order[round(0.50 * (_MC_SIMULATIONS - 1))])
    idx_p85 = int(order[min(_MC_SIMULATIONS - 1, round(0.85 * (_MC_SIMULATIONS - 1)))])

    # Calibrage du scénario krach : combine le 10e centile Monte Carlo avec le pire
    # bloc baissier réel observé dans l'historique long (ou un choc calibré).
    crash_path = paths[idx_p10].copy()
    long_source = (
        benchmark_long_bars
        if benchmark_long_bars is not None and len(benchmark_long_bars) >= 30
        else benchmark_bars
    )
    long_ret = long_source["close"].astype(float).pct_change().dropna()
    shock_len = min(22, max(8, round(horizon * 0.35)))
    if len(long_ret) >= shock_len:
        rolling_sum = long_ret.rolling(shock_len).sum()
        worst_end_pos = int(np.argmin(rolling_sum.fillna(0.0).to_numpy()))
        start_pos = max(0, worst_end_pos - shock_len + 1)
        worst_bench_slice = long_ret.iloc[start_pos : start_pos + shock_len].to_numpy()
    else:
        worst_bench_slice = np.full(shock_len, -0.012)

    bench_worst_drop = float(np.prod(1.0 + worst_bench_slice) - 1.0)
    if bench_worst_drop > -0.12:
        # Assure un vrai scénario de stress même si l'historique chargé est calme.
        extra_per_day = (-0.16 - bench_worst_drop) / shock_len
        worst_bench_slice = np.clip(worst_bench_slice + extra_per_day, -0.12, 0.05)

    bench_std = max(float(hist_stds[0]), 0.008)
    for col_idx in range(len(all_keys)):
        if defensive_mask[col_idx]:
            crash_path[:, col_idx] = np.clip(
                np.maximum(crash_path[:, col_idx], 0.00012), -0.003, 0.003
            )
            continue
        beta = _clamp(float(hist_stds[col_idx]) / bench_std, 0.75, 1.85)
        shock_col = np.clip(
            worst_bench_slice * beta * (0.85 + 0.35 * crash_risk_score),
            -0.16,
            0.08,
        )
        crash_path[: len(shock_col), col_idx] = np.minimum(
            crash_path[: len(shock_col), col_idx], shock_col
        )

    # Scénario haussier : assure une surperformance claire sur P50
    bull_path = paths[idx_p85].copy()
    central_path = paths[idx_p50].copy()

    # Assure un ordre monotone clair sur les actions (bull > central > crash)
    for col_idx in range(len(all_keys)):
        if defensive_mask[col_idx]:
            continue
        c_end = float(np.prod(1.0 + central_path[:, col_idx]))
        b_end = float(np.prod(1.0 + bull_path[:, col_idx]))
        if b_end <= c_end * 1.01:
            bull_path[:, col_idx] += 0.0008

    scenario_paths = {
        "central": (central_path, 1.0, 1.0),
        "bull": (bull_path, 0.95, 1.0),
        "crash": (crash_path, 1.85, 1.5),
    }

    scenarios: dict[str, dict[str, Any]] = {}
    for name, (mat, s_mult, v_mult) in scenario_paths.items():
        bench_proj = _returns_to_ohlcv_bars(
            benchmark_bars,
            mat[:, 0],
            future_dates,
            spread_mult=s_mult,
            vol_mult=v_mult,
        )
        data_proj: dict[str, pd.DataFrame] = {}
        for col_idx, sym in enumerate(ordered_syms, start=1):
            data_proj[sym] = _returns_to_ohlcv_bars(
                bars_by_symbol[sym],
                mat[:, col_idx],
                future_dates,
                spread_mult=1.05 if defensive_mask[col_idx] else s_mult,
                vol_mult=1.0 if defensive_mask[col_idx] else v_mult,
            )
        scenarios[name] = {"benchmark": bench_proj, "data": data_proj}

    mc_summary = {
        "simulations": _MC_SIMULATIONS,
        "block_size_days": block,
        "prob_positive_return": round(float((bench_final_rets > 0.0).mean()), 4),
        "prob_drawdown_10pct": round(float((bench_max_dds <= -0.10).mean()), 4),
        "benchmark_p10_return_pct": round(float(np.percentile(bench_final_rets, 10)), 4),
        "benchmark_p50_return_pct": round(float(np.percentile(bench_final_rets, 50)), 4),
        "benchmark_p90_return_pct": round(float(np.percentile(bench_final_rets, 90)), 4),
    }
    return scenarios, mc_summary


def _run_forward_backtest(
    strategy_name: str,
    settings: Settings,
    data: Mapping[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    *,
    start_ts: pd.Timestamp,
    initial_equity: float,
    news_assessments: Mapping[str, NewsAssessment],
    initial_positions: Mapping[str, Position] | None = None,
    initial_cash: float | None = None,
) -> BacktestResult | None:
    """Simule une stratégie sur l'horizon futur via BacktestRunner (sans effet de bord)."""
    strategy = build_strategy(strategy_name, settings=settings)
    if getattr(strategy, "benchmark_symbol", None) is not None:
        _inject_benchmark(strategy, benchmark, settings.benchmark_symbol)

    runner = BacktestRunner(
        strategy,
        config=BacktestConfig(initial_equity=max(1.0, initial_equity)),
        risk_config=settings.risk,
    )

    news_cutoff = start_ts + timedelta(days=21)

    def _news_hook(sym: str, ts: pd.Timestamp) -> NewsAssessment | None:
        if ts <= news_cutoff:
            return news_assessments.get(sym)
        return None

    try:
        return runner.run(
            data,
            benchmark=benchmark,
            start_date=start_ts,
            news_hook=_news_hook if news_assessments else None,
            initial_positions=initial_positions,
            initial_cash=initial_cash,
        )
    except (BacktestError, StrategyError):
        return None


def _strategy_posture(strategy_name: str, market_outlook: Mapping[str, Any]) -> str:
    regime = market_outlook.get("regime", "haussier")
    if strategy_name == "shield_sword":
        if regime == "baissier":
            return "Bouclier défensif maximal (SPY < SMA 200)"
        return "70 % Bouclier + 30 % Top 3 offensif (rebalancement mensuel)"
    if strategy_name == "donchian_breakout":
        return "Achats sur cassure 20j (> SMA 200), sorties sur canal bas 10j"
    if strategy_name == "relative_momentum":
        return "Rotation sur les titres battant le benchmark (60j / 120j)"
    return "Suivi de tendance par croisements EMA 20/50 + filtre RSI"


def _upcoming_actions_for_strategy(
    strategy_name: str,
    *,
    settings: Settings,
    bars_by_symbol: Mapping[str, pd.DataFrame],
    benchmark_bars: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    central_result: BacktestResult | None,
    crash_result: BacktestResult | None,
    news_assessments: Mapping[str, NewsAssessment],
    current_positions: Mapping[str, Position],
) -> list[dict[str, Any]]:
    """Construit le calendrier lisible des actions prévues du bot."""
    actions: list[dict[str, Any]] = []
    engine = RiskEngine(settings.risk)
    last_hist_date = benchmark_bars.index[-1].date()
    horizon_end_date = future_dates[-1].date()
    strat_label = _STRATEGY_LABELS.get(strategy_name, strategy_name)

    strategy = build_strategy(strategy_name, settings=settings)
    if getattr(strategy, "benchmark_symbol", None) is not None:
        with contextlib.suppress(StrategyError):
            _inject_benchmark(strategy, benchmark_bars, settings.benchmark_symbol)
    if hasattr(strategy, "set_universe"):
        with contextlib.suppress(StrategyError):
            strategy.set_universe(bars_by_symbol)

    next_session_iso = future_dates[0].date().isoformat()
    for sym, bars in sorted(bars_by_symbol.items()):
        if len(bars) <= strategy.warmup_bars:
            continue
        try:
            sig = strategy.latest_signal(bars, symbol=sym)
        except StrategyError:
            continue
        news = news_assessments.get(sym)
        if sig.action is SignalType.BUY:
            if news is not None and engine.news_vetoes_entry(news):
                actions.append(
                    {
                        "date": next_session_iso,
                        "horizon": "Prochaine séance",
                        "strategy": strategy_name,
                        "strategy_label": strat_label,
                        "kind": "veto",
                        "symbol": sym,
                        "title": f"Achat bloqué par les actualités ({sym})",
                        "detail": (
                            f"Signal BUY technique mais refusé par le LLM "
                            f"({news.sentiment_score:+.2f}) : {news.rationale}"
                        ),
                    }
                )
            else:
                actions.append(
                    {
                        "date": next_session_iso,
                        "horizon": "Prochaine séance",
                        "strategy": strategy_name,
                        "strategy_label": strat_label,
                        "kind": "buy",
                        "symbol": sym,
                        "title": f"Achat prévu sur {sym} (~{sig.price:.2f} $)",
                        "detail": sig.reason,
                    }
                )
        elif sig.action is SignalType.SELL:
            actions.append(
                {
                    "date": next_session_iso,
                    "horizon": "Prochaine séance",
                    "strategy": strategy_name,
                    "strategy_label": strat_label,
                    "kind": "sell",
                    "symbol": sym,
                    "title": f"Vente / allègement prévu sur {sym}",
                    "detail": sig.reason,
                }
            )

    for sym, pos in sorted(current_positions.items()):
        news = news_assessments.get(sym)
        if (
            news is not None
            and news.is_critical_threat
            and news.sentiment_score <= settings.risk.news_panic_exit_threshold
        ):
            actions.append(
                {
                    "date": next_session_iso,
                    "horizon": "Immédiat",
                    "strategy": strategy_name,
                    "strategy_label": strat_label,
                    "kind": "panic",
                    "symbol": sym,
                    "title": f"Sortie d'urgence LLM sur {sym}",
                    "detail": f"Menace critique ({news.sentiment_score:+.2f}) : {news.rationale}",
                }
            )
        if not engine.is_defensive_symbol(sym):
            if pos.stop_price is not None and pos.current_price > 0:
                dist = (pos.current_price - pos.stop_price) / pos.current_price
                if dist <= 0.05:
                    actions.append(
                        {
                            "date": next_session_iso,
                            "horizon": "Surveillance proche",
                            "strategy": strategy_name,
                            "strategy_label": strat_label,
                            "kind": "stop",
                            "symbol": sym,
                            "title": f"Stop proche sur {sym} (filet à {pos.stop_price:.2f} $)",
                            "detail": f"Marge actuelle de {dist * 100:.1f} % avant déclenchement automatique.",
                        }
                    )
            if (
                not pos.partial_tp_taken
                and pos.unrealized_pnl_pct >= settings.risk.partial_take_profit_pct * 0.75
            ):
                target_tp = pos.avg_entry_price * (
                    1.0 + settings.risk.partial_take_profit_pct
                )
                actions.append(
                    {
                        "date": next_session_iso,
                        "horizon": "Seuil proche",
                        "strategy": strategy_name,
                        "strategy_label": strat_label,
                        "kind": "tp",
                        "symbol": sym,
                        "title": f"Prise de profit partielle +20 % proche sur {sym}",
                        "detail": (
                            f"Vente automatique de 50 % et stop à Break-Even si le prix "
                            f"atteint {target_tp:.2f} $ (gain latent actuel : {pos.unrealized_pnl_pct * 100:+.1f} %)."
                        ),
                    }
                )

    if strategy_name == "shield_sword" and isinstance(strategy, ShieldSwordStrategy):
        cursor = last_hist_date
        rebal_count = 0
        while rebal_count < 6:
            nxt = strategy.next_rebalance_date(cursor)
            if nxt > horizon_end_date:
                break
            days_away = max(1, (nxt - last_hist_date).days)
            actions.append(
                {
                    "date": nxt.isoformat(),
                    "horizon": _horizon_label(max(5, round(days_away * 5 / 7))),
                    "strategy": strategy_name,
                    "strategy_label": strat_label,
                    "kind": "rebalance",
                    "symbol": settings.risk.shield_symbol,
                    "title": "Rebalancement mensuel Bouclier + Épée",
                    "detail": (
                        f"Réévaluation du Top {strategy.params.sword_top_n} offensif "
                        f"({strategy.params.sword_allocation_pct * 100:.0f} %) et ajustement "
                        f"du Bouclier {settings.risk.shield_symbol} "
                        f"({strategy.params.shield_allocation_pct * 100:.0f} %). Bascule défensive "
                        f"maximale si {settings.benchmark_symbol} < SMA(200)."
                    ),
                }
            )
            cursor = nxt
            rebal_count += 1

    if central_result is not None:
        seen_keys: set[tuple[str, str, str]] = set()
        for trade in central_result.trades[:16]:
            entry_d = trade.entry_time.date()
            exit_d = trade.exit_time.date()
            days_in = max(1, (entry_d - last_hist_date).days)
            days_out = max(1, (exit_d - last_hist_date).days)
            if entry_d >= future_dates[0].date():
                key_in = ("buy", trade.symbol, entry_d.isoformat())
                if key_in not in seen_keys:
                    seen_keys.add(key_in)
                    actions.append(
                        {
                            "date": entry_d.isoformat(),
                            "horizon": _horizon_label(max(5, round(days_in * 5 / 7))),
                            "strategy": strategy_name,
                            "strategy_label": strat_label,
                            "kind": "buy",
                            "symbol": trade.symbol,
                            "title": f"Entrée projetée sur {trade.symbol} (~{trade.entry_price:.2f} $)",
                            "detail": (
                                f"Signal {strat_label} validé par le moteur de risque "
                                "sur la trajectoire médiane (P50)."
                            ),
                        }
                    )
            key_out = ("sell", trade.symbol, exit_d.isoformat())
            if key_out not in seen_keys:
                seen_keys.add(key_out)
                reason_txt = _EXIT_REASON_LABELS.get(trade.exit_reason, trade.exit_reason)
                actions.append(
                    {
                        "date": exit_d.isoformat(),
                        "horizon": _horizon_label(max(5, round(days_out * 5 / 7))),
                        "strategy": strategy_name,
                        "strategy_label": strat_label,
                        "kind": "tp" if trade.exit_reason == "partial_take_profit" else "sell",
                        "symbol": trade.symbol,
                        "title": f"Sortie projetée sur {trade.symbol} (~{trade.exit_price:.2f} $)",
                        "detail": f"{reason_txt} (variation simulée : {trade.return_pct * 100:+.1f} %).",
                    }
                )

    if crash_result is not None and crash_result.halts:
        first_halt = crash_result.halts[0]
        days_h = max(1, (first_halt.session_date - last_hist_date).days)
        actions.append(
            {
                "date": first_halt.session_date.isoformat(),
                "horizon": _horizon_label(max(5, round(days_h * 5 / 7))),
                "strategy": strategy_name,
                "strategy_label": strat_label,
                "kind": "halt",
                "symbol": "PORTFOLIO",
                "title": "En cas de krach : déclenchement du coupe-circuit",
                "detail": (
                    f"Protection activée ({first_halt.scope.value}) : {first_halt.reason}. "
                    "Les positions offensives sont liquidées pour préserver le capital."
                ),
            }
        )

    actions.sort(key=lambda item: (item["date"], item["symbol"]))
    return actions[:24]


def _build_symbol_forecasts(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    scenarios: Mapping[str, Mapping[str, Any]],
    future_dates: pd.DatetimeIndex,
    news_outlook: Mapping[str, Any],
    settings: Settings,
) -> list[dict[str, Any]]:
    """Synthèse prévisionnelle titre par titre (prix, LLM, seuils techniques)."""
    per_news = news_outlook.get("per_symbol", {})
    rows: list[dict[str, Any]] = []
    for sym, hist in sorted(bars_by_symbol.items()):
        if _is_defensive(sym, settings):
            continue
        last_price = float(hist["close"].iloc[-1])
        c_close = float(scenarios["central"]["data"][sym]["close"].iloc[-1])
        b_close = float(scenarios["bull"]["data"][sym]["close"].iloc[-1])
        k_close = float(scenarios["crash"]["data"][sym]["close"].iloc[-1])

        high20 = float(hist["high"].tail(20).max())
        low10 = float(hist["low"].tail(10).min())
        sma50 = float(hist["close"].tail(min(50, len(hist))).mean())
        sma200 = float(hist["close"].tail(min(200, len(hist))).mean())
        try:
            atr_val = float(compute_atr(hist, length=14).iloc[-1])
            if not math.isfinite(atr_val):
                atr_val = last_price * 0.02
        except Exception:  # noqa: BLE001
            atr_val = last_price * 0.02

        z_fast, _ = momentum_zscores(hist["close"])
        mom_val = (
            float(z_fast.iloc[-1])
            if not z_fast.empty and math.isfinite(float(z_fast.iloc[-1]))
            else 0.0
        )

        news_info = per_news.get(sym)
        if last_price >= high20 * 0.995:
            trigger_hint = f"Proche cassure haussière > {high20:.2f} $"
        elif last_price <= low10 * 1.01:
            trigger_hint = f"Sous vigilance sortie < {low10:.2f} $"
        else:
            trigger_hint = f"Achat Donchian > {high20:.2f} $ · Sortie < {low10:.2f} $"

        rows.append(
            {
                "symbol": sym,
                "current_price": round(last_price, 2),
                "central_price": round(c_close, 2),
                "central_change_pct": round(c_close / last_price - 1.0, 4),
                "bull_price": round(b_close, 2),
                "bull_change_pct": round(b_close / last_price - 1.0, 4),
                "crash_price": round(k_close, 2),
                "crash_change_pct": round(k_close / last_price - 1.0, 4),
                "high_20d": round(high20, 2),
                "low_10d": round(low10, 2),
                "sma_50": round(sma50, 2),
                "sma_200": round(sma200, 2),
                "atr": round(atr_val, 2),
                "momentum_z": round(mom_val, 2),
                "trigger_hint": trigger_hint,
                "news": news_info,
                "horizon_date": future_dates[-1].date().isoformat(),
            }
        )
    return rows


def _extract_timeline_and_milestones(
    *,
    future_dates: pd.DatetimeIndex,
    initial_equity: float,
    central_res: BacktestResult | None,
    bull_res: BacktestResult | None,
    crash_res: BacktestResult | None,
    bench_central: pd.DataFrame,
    bench_crash: pd.DataFrame,
    primary_strategy: str,
    market_outlook: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Construit la courbe temporelle future et les jalons (semaines / mois)."""
    horizon = len(future_dates)
    bench_start = (
        float(bench_central["close"].iloc[-horizon - 1])
        if len(bench_central) > horizon
        else float(bench_central["close"].iloc[0])
    )

    def _series_map(res: BacktestResult | None) -> dict[pd.Timestamp, float]:
        if res is None or res.equity_curve.empty:
            return {}
        return {
            pd.Timestamp(idx): float(val) for idx, val in res.equity_curve["equity"].items()
        }

    c_map = _series_map(central_res)
    b_map = _series_map(bull_res)
    k_map = _series_map(crash_res)

    points: list[dict[str, Any]] = []
    last_c = initial_equity
    last_b = initial_equity
    last_k = initial_equity

    for i, ts in enumerate(future_dates):
        last_c = c_map.get(ts, last_c)
        last_b = b_map.get(ts, last_b)
        last_k = k_map.get(ts, last_k)
        bc_val = (
            float(bench_central["close"].loc[ts])
            if ts in bench_central.index
            else bench_start
        )
        bk_val = (
            float(bench_crash["close"].loc[ts])
            if ts in bench_crash.index
            else bench_start
        )
        points.append(
            {
                "date": ts.date().isoformat(),
                "day_index": i + 1,
                "central_equity": round(last_c, 2),
                "bull_equity": round(last_b, 2),
                "crash_equity": round(last_k, 2),
                "benchmark_equity": round(
                    initial_equity * (bc_val / max(bench_start, 0.01)), 2
                ),
                "benchmark_crash_equity": round(
                    initial_equity * (bk_val / max(bench_start, 0.01)), 2
                ),
            }
        )

    candidate_offsets = [10, 21, 42, 63, 126, 252]
    labels_by_day = {
        10: "Dans 2 semaines",
        21: "Dans 1 mois",
        42: "Dans 2 mois",
        63: "Dans 3 mois",
        126: "Dans 6 mois",
        252: "Dans 12 mois",
    }
    selected_days = [d for d in candidate_offsets if d <= horizon]
    if (not selected_days or selected_days[-1] != horizon) and horizon not in selected_days:
        selected_days.append(horizon)

    milestones: list[dict[str, Any]] = []
    posture = _strategy_posture(primary_strategy, market_outlook)
    for d in selected_days:
        pt = points[d - 1]
        c_ret = pt["central_equity"] / initial_equity - 1.0
        b_ret = pt["bull_equity"] / initial_equity - 1.0
        k_ret = pt["crash_equity"] / initial_equity - 1.0
        milestones.append(
            {
                "day_index": d,
                "label": labels_by_day.get(d, _horizon_label(d)),
                "date": pt["date"],
                "central_equity": pt["central_equity"],
                "central_return_pct": round(c_ret, 4),
                "bull_equity": pt["bull_equity"],
                "bull_return_pct": round(b_ret, 4),
                "crash_equity": pt["crash_equity"],
                "crash_return_pct": round(k_ret, 4),
                "benchmark_equity": pt["benchmark_equity"],
                "posture": posture,
            }
        )

    return points, milestones


def build_market_prediction(
    settings: Settings,
    *,
    bars_by_symbol: Mapping[str, pd.DataFrame],
    benchmark_bars: pd.DataFrame | None = None,
    benchmark_long_bars: pd.DataFrame | None = None,
    news_assessments: Mapping[str, NewsAssessment] | None = None,
    current_positions: Mapping[str, Position] | None = None,
    initial_equity: float = 100_000.0,
    initial_cash: float | None = None,
    primary_strategy: str = "shield_sword",
    horizon_days: int = 63,
    now: datetime | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """Génère une prédiction complète (informative) sur plusieurs semaines/mois."""
    if primary_strategy not in STRATEGY_NAMES:
        raise ValueError(f"Stratégie inconnue : {primary_strategy}")
    horizon = max(10, min(252, int(horizon_days)))
    equity_base = float(initial_equity) if initial_equity > 0 else 100_000.0
    moment = now or datetime.now(UTC)

    clean_bars: dict[str, pd.DataFrame] = {}
    for sym, df in bars_by_symbol.items():
        if df is not None and not df.empty:
            clean_bars[sym] = _ensure_utc_index(df)
    if not clean_bars:
        raise ValueError("Aucune donnée historique disponible pour calculer la prédiction.")

    if benchmark_bars is not None and not benchmark_bars.empty:
        bench_clean = _ensure_utc_index(benchmark_bars)
    else:
        synth = _build_synthetic_benchmark(clean_bars, settings)
        if synth is None or synth.empty:
            raise ValueError(
                "Impossible de construire la référence de marché pour la prédiction."
            )
        bench_clean = synth

    bench_long_clean = (
        _ensure_utc_index(benchmark_long_bars)
        if benchmark_long_bars is not None and not benchmark_long_bars.empty
        else None
    )

    assessments = dict(news_assessments or {})
    positions = dict(current_positions or {})
    warnings = list(errors or [])

    tradable = [sym for sym in clean_bars if sym != settings.benchmark_symbol]
    news_outlook = _summarize_news_impact(settings, tradable, assessments)
    market_outlook = _analyze_past_market(
        bench_clean, news_outlook, settings, benchmark_long_bars=bench_long_clean
    )
    crash_risk_score = float(market_outlook["crash_risk"]["score"])

    last_ts = max(
        bench_clean.index[-1],
        *(df.index[-1] for df in clean_bars.values()),
    )
    future_dates = _future_business_dates(pd.Timestamp(last_ts), horizon)
    first_future_ts = future_dates[0]

    scenarios, mc_summary = _project_future_bars_monte_carlo(
        clean_bars,
        bench_clean,
        future_dates,
        news_outlook=news_outlook,
        crash_risk_score=crash_risk_score,
        settings=settings,
        benchmark_long_bars=bench_long_clean,
    )

    effective_positions = {
        sym: pos for sym, pos in positions.items() if pos.qty > 0 and sym in clean_bars
    }
    uses_live_portfolio = bool(effective_positions)
    effective_cash = initial_cash if uses_live_portfolio else None

    strategies_comparison: list[dict[str, Any]] = []
    actions_by_strategy: dict[str, list[dict[str, Any]]] = {}
    primary_results: dict[str, BacktestResult | None] = {
        "central": None,
        "bull": None,
        "crash": None,
    }

    for strat in STRATEGY_NAMES:
        res_central = _run_forward_backtest(
            strat,
            settings,
            scenarios["central"]["data"],
            scenarios["central"]["benchmark"],
            start_ts=first_future_ts,
            initial_equity=equity_base,
            news_assessments=assessments,
            initial_positions=effective_positions or None,
            initial_cash=effective_cash,
        )
        res_bull = _run_forward_backtest(
            strat,
            settings,
            scenarios["bull"]["data"],
            scenarios["bull"]["benchmark"],
            start_ts=first_future_ts,
            initial_equity=equity_base,
            news_assessments=assessments,
            initial_positions=effective_positions or None,
            initial_cash=effective_cash,
        )
        res_crash = _run_forward_backtest(
            strat,
            settings,
            scenarios["crash"]["data"],
            scenarios["crash"]["benchmark"],
            start_ts=first_future_ts,
            initial_equity=equity_base,
            news_assessments=assessments,
            initial_positions=effective_positions or None,
            initial_cash=effective_cash,
        )
        if strat == primary_strategy:
            primary_results = {
                "central": res_central,
                "bull": res_bull,
                "crash": res_crash,
            }

        actions = _upcoming_actions_for_strategy(
            strat,
            settings=settings,
            bars_by_symbol=clean_bars,
            benchmark_bars=bench_clean,
            future_dates=future_dates,
            central_result=res_central,
            crash_result=res_crash,
            news_assessments=assessments,
            current_positions=positions,
        )
        actions_by_strategy[strat] = actions

        if res_central is not None:
            c_ret = float(res_central.metrics.get("total_return") or 0.0)
            c_eq = float(res_central.metrics.get("final_equity") or equity_base)
            exposure = float(res_central.metrics.get("exposure_pct") or 0.0)
        else:
            c_ret, c_eq, exposure = 0.0, equity_base, 0.0
            warnings.append(
                f"Historique insuffisant pour simuler complètement « {_STRATEGY_LABELS.get(strat, strat)} »."
            )

        b_ret = (
            float(res_bull.metrics.get("total_return") or 0.0)
            if res_bull is not None
            else c_ret
        )
        b_eq = (
            float(res_bull.metrics.get("final_equity") or equity_base)
            if res_bull is not None
            else c_eq
        )
        k_ret = (
            float(res_crash.metrics.get("total_return") or 0.0)
            if res_crash is not None
            else c_ret
        )
        k_eq = (
            float(res_crash.metrics.get("final_equity") or equity_base)
            if res_crash is not None
            else c_eq
        )
        k_dd = (
            float(res_crash.metrics.get("max_drawdown") or 0.0)
            if res_crash is not None
            else 0.0
        )

        strategies_comparison.append(
            {
                "strategy": strat,
                "label": _STRATEGY_LABELS.get(strat, strat),
                "simulated": res_central is not None,
                "central_return_pct": round(c_ret, 4),
                "central_final_equity": round(c_eq, 2),
                "bull_return_pct": round(b_ret, 4),
                "bull_final_equity": round(b_eq, 2),
                "crash_return_pct": round(k_ret, 4),
                "crash_final_equity": round(k_eq, 2),
                "crash_max_drawdown_pct": round(k_dd, 4),
                "exposure_pct": round(exposure, 4),
                "planned_actions_count": len(actions),
                "posture": _strategy_posture(strat, market_outlook),
            }
        )

    effective_initial = (
        float(primary_results["central"].initial_equity)
        if primary_results["central"] is not None
        else equity_base
    )

    timeline_points, milestones = _extract_timeline_and_milestones(
        future_dates=future_dates,
        initial_equity=effective_initial,
        central_res=primary_results["central"],
        bull_res=primary_results["bull"],
        crash_res=primary_results["crash"],
        bench_central=scenarios["central"]["benchmark"],
        bench_crash=scenarios["crash"]["benchmark"],
        primary_strategy=primary_strategy,
        market_outlook=market_outlook,
    )

    symbol_forecasts = _build_symbol_forecasts(
        clean_bars,
        scenarios,
        future_dates,
        news_outlook,
        settings,
    )

    primary_actions = actions_by_strategy.get(primary_strategy, [])
    other_actions = [
        act
        for strat, list_act in actions_by_strategy.items()
        if strat != primary_strategy
        for act in list_act[:3]
    ]
    predicted_actions = sorted(
        [*primary_actions, *other_actions],
        key=lambda item: (item["date"], item["strategy"]),
    )

    primary_row = next(
        (row for row in strategies_comparison if row["strategy"] == primary_strategy),
        strategies_comparison[0],
    )
    portfolio_origin = (
        "depuis vos positions actuelles"
        if uses_live_portfolio
        else "sur capital de référence"
    )
    market_summary = (
        f"Régime {market_outlook['regime']} sur {settings.benchmark_symbol} "
        f"({market_outlook['crash_risk']['label'].lower()}). "
        f"Sur {horizon} séances (~{round(horizon / 21.0)} mois, {portfolio_origin}), "
        f"« {primary_row['label']} » projette "
        f"{primary_row['central_return_pct'] * 100:+.1f} % (médiane P50) "
        f"et {primary_row['crash_return_pct'] * 100:+.1f} % en scénario de krach."
    )
    market_outlook["summary"] = market_summary

    return {
        "generated_at": moment.isoformat(),
        "primary_strategy": primary_strategy,
        "horizon_days": horizon,
        "horizon_weeks": round(horizon / 5.0, 1),
        "horizon_months": round(horizon / 21.0, 1),
        "initial_equity": round(effective_initial, 2),
        "uses_live_portfolio": uses_live_portfolio,
        "disclaimer": _DISCLAIMER,
        "monte_carlo_summary": mc_summary,
        "market_outlook": market_outlook,
        "news_outlook": news_outlook,
        "historical_context": {
            "observed_events": market_outlook["observed_events"],
            "crash_episodes": market_outlook["observed_events"].get("crash_episodes", []),
            "crash_analogues": market_outlook["crash_analogues"],
            "closest_analogue": market_outlook["closest_analogue"],
        },
        "timeline_points": timeline_points,
        "milestones": milestones,
        "strategies_comparison": strategies_comparison,
        "predicted_actions": predicted_actions,
        "symbol_forecasts": symbol_forecasts,
        "errors": list(dict.fromkeys(warnings)),
    }
