"""Configuration typée du bot de trading.

Toutes les valeurs sont surchargeables par variables d'environnement ou par un
fichier ``.env`` (voir ``.env.example``). Les objets de configuration sont figés
(``frozen=True``) : une configuration invalide échoue au démarrage, jamais en
cours de session de trading.

Sécurité : l'URL de trading est validée à la construction. Seul l'endpoint
officiel de paper trading Alpaca (``paper-api.alpaca.markets``) est accepté ;
toute tentative de pointer vers l'API live lève ``ConfigurationError``.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import re
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
PAPER_HOST: str = "paper-api.alpaca.markets"
LIVE_HOST: str = "api.alpaca.markets"
PAPER_ENDPOINT: str = f"https://{PAPER_HOST}"
DATA_ENDPOINT: str = "https://data.alpaca.markets"

Timeframe = Literal["1Min", "5Min", "15Min", "30Min", "1Hour", "1Day"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
WebhookFormat = Literal["discord", "telegram", "generic"]

_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


class ConfigurationError(RuntimeError):
    """Configuration absente, incohérente ou dangereuse."""


def is_paper_endpoint(url: str) -> bool:
    """Indique si ``url`` pointe vers l'endpoint Alpaca paper officiel.

    Args:
        url: URL de l'API de trading à valider.

    Returns:
        ``True`` uniquement si le hostname correspond à l'endpoint paper.
    """
    parsed = urlparse(url)
    return parsed.scheme == "https" and (parsed.hostname or "").lower() == PAPER_HOST


class RiskConfig(BaseModel):
    """Paramètres du moteur de risque, exprimés en fractions du capital.

    Toutes les tailles de position sont dérivées de ces paramètres. Aucune
    stratégie ne peut les contourner : ``RiskEngine`` est le seul composant
    autorisé à produire un ordre.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    risk_per_trade_pct: float = Field(
        0.01,
        gt=0.0,
        le=0.05,
        description="Fraction de l'équity risquée si le stop est touché (1 % = 0.01).",
    )
    max_position_pct: float = Field(
        0.20,
        gt=0.0,
        le=1.0,
        description="Poids maximal d'une position dans l'équity.",
    )
    max_gross_exposure_pct: float = Field(
        0.80,
        gt=0.0,
        le=1.0,
        description="Exposition brute maximale (somme des valeurs de position / équity).",
    )
    bull_gross_exposure_pct: float = Field(
        0.90,
        gt=0.0,
        le=1.0,
        description=(
            "Exposition brute maximale en régime haussier (benchmark au-dessus "
            "de sa SMA longue)."
        ),
    )
    bull_max_position_pct: float = Field(
        0.30,
        gt=0.0,
        le=1.0,
        description="Poids maximal d'une position en régime haussier.",
    )
    max_open_positions: int = Field(5, ge=1, le=50)
    atr_stop_multiple: float = Field(
        2.0,
        gt=0.0,
        le=10.0,
        description="Distance du stop initial en multiples d'ATR.",
    )
    trailing_atr_multiple: float = Field(
        3.0,
        ge=0.0,
        le=10.0,
        description="Distance du trailing stop en multiples d'ATR (0 désactive le trailing).",
    )
    trailing_activation_r: float = Field(
        1.0,
        ge=0.0,
        le=5.0,
        description="Gain en R nécessaire avant d'activer le trailing stop.",
    )
    max_daily_loss_pct: float = Field(
        0.03,
        gt=0.0,
        le=0.20,
        description="Perte journalière déclenchant le circuit breaker (arrêt jusqu'au lendemain).",
    )
    max_drawdown_pct: float = Field(
        0.15,
        gt=0.0,
        le=0.50,
        description="Drawdown depuis le plus haut d'équity déclenchant l'arrêt définitif.",
    )
    min_order_notional: float = Field(10.0, ge=0.0, le=10_000.0)
    fractional_shares: bool = Field(
        True,
        description="Autorise les fractions d'action (Alpaca paper).",
    )
    qty_decimals: int = Field(4, ge=0, le=8)
    order_ttl_seconds: int = Field(
        30,
        ge=1,
        le=300,
        description="Durée de validité du jeton d'approbation d'un ordre.",
    )
    news_veto_threshold: float = Field(
        -0.60,
        ge=-1.0,
        le=0.0,
        description=(
            "Score de sentiment au plus bas duquel un achat est refusé "
            "([NEWS_VETO]). 0.0 désactive de fait tout veto par score."
        ),
    )
    news_panic_exit_threshold: float = Field(
        -0.85,
        ge=-1.0,
        le=0.0,
        description=(
            "Score au plus bas duquel une menace critique liquide la position "
            "en urgence (news_panic_exit)."
        ),
    )
    safe_haven_enabled: bool = Field(
        True,
        description="Active la poche de repli défensive automatique (Safe Haven / Cash Sweep).",
    )
    safe_haven_symbols: list[str] = Field(
        default_factory=lambda: ["BRK.B", "WMT", "KO"],
        description="Symboles de la poche de repli.",
    )
    cash_buffer_pct: float = Field(
        0.10,
        ge=0.0,
        le=1.0,
        description="Fraction minimale d'équité conservée en cash pur (10 % = 0.10).",
    )
    max_safe_haven_total_pct: float = Field(
        0.50,
        ge=0.0,
        le=1.0,
        description="Plafond total d'exposition sur la poche de repli (50 % = 0.50).",
    )
    regime_sma_period: int = Field(
        200,
        ge=2,
        le=1000,
        description="Période de la moyenne mobile sur SPY pour la détection du régime.",
    )
    withdrawable_cash_floor_usd: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Montant minimal en dollars sanctuarisé sur le compte en cash pur, "
            "jamais réinvesti ni dans l'univers ni dans la poche de repli."
        ),
    )
    equity_hard_floor_usd: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Limite basse absolue du portefeuille total : si l'équité passe sous "
            "ce seuil, liquidation intégrale et arrêt de tout nouvel achat."
        ),
    )
    partial_take_profit_pct: float = Field(
        0.20,
        ge=0.0,
        le=10.0,
        description=(
            "Seuil de plus-value latente par action déclenchant une prise de profit "
            "partielle (ex: +20 % = 0.20)."
        ),
    )
    partial_take_profit_sell_ratio: float = Field(
        0.50,
        gt=0.0,
        le=1.0,
        description=(
            "Proportion de la position vendue lors d'une prise de profit partielle "
            "(ex: 50 % = 0.50)."
        ),
    )
    max_single_position_weight: float = Field(
        0.25,
        gt=0.0,
        le=1.0,
        description=(
            "Poids maximal d'une position dans l'équité totale avant écrémage "
            "automatique du surplus (ex: 25 % = 0.25)."
        ),
    )
    shield_symbol: str = Field(
        "SGOV",
        description=(
            "Actif du Bouclier (ETF de bons du Trésor court terme). Seul le "
            "moteur de risque le sait : c'est lui qui décide qu'un actif est "
            "défensif (pas de stop exigé, pas de trailing, pas d'écrémage)."
        ),
    )

    @field_validator("shield_symbol")
    @classmethod
    def _validate_shield_symbol(cls, value: str) -> str:
        """Normalise le symbole du Bouclier et rejette un ticker invalide."""
        normalized = value.strip().upper()
        if not _SYMBOL_RE.match(normalized):
            raise ValueError(f"symbole de Bouclier invalide: {value!r}")
        return normalized

    @model_validator(mode="after")
    def _validate_coherence(self) -> RiskConfig:
        """Vérifie les invariants croisés du profil de risque."""
        if self.safe_haven_enabled and not self.safe_haven_symbols:
            raise ValueError(
                "safe_haven_symbols ne peut pas être vide si safe_haven_enabled est activé"
            )
        if self.cash_buffer_pct + self.max_safe_haven_total_pct > 1.0:
            raise ValueError(
                "cash_buffer_pct + max_safe_haven_total_pct ne peut pas dépasser 1.0"
            )
        if self.news_panic_exit_threshold > self.news_veto_threshold:
            raise ValueError(
                "news_panic_exit_threshold doit être inférieur ou égal à "
                "news_veto_threshold (une panique doit être plus grave qu'un veto)"
            )
        if self.max_position_pct > self.max_gross_exposure_pct:
            raise ValueError("max_position_pct ne peut pas dépasser max_gross_exposure_pct")
        if self.bull_max_position_pct > self.bull_gross_exposure_pct:
            raise ValueError(
                "bull_max_position_pct ne peut pas dépasser bull_gross_exposure_pct"
            )
        if self.max_daily_loss_pct >= self.max_drawdown_pct:
            raise ValueError(
                "max_daily_loss_pct doit être strictement inférieur à "
                "max_drawdown_pct (sinon le circuit breaker journalier masque "
                "l'arrêt définitif)"
            )
        if not self.fractional_shares and self.qty_decimals != 0:
            raise ValueError(
                "qty_decimals doit valoir 0 lorsque fractional_shares est désactivé"
            )
        return self


class BacktestConfig(BaseModel):
    """Hypothèses d'exécution du moteur de backtest déterministe."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_equity: float = Field(100_000.0, gt=0.0)
    commission_pct: float = Field(
        0.001,
        ge=0.0,
        le=0.01,
        description="Commission par side, en fraction du notionnel (0.001 = 0,1 %).",
    )
    slippage_pct: float = Field(
        0.001,
        ge=0.0,
        le=0.01,
        description="Slippage par side, en fraction du prix de référence (0.001 = 0,1 %).",
    )
    annualization_factor: int = Field(252, ge=1, le=365)
    risk_free_rate: float = Field(0.0, ge=-0.5, le=1.0)
    execution_delay_bars: int = Field(
        1,
        ge=1,
        le=5,
        description="Nombre de barres entre le signal (clôture t) et l'exécution (ouverture t+n).",
    )
    market_timezone: str = "America/New_York"
    cash_yield_pct: float = Field(
        0.0,
        ge=0.0,
        le=0.20,
        description=(
            "Rendement annuel du cash en backtest. 0 par défaut : le compte "
            "paper Alpaca ne rémunère pas les liquidités, et un backtest qui "
            "les rémunère surestime la performance. Pour modéliser un placement "
            "monétaire, préférez un vrai support (SHIELD_SYMBOL), qui a un prix."
        ),
    )


class Settings(BaseSettings):
    """Configuration globale, chargée depuis l'environnement et ``.env``."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    alpaca_api_key: SecretStr | None = None
    alpaca_secret_key: SecretStr | None = None
    alpaca_trading_url: str = PAPER_ENDPOINT
    alpaca_data_url: str = DATA_ENDPOINT
    data_feed: Literal["iex", "sip", "otc", "delayed_sip"] = "iex"

    symbols: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "NVDA",
            "AMD",
            "AVGO",
            "PLTR",
            "MSFT",
            "AMZN",
            "META",
            "XOM",
            "NEE",
            "LLY",
            "JPM",
            "COST",
        ],
        description="Univers de trading (CSV en variable d'environnement).",
    )
    benchmark_symbol: str = Field(
        "SPY",
        description=(
            "Benchmark du régime de marché et de la force relative (jamais tradé)."
        ),
    )
    safe_haven_enabled: bool = Field(
        True,
        description="Active la poche de repli défensive automatique (Safe Haven / Cash Sweep).",
    )
    safe_haven_symbols: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["BRK.B", "WMT", "KO"],
        description="Poche de repli défensive automatique (exclue des offensifs).",
    )
    timeframe: Timeframe = "1Day"
    lookback_days: int = Field(400, ge=60, le=3650)
    poll_interval_seconds: int = Field(60, ge=5, le=3600)
    market_timezone: str = "America/New_York"
    dry_run: bool = Field(
        False,
        description="Analyse et journalise les décisions sans transmettre d'ordre.",
    )

    state_dir: Path = PROJECT_ROOT / "state"
    log_level: LogLevel = "INFO"
    log_file: Path | None = None

    webhook_url: SecretStr | None = Field(
        None,
        description=(
            "Webhook de notification (Discord ou Telegram) : récapitulatif JSON "
            "envoyé à la fin de chaque cycle exécuté. Absent = aucune "
            "notification."
        ),
    )
    webhook_format: WebhookFormat = Field(
        "discord",
        description="Dialecte du webhook : discord, telegram ou generic.",
    )
    webhook_chat_id: str | None = Field(
        None,
        description="Identifiant de conversation Telegram (obligatoire pour ce format).",
    )
    webhook_token: SecretStr | None = Field(
        None,
        description=(
            "Jeton transmis en en-tête « Authorization: Bearer » au webhook. "
            "Indispensable dès que l'URL de réception est publique : sans lui, "
            "n'importe qui connaissant l'adresse peut injecter de fausses stats."
        ),
    )
    webhook_timeout_seconds: float = Field(10.0, ge=1.0, le=60.0)

    stats_github_repo: str | None = Field(
        None,
        description=(
            "Dépôt GitHub où publier les statistiques, au format « compte/nom ». "
            "Laisser vide désactive la publication. Une URL de pages est alors "
            "servie depuis le dossier docs/ de la branche principale."
        ),
    )
    stats_github_token: SecretStr | None = Field(
        None,
        description=(
            "Jeton GitHub autorisé à écrire le contenu du dépôt. Obligatoire "
            "dès que STATS_GITHUB_REPO est renseigné."
        ),
    )
    stats_github_path: str = "docs/data/stats.json"
    stats_github_branch: str = "main"

    # --- Analyse de sentiment des actualités (Alpaca News + OpenRouter) ---
    news_filter_enabled: bool = Field(
        True,
        description=(
            "Active le filtre de sentiment à l'entrée. Désactivé, le bot trade "
            "exactement comme avant, sans aucun appel réseau supplémentaire."
        ),
    )
    openrouter_api_key: SecretStr | None = Field(
        None,
        description=(
            "Clé OpenRouter servant à noter les actualités. Absente, le filtre "
            "reste inerte : aucun achat n'est refusé."
        ),
    )
    openrouter_model: str = Field(
        "google/gemini-3.8-flash",
        description=("Modèle OpenRouter pour l'analyse de sentiment des actualités."),
    )
    news_lookback_hours: int = Field(
        24, ge=1, le=168, description="Fenêtre d'actualités analysée, en heures."
    )
    news_max_articles: int = Field(
        10,
        ge=1,
        le=50,
        description=(
            "Nombre maximal d'articles transmis au LLM par symbole et par appel : "
            "c'est le principal levier de coût après le cache d'articles."
        ),
    )
    news_timeout_seconds: float = Field(
        5.0,
        ge=1.0,
        le=60.0,
        description=("Délai maximal d'un appel OpenRouter en secondes (défaut 5s)."),
    )

    # --- Stratégie Bouclier + Épée (Shield + Sword / Core-Satellite CPPI) ---
    shield_allocation_pct: float = Field(
        0.70,
        ge=0.0,
        le=1.0,
        description="Fraction du portefeuille allouée au Bouclier (défensif/cash).",
    )
    sword_allocation_pct: float = Field(
        0.30,
        ge=0.0,
        le=1.0,
        description="Fraction du portefeuille allouée à l'Épée (Top N offensif).",
    )
    sword_top_n: int = Field(
        3,
        ge=1,
        le=20,
        description="Nombre maximal d'actions offensives détenues dans l'Épée.",
    )
    sword_rebalance_days: int = Field(
        20,
        ge=1,
        le=252,
        description="Périodicité de rebalancement de l'Épée en jours de bourse.",
    )
    sword_trend_sma: int = Field(
        100,
        ge=10,
        le=500,
        description="Période de la moyenne mobile de tendance (SMA) pour l'Épée.",
    )
    sword_stop_atr_multiplier: float = Field(
        5.0,
        ge=1.0,
        le=20.0,
        description="Multiple d'ATR pour le stop catastrophe large de l'Épée.",
    )
    sword_max_loss_pct: float = Field(
        0.12,
        ge=0.01,
        le=0.50,
        description="Perte maximale par ligne de l'Épée (12 % = 0.12).",
    )
    shield_symbol: str | None = Field(
        None,
        description=(
            "Actif prioritaire du Bouclier (ETF Bons du Trésor US court terme). "
            "Surcharge plate de RiskConfig.shield_symbol."
        ),
    )

    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    # Surcharges "plates" du profil de risque (plus lisibles en .env).
    risk_per_trade_pct: float | None = None
    max_position_pct: float | None = None
    max_gross_exposure_pct: float | None = None
    bull_gross_exposure_pct: float | None = None
    bull_max_position_pct: float | None = None
    max_open_positions: int | None = None
    atr_stop_multiple: float | None = None
    trailing_atr_multiple: float | None = None
    trailing_activation_r: float | None = None
    max_daily_loss_pct: float | None = None
    max_drawdown_pct: float | None = None
    min_order_notional: float | None = None
    fractional_shares: bool | None = None
    news_veto_threshold: float | None = None
    news_panic_exit_threshold: float | None = None
    cash_buffer_pct: float | None = None
    max_safe_haven_total_pct: float | None = None
    regime_sma_period: int | None = None
    withdrawable_cash_floor_usd: float | None = None
    equity_hard_floor_usd: float | None = None
    partial_take_profit_pct: float | None = None
    partial_take_profit_sell_ratio: float | None = None
    max_single_position_weight: float | None = None

    @field_validator("symbols", mode="before")
    @classmethod
    def _parse_symbols(cls, value: object) -> object:
        """Accepte ``"AAPL,MSFT"`` en plus d'une liste JSON."""
        if isinstance(value, str):
            raw = value.replace(";", ",").split(",")
            return [item.strip().upper() for item in raw if item.strip()]
        return value

    @field_validator("symbols")
    @classmethod
    def _validate_symbols(cls, values: list[str]) -> list[str]:
        """Normalise l'univers et rejette les tickers invalides."""
        seen: list[str] = []
        for symbol in values:
            normalized = symbol.strip().upper()
            if not _SYMBOL_RE.match(normalized):
                raise ValueError(f"symbole invalide: {symbol!r}")
            if normalized not in seen:
                seen.append(normalized)
        if not seen:
            raise ValueError("l'univers de trading ne peut pas être vide")
        return seen

    @field_validator("safe_haven_symbols", mode="before")
    @classmethod
    def _parse_safe_haven_symbols(cls, value: object) -> object:
        """Accepte ``"BRK.B,WMT,KO"`` en plus d'une liste JSON."""
        if isinstance(value, str):
            raw = value.replace(";", ",").split(",")
            return [item.strip().upper() for item in raw if item.strip()]
        return value

    @field_validator("safe_haven_symbols")
    @classmethod
    def _validate_safe_haven_symbols(cls, values: list[str]) -> list[str]:
        """Normalise la poche de repli et rejette les tickers invalides."""
        seen: list[str] = []
        for symbol in values:
            normalized = symbol.strip().upper()
            if not _SYMBOL_RE.match(normalized):
                raise ValueError(f"symbole safe haven invalide: {symbol!r}")
            if normalized not in seen:
                seen.append(normalized)
        if not seen:
            raise ValueError("safe_haven_symbols ne peut pas être vide")
        return seen

    @field_validator("benchmark_symbol")
    @classmethod
    def _validate_benchmark_symbol(cls, value: str) -> str:
        """Normalise le benchmark et rejette un ticker invalide."""
        normalized = value.strip().upper()
        if not _SYMBOL_RE.match(normalized):
            raise ValueError(f"symbole de benchmark invalide: {value!r}")
        return normalized

    @field_validator("alpaca_trading_url")
    @classmethod
    def _validate_paper_url(cls, value: str) -> str:
        """Refuse toute URL qui ne serait pas le paper endpoint officiel."""
        normalized = value.rstrip("/")
        if not is_paper_endpoint(normalized):
            raise ValueError(
                "Le bot est volontairement limité au paper trading Alpaca : "
                f"utilisez {PAPER_ENDPOINT} (reçu: {value!r})."
            )
        return normalized

    @field_validator("state_dir", "log_file", mode="after")
    @classmethod
    def _resolve_paths(cls, value: Path | None) -> Path | None:
        """Résout les chemins relatifs depuis la racine du projet."""
        if value is None:
            return None
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @model_validator(mode="after")
    def _validate_runtime(self) -> Settings:
        """Contrôles de cohérence globaux et application des surcharges risque."""
        if self.lookback_days < 60:
            raise ValueError("lookback_days trop court pour les indicateurs")

        # Le benchmark est une référence, jamais un actif : s'il traîne dans
        # l'univers ou dans les safe havens, il est retiré automatiquement.
        if self.benchmark_symbol in self.symbols:
            self.symbols = [
                symbol for symbol in self.symbols if symbol != self.benchmark_symbol
            ]
        if self.benchmark_symbol in self.safe_haven_symbols:
            self.safe_haven_symbols = [
                symbol
                for symbol in self.safe_haven_symbols
                if symbol != self.benchmark_symbol
            ]

        # Les symboles safe haven sont exclus automatiquement de l'univers offensif
        safe_set = set(self.safe_haven_symbols)
        self.symbols = [symbol for symbol in self.symbols if symbol not in safe_set]

        if not self.symbols:
            raise ValueError(
                "l'univers de trading serait vide une fois le benchmark et les "
                "safe havens retirés : ajoutez au moins un autre symbole"
            )

        risk_fields = (
            "risk_per_trade_pct",
            "max_position_pct",
            "max_gross_exposure_pct",
            "bull_gross_exposure_pct",
            "bull_max_position_pct",
            "max_open_positions",
            "atr_stop_multiple",
            "trailing_atr_multiple",
            "trailing_activation_r",
            "max_daily_loss_pct",
            "max_drawdown_pct",
            "min_order_notional",
            "fractional_shares",
            "news_veto_threshold",
            "news_panic_exit_threshold",
            "safe_haven_enabled",
            "safe_haven_symbols",
            "cash_buffer_pct",
            "max_safe_haven_total_pct",
            "regime_sma_period",
            "withdrawable_cash_floor_usd",
            "equity_hard_floor_usd",
            "partial_take_profit_pct",
            "partial_take_profit_sell_ratio",
            "max_single_position_weight",
            "shield_symbol",
        )
        overrides = {
            name: getattr(self, name)
            for name in risk_fields
            if name in self.model_fields_set and getattr(self, name) is not None
        }
        if overrides:
            merged = {**self.risk.model_dump(), **overrides}
            self.risk = RiskConfig(**merged)
        return self

    @property
    def is_paper(self) -> bool:
        """``True`` si l'endpoint configuré est bien celui du paper trading."""
        return is_paper_endpoint(self.alpaca_trading_url)

    def require_credentials(self) -> tuple[str, str]:
        """Retourne ``(api_key, secret_key)`` ou échoue explicitement.

        Raises:
            ConfigurationError: si l'une des deux clés est absente.
        """
        if self.alpaca_api_key is None or self.alpaca_secret_key is None:
            raise ConfigurationError(
                "ALPACA_API_KEY et ALPACA_SECRET_KEY sont requis pour le paper "
                "trading. Renseignez-les dans .env (voir .env.example)."
            )
        api_key = self.alpaca_api_key.get_secret_value().strip()
        secret = self.alpaca_secret_key.get_secret_value().strip()
        if not api_key or not secret:
            raise ConfigurationError("Les clés Alpaca ne peuvent pas être vides.")
        return api_key, secret

    def redacted_summary(self) -> dict[str, str]:
        """Résumé journalisable, sans aucun secret."""
        return {
            "alpaca_trading_url": self.alpaca_trading_url,
            "data_feed": self.data_feed,
            "symbols": ",".join(self.symbols),
            "timeframe": self.timeframe,
            "dry_run": str(self.dry_run),
            "state_dir": str(self.state_dir),
            "risk_per_trade_pct": str(self.risk.risk_per_trade_pct),
            "max_daily_loss_pct": str(self.risk.max_daily_loss_pct),
        }


def configure_logging(settings: Settings) -> logging.Logger:
    """Configure le logging racine (console + fichier rotatif optionnel).

    Args:
        settings: Configuration contenant le niveau et le fichier de log.

    Returns:
        Le logger racine ``trading_bot``.
    """
    logger = logging.getLogger("trading_bot")
    logger.setLevel(settings.log_level)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if settings.log_file is not None:
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            settings.log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def update_env_file(env_path: Path, updates: dict[str, str]) -> None:
    """Mise à jour robuste du .env : évite les doublons et préserve le formatage."""
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []

    for key, value in updates.items():
        # Un retour à la ligne dans une valeur permettrait d'injecter une clé
        # arbitraire dans le .env : on refuse.
        if any(c in f"{key}{value}" for c in ("\n", "\r", "\0")):
            raise ValueError(f"valeur invalide pour {key} : caractère de contrôle interdit")
        target = f"{key}={value}" if value else f"# {key}="
        found = False
        for i, line in enumerate(lines):
            if re.match(rf"^#*\s*{re.escape(key)}=.*$", line):
                lines[i] = target
                found = True
                break

        if not found:
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(target)

    env_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = env_path.with_suffix(env_path.suffix + ".tmp")
    # Le .env contient les clés Alpaca : lecture/écriture réservées au propriétaire.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.chmod(tmp, 0o600)
    tmp.replace(env_path)
