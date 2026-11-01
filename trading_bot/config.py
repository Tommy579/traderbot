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

    @model_validator(mode="after")
    def _validate_coherence(self) -> RiskConfig:
        """Vérifie les invariants croisés du profil de risque."""
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
        description="Modèle OpenRouter, retenu pour son coût et sa latence.",
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
        5.0, ge=1.0, le=30.0, description="Délai maximal d'un appel OpenRouter."
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
        # l'univers (faute de frappe, copier-coller), le bot achèterait sa
        # propre référence et le panier doublerait l'exposition au marché.
        if self.benchmark_symbol in self.symbols:
            self.symbols = [
                symbol for symbol in self.symbols if symbol != self.benchmark_symbol
            ]
        if not self.symbols:
            raise ValueError(
                "l'univers de trading serait vide une fois le benchmark "
                f"{self.benchmark_symbol} retiré : ajoutez au moins un autre symbole"
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
        )
        overrides = {
            name: getattr(self, name)
            for name in risk_fields
            if getattr(self, name) is not None
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
