"""Moteur de risque : dimensionnement, stops et circuit breakers.

C'est le **seul** composant autorisé à transformer un :class:`Signal` en ordre.
La stratégie produit une intention d'analyse ; le moteur de risque décide si elle
est admissible, la dimensionne, l'ancre sur un stop ATR et émet un
:class:`OrderIntent` scellé par un :class:`ApprovalToken`.

Garanties structurelles :

* aucun ordre ne peut être construit sans jeton émis par ce module (le jeton
  porte une référence privée au module, impossible à forger depuis l'extérieur) ;
* toute modification de quantité, de sens ou de symbole après approbation casse
  l'invariant vérifié dans ``OrderIntent.__post_init__`` ;
* les sorties (SELL) restent autorisées quand le circuit breaker est déclenché,
  afin de pouvoir liquider le portefeuille.

Le moteur est sans effet de bord externe : ni horloge système, ni I/O, ni
courtier. Le temps et la date de séance sont injectés, ce qui rend le backtest
strictement reproductible.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Final, Protocol

from trading_bot.config import RiskConfig

_TOKEN_ISSUER: Final[object] = object()
_STOP_TICK: Final[float] = 0.01
#: Écart minimal (en fraction de l'équité) justifiant un réajustement du Bouclier.
#: Sans ce seuil, la poche serait complétée de quelques dollars à chaque séance.
_SHIELD_MIN_TOPUP_PCT: Final[float] = 0.005
_QTY_EPSILON: Final[float] = 1e-9


class RiskError(RuntimeError):
    """Violation d'un invariant du moteur de risque."""


class Side(StrEnum):
    """Sens d'un ordre."""

    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    """Type d'ordre accepté par la couche d'exécution."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class HaltScope(StrEnum):
    """Portée d'un arrêt déclenché par un circuit breaker."""

    DAY = "day"
    PERMANENT = "permanent"


class RejectionCode(StrEnum):
    """Motif machine-lisible d'un refus du moteur de risque."""

    APPROVED = "APPROVED"
    HALTED = "HALTED"
    INVALID_REQUEST = "INVALID_REQUEST"
    POSITION_ALREADY_OPEN = "POSITION_ALREADY_OPEN"
    MAX_POSITIONS_REACHED = "MAX_POSITIONS_REACHED"
    GROSS_EXPOSURE_LIMIT = "GROSS_EXPOSURE_LIMIT"
    INSUFFICIENT_BUYING_POWER = "INSUFFICIENT_BUYING_POWER"
    BELOW_MIN_NOTIONAL = "BELOW_MIN_NOTIONAL"
    INVALID_STOP = "INVALID_STOP"
    NEWS_VETO = "NEWS_VETO"
    BEAR_REGIME_VETO = "BEAR_REGIME_VETO"
    EQUITY_FLOOR_BREACHED = "EQUITY_FLOOR_BREACHED"


class NewsSignal(Protocol):
    """Vue minimale d'une note d'actualités attendue par le moteur.

    Le moteur de risque n'importe pas le module d'analyse : il ne connaît
    qu'un score et un booléen. Cela le garde sans dépendance réseau — donc
    testable hors ligne et reproductible en backtest — sans créer de cycle
    d'imports entre ``risk`` et ``data``.

    Les membres sont déclarés en lecture seule : une note d'actualités est un
    instantané immuable (:class:`~trading_bot.data.news_sentiment.NewsAssessment`
    est une dataclass gelée), et un attribut muable ne serait pas satisfait
    par un champ gelé.
    """

    @property
    def sentiment_score(self) -> float: ...

    @property
    def is_critical_threat(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class ApprovalToken:
    """Jeton d'approbation d'ordre, émis exclusivement par :class:`RiskEngine`.

    Le champ ``issuer`` référence un objet privé du module. Un jeton construit
    ailleurs — même avec les bons champs — est rejeté par :meth:`is_valid`.
    """

    decision_id: str
    symbol: str
    side: Side
    qty: float
    issued_at: datetime
    expires_at: datetime
    issuer: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.decision_id:
            raise ValueError("jeton sans identifiant de décision")
        if not self.symbol:
            raise ValueError("jeton sans symbole")
        if not math.isfinite(self.qty) or self.qty <= 0:
            raise ValueError(f"quantité de jeton invalide: {self.qty!r}")
        if self.expires_at <= self.issued_at:
            raise ValueError("expiration du jeton antérieure à son émission")

    def is_valid(self, now: datetime) -> bool:
        """Vérifie l'origine, la fenêtre de validité et la quantité."""
        return (
            self.issuer is _TOKEN_ISSUER
            and self.issued_at <= now < self.expires_at
            and math.isfinite(self.qty)
            and self.qty > 0.0
        )


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """Ordre approuvé et scellé, seule structure acceptée par l'exécution.

    Toute incohérence entre le jeton et les paramètres de l'ordre lève
    ``ValueError`` à la construction : il est impossible d'augmenter la quantité
    après validation du risque.
    """

    symbol: str
    side: Side
    qty: float
    token: ApprovalToken
    order_type: OrderType = OrderType.MARKET
    stop_price: float | None = None
    limit_price: float | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.symbol != self.token.symbol:
            raise ValueError("symbole incohérent avec le jeton d'approbation")
        if self.side is not self.token.side:
            raise ValueError("sens incohérent avec le jeton d'approbation")
        if not math.isclose(self.qty, self.token.qty, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("quantité incohérente avec le jeton d'approbation")
        if not self.token.is_valid(self.token.issued_at):
            raise ValueError("jeton non émis par le RiskEngine")
        if not math.isfinite(self.qty) or self.qty <= 0:
            raise ValueError(f"quantité d'ordre invalide: {self.qty!r}")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("un ordre LIMIT exige limit_price")
        if self.order_type is OrderType.STOP and self.stop_price is None:
            raise ValueError("un ordre STOP exige stop_price")
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError("stop_price doit être strictement positif")
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("limit_price doit être strictement positif")

    @property
    def client_order_id(self) -> str:
        """Identifiant idempotent dérivé de la décision (max 34 caractères)."""
        return f"tb{self.token.decision_id}"

    def is_executable(self, now: datetime) -> bool:
        """``True`` si le jeton est encore valide à l'instant ``now``."""
        return self.token.is_valid(now)


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Décision d'audit du moteur de risque (acceptée ou refusée)."""

    decision_id: str
    created_at: datetime
    approved: bool
    code: RejectionCode
    reason: str
    intent: OrderIntent | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.approved != (self.intent is not None):
            raise ValueError("une décision approuvée doit porter exactement un ordre")
        if self.approved and self.code is not RejectionCode.APPROVED:
            raise ValueError("code incohérent avec une décision approuvée")
        if not self.approved and self.code is RejectionCode.APPROVED:
            raise ValueError("un refus ne peut pas porter le code APPROVED")


@dataclass(slots=True)
class Position:
    """Position ouverte, avec ses métadonnées de protection."""

    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    stop_price: float | None = None
    initial_stop_price: float | None = None
    highest_price: float = 0.0
    opened_at: datetime | None = None
    partial_tp_taken: bool = False

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("position sans symbole")
        if not math.isfinite(self.qty) or self.qty <= 0:
            raise ValueError(f"quantité de position invalide: {self.qty!r}")
        for name in ("avg_entry_price", "current_price"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} invalide: {value!r}")
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError("stop_price de position invalide")
        if self.highest_price <= 0:
            self.highest_price = max(self.avg_entry_price, self.current_price)

    @property
    def market_value(self) -> float:
        """Valeur de marché au dernier prix connu."""
        return self.qty * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """P&L latent en dollars."""
        return (self.current_price - self.avg_entry_price) * self.qty

    @property
    def unrealized_pnl_pct(self) -> float:
        """P&L latent rapporté au coût d'entrée."""
        cost = self.avg_entry_price * self.qty
        return 0.0 if cost <= 0 else self.unrealized_pnl / cost

    @property
    def unrealized_plpc(self) -> float:
        """Alias pour le P&L latent en pourcentage."""
        return self.unrealized_pnl_pct

    @property
    def risk_per_share(self) -> float | None:
        """Risque initial par action (entrée - stop initial), si défini."""
        if self.initial_stop_price is None:
            return None
        distance = self.avg_entry_price - self.initial_stop_price
        return distance if distance > 0 else None

    @property
    def r_multiple(self) -> float:
        """Gain courant exprimé en multiples du risque initial."""
        risk = self.risk_per_share
        if risk is None:
            return 0.0
        return (self.current_price - self.avg_entry_price) / risk

    def stop_state(self) -> dict[str, object]:
        """État minimal à persister pour restaurer la protection après restart."""
        return {
            "stop_price": self.stop_price,
            "initial_stop_price": self.initial_stop_price,
            "highest_price": self.highest_price,
            "partial_tp_taken": self.partial_tp_taken,
        }

    def apply_stop_state(self, state: Mapping[str, object]) -> None:
        """Restaure un état de stop précédemment persisté."""
        stop = state.get("stop_price")
        initial = state.get("initial_stop_price")
        highest = state.get("highest_price")
        partial_tp = state.get("partial_tp_taken")
        if isinstance(stop, (int, float)) and stop > 0:
            self.stop_price = float(stop)
        if isinstance(initial, (int, float)) and initial > 0:
            self.initial_stop_price = float(initial)
        if isinstance(highest, (int, float)) and highest > 0:
            self.highest_price = max(float(highest), self.highest_price)
        if isinstance(partial_tp, bool):
            self.partial_tp_taken = partial_tp


@dataclass(slots=True)
class AccountState:
    """Photographie du compte utilisée par les décisions de risque."""

    equity: float
    cash: float
    buying_power: float
    positions: dict[str, Position] = field(default_factory=dict)
    day_start_equity: float | None = None
    peak_equity: float | None = None
    realized_pnl_today: float = 0.0
    trading_day: date | None = None
    halted: bool = False
    halt_reason: str | None = None
    halt_scope: HaltScope | None = None

    @property
    def gross_exposure(self) -> float:
        """Somme des valeurs de marché des positions."""
        return sum(position.market_value for position in self.positions.values())

    @property
    def gross_exposure_pct(self) -> float:
        """Exposition brute rapportée à l'équity."""
        return 0.0 if self.equity <= 0 else self.gross_exposure / self.equity

    @property
    def daily_pnl_pct(self) -> float:
        """Performance depuis l'équity de début de séance."""
        if not self.day_start_equity or self.day_start_equity <= 0:
            return 0.0
        return self.equity / self.day_start_equity - 1.0

    @property
    def drawdown_pct(self) -> float:
        """Drawdown courant depuis le plus haut d'équity."""
        if not self.peak_equity or self.peak_equity <= 0:
            return 0.0
        return self.equity / self.peak_equity - 1.0

    def active_positions(self) -> list[Position]:
        """Positions effectivement ouvertes (quantité strictement positive)."""
        return [position for position in self.positions.values() if position.qty > 0]


@dataclass(frozen=True, slots=True)
class HaltEvent:
    """Événement de circuit breaker, journalisé par la couche d'orchestration."""

    scope: HaltScope
    reason: str
    session_date: date
    equity: float
    daily_pnl_pct: float
    drawdown_pct: float


@dataclass(frozen=True, slots=True)
class TrailingStopUpdate:
    """Relèvement de stop proposé par le moteur (à appliquer par l'appelant)."""

    symbol: str
    previous_stop: float
    new_stop: float
    price: float
    atr: float
    r_multiple: float


class RiskEngine:
    """Applique le profil de risque et scelle les ordres approuvés.

    Args:
        config: Profil de risque. Par défaut, les valeurs prudentes de
            :class:`RiskConfig`.
        audit_limit: Taille maximale du journal de décisions conservé en mémoire.
    """

    def __init__(
        self,
        config: RiskConfig | None = None,
        *,
        audit_limit: int = 1_000,
    ) -> None:
        if audit_limit < 1:
            raise ValueError("audit_limit doit être >= 1")
        self._config = config or RiskConfig()
        self._audit: deque[RiskDecision] = deque(maxlen=audit_limit)
        self._halts: list[HaltEvent] = []
        self._resumptions: list[str] = []
        self._sequence = 0
        self._bull_market = False
        self._bear_market = False

    @property
    def config(self) -> RiskConfig:
        """Profil de risque effectif."""
        return self._config

    @property
    def bull_market(self) -> bool:
        """Régime de marché courant (benchmark au-dessus de sa SMA longue)."""
        return self._bull_market

    @property
    def bear_market(self) -> bool:
        """Régime de marché baissier (benchmark en-dessous de sa SMA longue)."""
        return self._bear_market

    def is_defensive_symbol(self, symbol: str) -> bool:
        """``True`` si l'actif appartient au Bouclier (repli ou cash rémunéré).

        Le symbole du Bouclier vient de la configuration : le moteur ne doit pas
        figer « SGOV » dans son code, sinon un ``SHIELD_SYMBOL`` personnalisé
        resterait traité comme une action offensive (stop, trailing, écrémage).
        """
        return (
            symbol == self._config.shield_symbol
            or symbol in self._config.safe_haven_symbols
        )

    def set_market_regime(
        self,
        *,
        bull_market: bool,
        bear_market: bool = False,
    ) -> None:
        """Fixe le régime de marché pour la suite des décisions.

        Appelé à chaque cycle par l'orchestration, à partir de la clôture
        du benchmark par rapport à sa SMA (par défaut 200).
        Si le benchmark est sous sa SMA, bear_market est True.
        """
        self._bull_market = bool(bull_market)
        self._bear_market = bool(bear_market)

    def effective_position_cap_pct(self) -> float:
        """Plafond de poids de position selon le régime courant."""
        if self._bull_market:
            return self._config.bull_max_position_pct
        return self._config.max_position_pct

    def effective_gross_cap_pct(self) -> float:
        """Plafond d'exposition brute selon le régime courant."""
        if self._bull_market:
            return self._config.bull_gross_exposure_pct
        return self._config.max_gross_exposure_pct

    def rank_candidates(self, scores: Mapping[str, float]) -> list[str]:
        """Classe les candidats à l'entrée par z-score de momentum décroissant.

        Le budget de liquidités est consommé dans cet ordre : les actifs les
        plus vigoureux sont servis en premier, les suivants ne reçoivent que
        le reliquat. Les scores non finis sont relégués en fin de classement
        (tri par symbole pour la reproductibilité), jamais rejetés : c'est le
        rôle des plafonds, pas du classement.
        """
        ranked = sorted(
            scores,
            key=lambda symbol: (
                not math.isfinite(scores[symbol]),
                -scores[symbol] if math.isfinite(scores[symbol]) else 0.0,
                symbol,
            ),
        )
        return ranked

    @property
    def audit(self) -> tuple[RiskDecision, ...]:
        """Journal ordonné des décisions (borné par ``audit_limit``)."""
        return tuple(self._audit)

    @property
    def halt_events(self) -> tuple[HaltEvent, ...]:
        """Circuit breakers déclenchés depuis la création du moteur."""
        return tuple(self._halts)

    def start_session(self, account: AccountState, *, session_date: date) -> bool:
        """Initialise (ou réinitialise) l'état de séance.

        Un changement de date réarme un arrêt de portée ``DAY`` mais jamais un
        arrêt ``PERMANENT`` (drawdown maximal), qui exige une intervention
        humaine explicite via :meth:`reset_permanent_halt`.

        Returns:
            ``True`` si une nouvelle séance a été ouverte.
        """
        if account.equity < 0 or not math.isfinite(account.equity):
            raise RiskError(f"équity invalide: {account.equity!r}")

        if account.day_start_equity is None or account.trading_day != session_date:
            account.trading_day = session_date
            account.day_start_equity = account.equity
            account.realized_pnl_today = 0.0
            if account.halt_scope is HaltScope.DAY:
                account.halted = False
                account.halt_reason = None
                account.halt_scope = None
            if account.peak_equity is None:
                account.peak_equity = account.equity
            return True

        if account.peak_equity is None:
            account.peak_equity = account.equity
        return False

    def update_risk_state(
        self,
        account: AccountState,
        *,
        session_date: date,
    ) -> HaltEvent | None:
        """Met à jour le pic d'équity puis évalue les circuit breakers.

        Returns:
            L'événement de halt s'il vient d'être déclenché, sinon ``None``.
        """
        self.start_session(account, session_date=session_date)
        account.peak_equity = max(account.peak_equity or account.equity, account.equity)

        if account.halted:
            return None
        if account.equity <= 0:
            return self._halt(
                account,
                session_date=session_date,
                scope=HaltScope.PERMANENT,
                reason="équity nulle ou négative : trading interrompu",
            )

        if (
            self._config.equity_hard_floor_usd > 0.0
            and account.equity <= self._config.equity_hard_floor_usd
        ):
            return self._halt(
                account,
                session_date=session_date,
                scope=HaltScope.PERMANENT,
                reason=(
                    f"équité {account.equity:.2f} <= plancher de sécurité "
                    f"{self._config.equity_hard_floor_usd:.2f} [EQUITY_FLOOR_BREACHED]"
                ),
            )

        if account.drawdown_pct <= -self._config.max_drawdown_pct:
            return self._halt(
                account,
                session_date=session_date,
                scope=HaltScope.PERMANENT,
                reason=(
                    f"drawdown {account.drawdown_pct:.2%} au-delà de la limite "
                    f"{-self._config.max_drawdown_pct:.2%}"
                ),
            )
        if account.daily_pnl_pct <= -self._config.max_daily_loss_pct:
            return self._halt(
                account,
                session_date=session_date,
                scope=HaltScope.DAY,
                reason=(
                    f"perte journalière {account.daily_pnl_pct:.2%} au-delà de la "
                    f"limite {-self._config.max_daily_loss_pct:.2%}"
                ),
            )
        return None

    def mark_to_market(
        self,
        account: AccountState,
        prices: Mapping[str, float],
    ) -> None:
        """Met à jour les prix courants et le plus haut atteint par position."""
        for symbol, position in account.positions.items():
            price = prices.get(symbol)
            if price is None or not math.isfinite(price) or price <= 0:
                continue
            position.current_price = float(price)
            position.highest_price = max(position.highest_price, position.current_price)

    def evaluate_entry(
        self,
        *,
        symbol: str,
        price: float,
        atr: float,
        account: AccountState,
        now: datetime,
        session_date: date,
        news: NewsSignal | None = None,
        target_weight: float | None = None,
        atr_stop_multiple: float | None = None,
        max_loss_pct: float | None = None,
        is_shield: bool = False,
    ) -> RiskDecision:
        """Valide et dimensionne une entrée longue.

        Dimensionnement : la quantité est le minimum de

        * ``équity * risk_per_trade_pct / (atr * atr_stop_multiple)`` — risque
          fixe si le stop est touché ;
        * ``capital_engagé_max / price`` — plafonds de position, d'exposition
          brute et de liquidités disponibles.

        Une note d'actualités franchement négative (``news``) bloque l'entrée
        avec le code :attr:`RejectionCode.NEWS_VETO`. Le paramètre est
        optionnel : appelé sans lui, le moteur se comporte exactement comme
        avant, ce qui garantit la reproductibilité du backtest.

        Args:
            symbol: Ticker à acheter.
            price: Prix de référence de la décision (dernier prix connu).
            atr: ATR de la barre de décision, strictement positif.
            account: État du compte (équity, cash, positions).
            now: Horodatage de la décision (timezone-aware).
            session_date: Date de séance pour les compteurs journaliers.
            news: Note d'actualités du symbole, si le filtre est actif.

        Returns:
            Décision approuvée portant un :class:`OrderIntent`, ou refus motivé.
        """
        self._require_aware(now)
        self.update_risk_state(account, session_date=session_date)

        base_metrics: dict[str, float] = {
            "price": float(price) if math.isfinite(price) else 0.0,
            "atr": float(atr) if math.isfinite(atr) else 0.0,
            "equity": account.equity,
            "bull_market": float(self._bull_market),
            "position_cap": self.effective_position_cap_pct(),
            "gross_cap": self.effective_gross_cap_pct(),
        }
        if not symbol:
            return self._reject(
                RejectionCode.INVALID_REQUEST, "symbole vide", now, base_metrics
            )
        if not math.isfinite(price) or price <= 0:
            return self._reject(
                RejectionCode.INVALID_REQUEST, "prix non fini ou négatif", now, base_metrics
            )
        if not math.isfinite(atr) or atr <= 0:
            return self._reject(
                RejectionCode.INVALID_REQUEST, "ATR non fini ou négatif", now, base_metrics
            )
        if (
            self._config.equity_hard_floor_usd > 0.0
            and account.equity <= self._config.equity_hard_floor_usd
        ):
            return self._reject(
                RejectionCode.EQUITY_FLOOR_BREACHED,
                (
                    f"[EQUITY_FLOOR_BREACHED] équité {account.equity:.2f} <= "
                    f"plancher de sécurité {self._config.equity_hard_floor_usd:.2f}"
                ),
                now,
                base_metrics,
            )
        if account.halted:
            return self._reject(
                RejectionCode.HALTED,
                account.halt_reason or "circuit breaker actif",
                now,
                base_metrics,
            )
        if (
            self._bear_market
            and symbol not in self._config.safe_haven_symbols
            and not is_shield
        ):
            return self._reject(
                RejectionCode.BEAR_REGIME_VETO,
                (
                    f"[BEAR_REGIME_VETO] marché baissier ({self._config.regime_sma_period} SMA) : "
                    f"entrée offensive interdite sur {symbol}"
                ),
                now,
                base_metrics,
            )
        if news is not None and self.news_vetoes_entry(news):
            threat = " (menace critique signalée)" if news.is_critical_threat else ""
            return self._reject(
                RejectionCode.NEWS_VETO,
                (
                    f"actualité défavorable sur {symbol} : score "
                    f"{news.sentiment_score:+.2f} ≤ seuil "
                    f"{self._config.news_veto_threshold:+.2f}{threat}"
                ),
                now,
                {
                    **base_metrics,
                    "news_score": float(news.sentiment_score),
                    "news_critical": float(news.is_critical_threat),
                },
            )
        existing = account.positions.get(symbol)
        if existing is not None and existing.qty > 0:
            return self._reject(
                RejectionCode.POSITION_ALREADY_OPEN,
                f"position déjà ouverte sur {symbol}",
                now,
                base_metrics,
            )
        offensive_positions = [
            p
            for sym, p in account.positions.items()
            if p.qty > 0 and not self.is_defensive_symbol(sym)
        ]
        if not is_shield and len(offensive_positions) >= self._config.max_open_positions:
            return self._reject(
                RejectionCode.MAX_POSITIONS_REACHED,
                f"maximum de {self._config.max_open_positions} positions atteint",
                now,
                base_metrics,
            )

        if is_shield:
            stop_price = None
            stop_distance = 0.0
        else:
            mult = (
                atr_stop_multiple
                if atr_stop_multiple is not None
                else self._config.atr_stop_multiple
            )
            stop_distance = atr * mult
            stop_by_atr = price - stop_distance
            if max_loss_pct is not None:
                stop_by_loss = price * (1.0 - max_loss_pct)
                stop_price = round(max(stop_by_atr, stop_by_loss), 2)
            else:
                stop_price = round(stop_by_atr, 2)
            if stop_price <= 0:
                return self._reject(
                    RejectionCode.INVALID_STOP,
                    f"stop ATR non positif ({stop_price:.2f})",
                    now,
                    {**base_metrics, "stop_distance": stop_distance},
                )

        gross_remaining = max(
            0.0,
            account.equity * self.effective_gross_cap_pct() - account.gross_exposure,
        )
        cash_buffer = (
            account.equity * self._config.cash_buffer_pct
            if (self._config.safe_haven_enabled and not is_shield)
            else 0.0
        )
        protected_cash = max(cash_buffer, self._config.withdrawable_cash_floor_usd)
        cash_available = max(0.0, min(account.cash - protected_cash, account.buying_power))
        if cash_available <= 0.0:
            return self._reject(
                RejectionCode.INSUFFICIENT_BUYING_POWER,
                "aucun pouvoir d'achat disponible",
                now,
                base_metrics,
            )
        if gross_remaining <= 0.0:
            return self._reject(
                RejectionCode.GROSS_EXPOSURE_LIMIT,
                (
                    f"exposition brute {account.gross_exposure_pct:.1%} au plafond "
                    f"{self.effective_gross_cap_pct():.1%}"
                ),
                now,
                base_metrics,
            )

        if target_weight is not None:
            notional_cap = min(
                account.equity * target_weight,
                gross_remaining,
                cash_available,
            )
            raw_qty = notional_cap / price
            qty_by_risk = raw_qty
            qty_by_notional = raw_qty
            risk_budget = raw_qty * stop_distance
            binding = "target_weight"
        else:
            risk_budget = account.equity * self._config.risk_per_trade_pct
            qty_by_risk = risk_budget / stop_distance
            notional_cap = min(
                account.equity * self.effective_position_cap_pct(),
                gross_remaining,
                cash_available,
            )
            qty_by_notional = notional_cap / price
            raw_qty = min(qty_by_risk, qty_by_notional)
            binding = "risk" if qty_by_risk <= qty_by_notional else "notional"
        qty = self._round_qty(raw_qty)
        binding = "risk" if qty_by_risk <= qty_by_notional else "notional"

        metrics: dict[str, float] = {
            **base_metrics,
            "stop_distance": stop_distance,
            "stop_price": float(stop_price) if stop_price is not None else 0.0,
            "risk_budget": risk_budget,
            "qty_by_risk": qty_by_risk,
            "qty_by_notional": qty_by_notional,
            "notional_cap": notional_cap,
            "qty": qty,
            "notional": qty * price,
            "risk_amount": qty * stop_distance,
        }

        if qty <= 0.0:
            return self._reject(
                RejectionCode.BELOW_MIN_NOTIONAL,
                "quantité arrondie à zéro (capital ou risque insuffisant)",
                now,
                metrics,
            )
        notional = qty * price
        if notional < self._config.min_order_notional:
            return self._reject(
                RejectionCode.BELOW_MIN_NOTIONAL,
                (
                    f"notionnel {notional:.2f} < minimum "
                    f"{self._config.min_order_notional:.2f}"
                ),
                now,
                metrics,
            )
        if qty * stop_distance > risk_budget * (1.0 + 1e-9):
            raise RiskError(
                "le dimensionnement dépasse le budget de risque : invariant violé"
            )

        token = self._mint_token(
            symbol=symbol,
            side=Side.BUY,
            qty=qty,
            now=now,
        )
        intent = OrderIntent(
            symbol=symbol,
            side=Side.BUY,
            qty=qty,
            token=token,
            order_type=OrderType.MARKET,
            stop_price=stop_price,
            reason=f"entrée {binding}",
        )
        return self._approve(intent, now, metrics)

    def evaluate_exit(
        self,
        *,
        position: Position,
        price: float,
        now: datetime,
        reason: str,
        qty: float | None = None,
    ) -> RiskDecision:
        """Approuve la liquidation complète ou partielle d'une position.

        Une sortie reste possible lorsque le circuit breaker est actif : c'est
        précisément le mécanisme de mise à plat du portefeuille.

        Raises:
            RiskError: si la position est vide ou si le prix est invalide.
        """
        self._require_aware(now)
        if position.qty <= 0:
            raise RiskError(f"position {position.symbol} sans quantité à vendre")
        if not math.isfinite(price) or price <= 0:
            raise RiskError(f"prix de sortie invalide pour {position.symbol}: {price!r}")

        sell_qty = position.qty if qty is None else min(position.qty, self._round_qty(qty))
        if sell_qty <= 0:
            raise RiskError(f"quantité à vendre nulle pour {position.symbol}")

        token = self._mint_token(
            symbol=position.symbol,
            side=Side.SELL,
            qty=sell_qty,
            now=now,
        )
        intent = OrderIntent(
            symbol=position.symbol,
            side=Side.SELL,
            qty=sell_qty,
            token=token,
            order_type=OrderType.MARKET,
            reason=reason,
        )
        metrics = {
            "price": price,
            "qty": sell_qty,
            "notional": sell_qty * price,
            "unrealized_pnl": (price - position.avg_entry_price) * sell_qty,
            "r_multiple": position.r_multiple,
        }
        return self._approve(intent, now, metrics)

    def evaluate_safe_haven_unwind(
        self,
        account: AccountState,
        needed_cash: float,
        prices: Mapping[str, float],
        now: datetime,
        *,
        reason: str = "safe_haven_unwind",
    ) -> list[RiskDecision]:
        """Génère les ordres de vente nécessaires sur la poche safe haven pour libérer du cash.

        Seul le montant nécessaire (needed_cash) est vendu, dans la limite des
        positions de repli disponibles.
        """
        self._require_aware(now)
        if not self._config.safe_haven_enabled or needed_cash <= 0:
            return []

        decisions: list[RiskDecision] = []
        remaining_shortfall = needed_cash

        # Parcourt les positions de repli dans un ordre déterministe
        safe_positions = [
            p
            for sym, p in sorted(account.positions.items())
            if sym in self._config.safe_haven_symbols and p.qty > 0
        ]

        for pos in safe_positions:
            if remaining_shortfall <= 0:
                break
            price = prices.get(pos.symbol, pos.current_price)
            if price <= 0 or not math.isfinite(price):
                continue

            pos_value = pos.qty * price
            if pos_value <= 0:
                continue

            if pos_value <= remaining_shortfall:
                sell_qty = pos.qty
            else:
                raw_qty = remaining_shortfall / price
                sell_qty = self._round_qty(raw_qty)
                if sell_qty <= 0:
                    continue
                sell_qty = min(pos.qty, sell_qty)

            if sell_qty <= 0:
                continue

            decision = self.evaluate_exit(
                position=pos,
                price=price,
                now=now,
                reason=reason,
                qty=sell_qty,
            )
            decisions.append(decision)
            remaining_shortfall -= sell_qty * price

        return decisions

    def evaluate_safe_haven_cash_floor_reconstitution(
        self,
        account: AccountState,
        prices: Mapping[str, float],
        now: datetime,
    ) -> list[RiskDecision]:
        """Si le cash réel est inférieur au plancher de retrait, vend du safe haven pour le reconstituer."""
        self._require_aware(now)
        floor = self._config.withdrawable_cash_floor_usd
        if floor <= 0.0 or account.cash >= floor:
            return []
        needed_cash = floor - account.cash
        return self.evaluate_safe_haven_unwind(
            account,
            needed_cash=needed_cash,
            prices=prices,
            now=now,
            reason="safe_haven_cash_floor_reconstitution",
        )

    def evaluate_cash_sweep(
        self,
        account: AccountState,
        prices: Mapping[str, float],
        now: datetime,
        *,
        news: Mapping[str, NewsSignal] | None = None,
        atrs: Mapping[str, float] | None = None,
    ) -> list[RiskDecision]:
        """Alloue l'excédent de cash au-dessus du buffer vers les symboles safe haven.

        Se déclenche à la fin de chaque cycle horaire si safe_haven_enabled est vrai.
        """
        self._require_aware(now)
        if not self._config.safe_haven_enabled or not self._config.safe_haven_symbols:
            return []
        if account.halted:
            return []

        cash_buffer = account.equity * self._config.cash_buffer_pct
        protected_cash = max(cash_buffer, self._config.withdrawable_cash_floor_usd)
        excess_cash = max(0.0, account.cash - protected_cash)
        if excess_cash < self._config.min_order_notional:
            return []

        current_safe_haven_value = sum(
            p.market_value
            for sym, p in account.positions.items()
            if sym in self._config.safe_haven_symbols
            and sym != self._config.shield_symbol
            and p.qty > 0
        )
        max_safe_haven_value = account.equity * self._config.max_safe_haven_total_pct
        room_safe_haven = max(0.0, max_safe_haven_value - current_safe_haven_value)
        if room_safe_haven < self._config.min_order_notional:
            return []

        gross_cap = account.equity * self.effective_gross_cap_pct()
        gross_remaining = max(0.0, gross_cap - account.gross_exposure)
        if gross_remaining < self._config.min_order_notional:
            return []

        sweep_budget = min(
            excess_cash, room_safe_haven, gross_remaining, account.buying_power
        )
        if sweep_budget < self._config.min_order_notional:
            return []

        eligible_symbols: list[str] = []
        news_map = news or {}
        for sym in self._config.safe_haven_symbols:
            if sym == self._config.shield_symbol:
                # Le Bouclier a son propre financement, avec son propre poids
                # cible : le compter deux fois doublerait la poche défensive.
                continue
            price = prices.get(sym, 0.0)
            if not math.isfinite(price) or price <= 0:
                continue
            sym_news = news_map.get(sym)
            if sym_news is not None and self.news_vetoes_entry(sym_news):
                continue
            eligible_symbols.append(sym)

        if not eligible_symbols:
            return []

        allocation_per_symbol = sweep_budget / len(eligible_symbols)
        decisions: list[RiskDecision] = []
        atrs_map = atrs or {}

        for sym in eligible_symbols:
            price = prices[sym]
            raw_qty = allocation_per_symbol / price
            qty = self._round_qty(raw_qty)
            notional = qty * price
            if notional < self._config.min_order_notional:
                continue

            atr = atrs_map.get(sym, 0.0)
            stop_price = None
            if atr > 0:
                stop_dist = atr * self._config.atr_stop_multiple
                cand_stop = round(price - stop_dist, 2)
                if cand_stop > 0:
                    stop_price = cand_stop

            token = self._mint_token(
                symbol=sym,
                side=Side.BUY,
                qty=qty,
                now=now,
            )
            intent = OrderIntent(
                symbol=sym,
                side=Side.BUY,
                qty=qty,
                token=token,
                order_type=OrderType.MARKET,
                stop_price=stop_price,
                reason="cash_sweep",
            )
            metrics = {
                "price": price,
                "qty": qty,
                "notional": notional,
                "sweep_budget": sweep_budget,
                "allocation": allocation_per_symbol,
            }
            decisions.append(self._approve(intent, now, metrics))

        return decisions

    def evaluate_shield_allocation(
        self,
        account: AccountState,
        *,
        symbol: str,
        price: float,
        target_weight: float,
        now: datetime,
    ) -> RiskDecision | None:
        """Complète la poche défensive jusqu'à son poids cible.

        Le Bouclier n'obéit pas à un signal : il se finance sur le cash
        excédentaire (au-delà du buffer de liquidités et du plancher retirable)
        jusqu'à représenter ``target_weight`` de l'équité. Le complément est
        plafonné par l'exposition brute restante pour ne jamais franchir les
        limites du profil de risque.

        Returns:
            Une décision d'achat approuvée portant un :class:`OrderIntent`, ou
            ``None`` si la poche est déjà financée, si le prix est
            inexploitable, ou si le complément est trop petit pour valoir un
            ordre.
        """
        self._require_aware(now)
        if target_weight <= 0.0 or account.equity <= 0.0 or account.halted:
            return None
        if not math.isfinite(price) or price <= 0.0:
            return None

        position = account.positions.get(symbol)
        current_value = (
            position.market_value if position is not None and position.qty > 0 else 0.0
        )
        gap = account.equity * target_weight - current_value
        min_topup = max(
            self._config.min_order_notional, account.equity * _SHIELD_MIN_TOPUP_PCT
        )
        if gap < min_topup:
            return None

        protected_cash = max(
            account.equity * self._config.cash_buffer_pct,
            self._config.withdrawable_cash_floor_usd,
        )
        cash_available = max(0.0, min(account.cash - protected_cash, account.buying_power))
        gross_remaining = max(
            0.0, account.equity * self.effective_gross_cap_pct() - account.gross_exposure
        )
        budget = min(gap, cash_available, gross_remaining)
        qty = self._round_qty(budget / price)
        if qty <= 0.0 or qty * price < self._config.min_order_notional:
            return None

        token = self._mint_token(symbol=symbol, side=Side.BUY, qty=qty, now=now)
        intent = OrderIntent(
            symbol=symbol,
            side=Side.BUY,
            qty=qty,
            token=token,
            order_type=OrderType.MARKET,
            stop_price=None,
            reason="shield_allocation",
        )
        metrics = {
            "price": price,
            "qty": qty,
            "notional": qty * price,
            "target_weight": target_weight,
            "gap": gap,
        }
        return self._approve(intent, now, metrics)

    def evaluate_stop_exit(
        self,
        position: Position,
        *,
        price: float,
        now: datetime,
        reason: str | None = None,
    ) -> RiskDecision | None:
        """Retourne une sortie si ``price`` a atteint le stop protecteur."""
        if position.stop_price is None or price > position.stop_price:
            return None
        trailing = (
            position.initial_stop_price is not None
            and position.stop_price > position.initial_stop_price
        )
        label = reason or ("trailing_stop" if trailing else "stop_loss")
        return self.evaluate_exit(
            position=position,
            price=price,
            now=now,
            reason=label,
        )

    def news_vetoes_entry(self, news: NewsSignal) -> bool:
        """Indique si la note d'actualités interdit une nouvelle entrée.

        Le drapeau ``is_critical_threat`` suffit à lui seul : une fraude ou
        une menace existentielle ne se moyenne pas, contrairement à une
        tonalité simplement maussade.

        Args:
            news: Note d'actualités du symbole visé.

        Returns:
            ``True`` si l'entrée doit être refusée.
        """
        if news.is_critical_threat:
            return True
        return news.sentiment_score <= self._config.news_veto_threshold

    def evaluate_news_panic_exit(
        self,
        position: Position,
        *,
        news: NewsSignal | None,
        price: float,
        now: datetime,
    ) -> RiskDecision | None:
        """Liquide une position sur menace critique, sans attendre le stop.

        Un score effondré accompagné d'une menace critique justifie de sortir
        au marché plutôt que d'attendre le stop ATR : sur un krach, le stop
        glisse et se remplit bien plus bas que son seuil. La sortie emprunte
        le chemin normal (:meth:`evaluate_exit`), et reste donc autorisée
        même circuit breaker déclenché — c'est précisément le mécanisme de
        mise à plat du portefeuille.

        Args:
            position: Position ouverte à examiner.
            news: Note d'actualités, ou ``None`` si le filtre est inactif.
            price: Dernier prix connu du symbole.
            now: Horodatage de la décision.

        Returns:
            Une décision de sortie motivée ``news_panic_exit``, ou ``None``
            si la menace ne justifie pas une liquidation immédiate. Un prix
            inexploitable retourne ``None`` plutôt que de lever : ce moteur
            est pur, il n'a pas à faire échouer un cycle.
        """
        if news is None or position.qty <= 0:
            return None
        if not news.is_critical_threat:
            return None
        if news.sentiment_score > self._config.news_panic_exit_threshold:
            return None
        if not math.isfinite(price) or price <= 0:
            return None
        return self.evaluate_exit(
            position=position,
            price=price,
            now=now,
            reason="news_panic_exit",
        )

    def evaluate_partial_take_profit(
        self,
        position: Position,
        *,
        price: float,
        now: datetime,
    ) -> RiskDecision | None:
        """Évalue si une position offensive est éligible au take-profit partiel (+20 %).

        En cas d'éligibilité (plus-value >= config.partial_take_profit_pct et
        partial_tp_taken est Faux), génère une vente de 50 % des titres
        (config.partial_take_profit_sell_ratio) avec le motif partial_take_profit.
        """
        self._require_aware(now)
        if self._config.partial_take_profit_pct <= 0:
            return None
        if position.partial_tp_taken or position.qty <= 0:
            return None
        if self.is_defensive_symbol(position.symbol):
            return None
        if not math.isfinite(price) or price <= 0:
            return None

        unrealized_pct = (price - position.avg_entry_price) / position.avg_entry_price
        if unrealized_pct < self._config.partial_take_profit_pct - 1e-9:
            return None

        sell_ratio = min(1.0, max(0.01, self._config.partial_take_profit_sell_ratio))
        raw_qty = position.qty * sell_ratio
        sell_qty = self._round_qty(raw_qty)
        if sell_qty <= 0 or sell_qty >= position.qty:
            return None
        if sell_qty * price < self._config.min_order_notional:
            return None

        return self.evaluate_exit(
            position=position,
            price=price,
            now=now,
            reason="partial_take_profit",
            qty=sell_qty,
        )

    def evaluate_overweight_trim(
        self,
        position: Position,
        *,
        price: float,
        account: AccountState,
        now: datetime,
    ) -> RiskDecision | None:
        """Écréme l'excédent d'une position dépassant le plafond de poids max (25 %).

        Ramène la position à la taille cible (effective_position_cap_pct, 20 %).
        """
        self._require_aware(now)
        if self._config.max_single_position_weight <= 0:
            return None
        if account.equity <= 0 or position.qty <= 0 or price <= 0:
            return None
        if self.is_defensive_symbol(position.symbol):
            return None
        if not math.isfinite(price) or price <= 0:
            return None

        current_val = position.qty * price
        weight = current_val / account.equity
        if weight <= self._config.max_single_position_weight + 1e-9:
            return None

        target_cap = self.effective_position_cap_pct()
        target_val = account.equity * target_cap
        excess_val = current_val - target_val
        if excess_val <= 0:
            return None

        raw_qty = excess_val / price
        sell_qty = self._round_qty(raw_qty)
        if sell_qty <= 0 or sell_qty >= position.qty:
            return None
        if sell_qty * price < self._config.min_order_notional:
            return None

        return self.evaluate_exit(
            position=position,
            price=price,
            now=now,
            reason="position_trim_overweight",
            qty=sell_qty,
        )

    def evaluate_manual_withdrawal(
        self,
        position: Position,
        *,
        price: float,
        qty: float,
        now: datetime,
        reason: str = "manual_withdrawal",
    ) -> RiskDecision:
        """Autorise et scelle un ordre de vente pour un retrait manuel de cash."""
        self._require_aware(now)
        if qty <= 0 or not math.isfinite(qty):
            raise RiskError(f"quantité de retrait invalide : {qty}")
        if price <= 0 or not math.isfinite(price):
            raise RiskError(f"prix invalide : {price}")

        sell_qty = min(position.qty, self._round_qty(qty))
        if sell_qty <= 0:
            raise RiskError("quantité de retrait arrondie à zéro")

        return self.evaluate_exit(
            position=position,
            price=price,
            now=now,
            reason=reason,
            qty=sell_qty,
        )

    def evaluate_stop_replace(
        self,
        position: Position,
        *,
        price: float,
        now: datetime,
    ) -> RiskDecision | None:
        """Scelle un ordre stop côté courtier pour refléter le stop de la position.

        Utilisé pour matérialiser chez le broker le stop d'une position adoptée
        (ouverte hors bot) ou le relèvement d'un stop trailing : le stop doit
        survivre à un crash du process. Retourne ``None`` si aucun stop valide
        ne peut être placé sous le prix courant.
        """
        self._require_aware(now)
        if position.qty <= 0:
            return None
        if position.stop_price is None or not math.isfinite(position.stop_price):
            return None
        if not math.isfinite(price) or price <= 0:
            return None
        if position.stop_price <= 0 or position.stop_price >= price:
            return None
        token = self._mint_token(
            symbol=position.symbol,
            side=Side.SELL,
            qty=position.qty,
            now=now,
        )
        intent = OrderIntent(
            symbol=position.symbol,
            side=Side.SELL,
            qty=position.qty,
            token=token,
            order_type=OrderType.STOP,
            stop_price=round(position.stop_price, 2),
            reason="stop_replace",
        )
        metrics = {
            "price": price,
            "qty": position.qty,
            "stop_price": position.stop_price,
            "r_multiple": position.r_multiple,
        }
        return self._approve(intent, now, metrics)

    def update_trailing_stop(
        self,
        position: Position,
        *,
        price: float,
        atr: float,
    ) -> TrailingStopUpdate | None:
        """Propose un relèvement du stop une fois le seuil d'activation atteint.

        Le stop ne redescend jamais. À l'activation (``trailing_activation_r``),
        il est remonté au minimum au prix d'entrée, puis suit
        ``prix - trailing_atr_multiple * ATR``. La proposition prend effet à la
        barre suivante : l'appelant l'applique après la clôture, jamais avant.
        """
        config = self._config
        if self.is_defensive_symbol(position.symbol):
            return None
        if config.trailing_atr_multiple <= 0.0:
            return None
        if position.stop_price is None or position.initial_stop_price is None:
            return None
        risk_per_share = position.risk_per_share
        if risk_per_share is None:
            return None
        if not math.isfinite(price) or price <= 0 or not math.isfinite(atr) or atr <= 0:
            return None
        activation_price = position.avg_entry_price + (
            config.trailing_activation_r * risk_per_share
        )
        if price < activation_price:
            return None

        candidate = max(
            price - config.trailing_atr_multiple * atr,
            position.avg_entry_price,
        )
        candidate = round(candidate, 2)
        if candidate >= price or candidate <= position.stop_price + _STOP_TICK:
            return None
        return TrailingStopUpdate(
            symbol=position.symbol,
            previous_stop=position.stop_price,
            new_stop=candidate,
            price=price,
            atr=atr,
            r_multiple=position.r_multiple,
        )

    def apply_trailing_stop(
        self,
        account: AccountState,
        update: TrailingStopUpdate,
    ) -> bool:
        """Applique un relèvement de stop si la position existe toujours.

        Returns:
            ``True`` si le stop a effectivement été relevé.
        """
        position = account.positions.get(update.symbol)
        if position is None:
            return False
        if position.stop_price is not None and update.new_stop <= position.stop_price:
            return False
        position.stop_price = update.new_stop
        return True

    def flatten_decisions(
        self,
        account: AccountState,
        *,
        prices: Mapping[str, float],
        now: datetime,
        reason: str,
    ) -> list[RiskDecision]:
        """Produit les sorties de toutes les positions liquides."""
        decisions: list[RiskDecision] = []
        for symbol, position in sorted(account.positions.items()):
            price = prices.get(symbol, position.current_price)
            if position.qty <= 0 or price <= 0:
                continue
            decisions.append(
                self.evaluate_exit(
                    position=position,
                    price=price,
                    now=now,
                    reason=reason,
                )
            )
        return decisions

    def record_realized_pnl(self, account: AccountState, amount: float) -> None:
        """Comptabilise un P&L réalisé dans le compteur journalier."""
        if not math.isfinite(amount):
            raise RiskError("P&L réalisé non fini")
        account.realized_pnl_today += amount

    def rebase_equity_reference(
        self, account: AccountState, *, reason: str
    ) -> tuple[float | None, bool]:
        """Ré-ancre le pic et l'équité de séance sur l'équité courante.

        Un dépôt ou un retrait effectué chez le courtier modifie l'équité sans
        qu'aucun trade n'ait eu lieu : le drawdown mesuré depuis l'ancien pic
        devient fictif et déclenche l'arrêt permanent. Cette opération, toujours
        justifiée par l'appelant, recale la référence et lève l'arrêt permanent
        s'il était actif — c'est précisément l'intervention humaine que
        :meth:`reset_permanent_halt` exige.

        Args:
            account: État du compte, modifié sur place.
            reason: Justification en clair (auditée par l'appelant).

        Returns:
            ``(ancien pic, arrêt permanent levé)``.

        Raises:
            RiskError: si la justification est vide.
        """
        if not reason.strip():
            raise RiskError("une justification est requise pour ré-ancrer l'équité")
        if not math.isfinite(account.equity) or account.equity < 0:
            raise RiskError(f"équité invalide: {account.equity!r}")

        previous_peak = account.peak_equity
        account.peak_equity = account.equity
        account.day_start_equity = account.equity
        account.realized_pnl_today = 0.0
        lifted = account.halt_scope is HaltScope.PERMANENT
        if lifted:
            account.halted = False
            account.halt_scope = None
            account.halt_reason = None
        self._resumptions.append(reason.strip())
        return previous_peak, lifted

    def reset_permanent_halt(self, account: AccountState, *, reason: str) -> None:
        """Lève un arrêt définitif après inspection humaine explicite."""
        if account.halt_scope is not HaltScope.PERMANENT:
            raise RiskError("aucun arrêt permanent à lever")
        if not reason.strip():
            raise RiskError("une justification est requise pour lever un arrêt permanent")
        account.halted = False
        account.halt_scope = None
        account.halt_reason = None
        account.peak_equity = account.equity
        account.day_start_equity = account.equity
        self._resumptions.append(reason.strip())

    def _halt(
        self,
        account: AccountState,
        *,
        session_date: date,
        scope: HaltScope,
        reason: str,
    ) -> HaltEvent:
        account.halted = True
        account.halt_scope = scope
        account.halt_reason = reason
        event = HaltEvent(
            scope=scope,
            reason=reason,
            session_date=session_date,
            equity=account.equity,
            daily_pnl_pct=account.daily_pnl_pct,
            drawdown_pct=account.drawdown_pct,
        )
        self._halts.append(event)
        return event

    def _mint_token(
        self,
        *,
        symbol: str,
        side: Side,
        qty: float,
        now: datetime,
    ) -> ApprovalToken:
        """Crée un jeton à durée de vie courte, signature privée du module."""
        self._sequence += 1
        digest = hashlib.sha256(
            f"{now.isoformat()}|{self._sequence}|{symbol}|{side.value}|{qty:.8f}".encode()
        ).hexdigest()[:24]
        return ApprovalToken(
            decision_id=digest,
            symbol=symbol,
            side=side,
            qty=qty,
            issued_at=now,
            expires_at=now + timedelta(seconds=self._config.order_ttl_seconds),
            issuer=_TOKEN_ISSUER,
        )

    def _approve(
        self,
        intent: OrderIntent,
        now: datetime,
        metrics: Mapping[str, float],
    ) -> RiskDecision:
        decision = RiskDecision(
            decision_id=intent.token.decision_id,
            created_at=now,
            approved=True,
            code=RejectionCode.APPROVED,
            reason=intent.reason or "ordre approuvé",
            intent=intent,
            metrics=dict(metrics),
        )
        self._audit.append(decision)
        return decision

    def _reject(
        self,
        code: RejectionCode,
        reason: str,
        now: datetime,
        metrics: Mapping[str, float],
    ) -> RiskDecision:
        self._sequence += 1
        digest = hashlib.sha256(
            f"{now.isoformat()}|{self._sequence}|reject|{code.value}".encode()
        ).hexdigest()[:24]
        decision = RiskDecision(
            decision_id=digest,
            created_at=now,
            approved=False,
            code=code,
            reason=reason,
            intent=None,
            metrics=dict(metrics),
        )
        self._audit.append(decision)
        return decision

    def _round_qty(self, qty: float) -> float:
        """Arrondit vers le bas au pas autorisé (jamais au-dessus du risque)."""
        decimals = self._config.qty_decimals if self._config.fractional_shares else 0
        factor = 10**decimals
        return float(math.floor(qty * factor + _QTY_EPSILON) / factor)

    @staticmethod
    def _require_aware(now: datetime) -> None:
        """Exige un horodatage timezone-aware pour éviter les ambiguïtés."""
        if now.tzinfo is None:
            raise RiskError("horodatage naïf interdit : fournir un datetime UTC")


__all__ = [
    "AccountState",
    "ApprovalToken",
    "HaltEvent",
    "HaltScope",
    "OrderIntent",
    "OrderType",
    "Position",
    "RejectionCode",
    "RiskDecision",
    "RiskEngine",
    "RiskError",
    "Side",
    "TrailingStopUpdate",
]
