"""Retraits manuels et sanctuarisation de cash : un seul chemin d'exécution.

L'interface web et la commande ``withdraw`` faisaient la même chose en deux
copies, avec une divergence de sécurité : le cycle de trading refuse de vendre
tant que l'annulation des stops n'est pas **confirmée**, alors que les deux
copies se contentaient d'un ``hasattr`` et ignoraient le résultat. Une jambe de
stop survivante peut vendre à découvert après la vente. Ce module rassemble
donc, pour les deux appelants :

* l'annulation vérifiée des ordres ouverts (:func:`cancel_open_orders_verified`) ;
* la vente validée par le moteur de risque, avec remise du stop sur le reliquat ;
* l'audit dans ``audit.jsonl``, avec les mêmes événements que le cycle de trading ;
* la mise à jour du plancher de cash dans ``.env``.

La CLI et la page web ne portent plus que l'habillage (arguments, messages).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from trading_bot.clock import get_online_now
from trading_bot.config import PROJECT_ROOT, Settings, update_env_file
from trading_bot.execution.alpaca_client import (
    AlpacaPaperClient,
    BrokerError,
    OrderReceipt,
)
from trading_bot.risk.engine import (
    AccountState,
    Position,
    RiskDecision,
    RiskEngine,
)

LOGGER = logging.getLogger(__name__)

_ORDER_FILL_TIMEOUT_SECONDS = 30.0
_QTY_EPSILON = 1e-6


class AuditSink(Protocol):
    """Journal d'audit minimal (``AuditLog`` de l'orchestrateur)."""

    def record(self, event: str, payload: Mapping[str, Any]) -> None:
        """Ajoute un événement horodaté."""
        ...


class StateStoreLike(Protocol):
    """Persistance d'état minimale (``StateStore`` de l'orchestrateur)."""

    def load(self) -> Any:
        """Charge l'état persisté."""
        ...

    def save(self, state: Any) -> None:
        """Écrit l'état persisté."""
        ...


def cancel_open_orders_verified(
    broker: Any,
    symbol: str,
    *,
    audit: AuditSink | None = None,
    errors: list[str] | None = None,
) -> bool:
    """Annule les ordres ouverts d'un symbole et vérifie l'annulation.

    Un ordre stop orphelin peut se déclencher après une vente et laisser le
    compte net vendeur. L'appelant ne doit donc vendre que si cette fonction a
    rendu ``True``.

    Args:
        broker: Passerelle d'exécution (``cancel_open_orders`` attendu).
        symbol: Symbole concerné.
        audit: Journal d'audit à enrichir, si disponible.
        errors: Liste de messages à enrichir en cas d'échec.

    Returns:
        ``True`` si le courtier confirme qu'il ne reste aucun ordre ouvert.
    """
    sink = errors if errors is not None else []
    cancel = getattr(broker, "cancel_open_orders", None)
    if cancel is None:
        message = (
            f"{symbol}: courtier incapable de confirmer l'annulation des ordres ouverts"
        )
        sink.append(message)
        _record(audit, "cancel_open_orders_failed", {"symbol": symbol, "error": message})
        LOGGER.error("%s", message)
        return False
    try:
        cancelled = int(cancel(symbol))
    except BrokerError as exc:
        message = f"{symbol}: annulation des ordres ouverts impossible: {exc}"
        sink.append(message)
        _record(audit, "cancel_open_orders_failed", {"symbol": symbol, "error": str(exc)})
        LOGGER.error("%s", message)
        return False
    _record(audit, "cancel_open_orders", {"symbol": symbol, "count": cancelled})
    return True


def _record(
    audit: AuditSink | None,
    event: str,
    payload: Mapping[str, Any],
) -> None:
    """Écrit dans l'audit si un journal a été fourni."""
    if audit is None:
        return
    audit.record(event, payload)


def _record_decision(audit: AuditSink | None, decision: RiskDecision) -> None:
    """Journalise une décision de risque, au même format que le cycle live."""
    _record(
        audit,
        "risk_decision",
        {
            "decision_id": decision.decision_id,
            "approved": decision.approved,
            "code": decision.code.value,
            "reason": decision.reason,
            "symbol": decision.intent.symbol if decision.intent else None,
            "side": decision.intent.side.value if decision.intent else None,
            "qty": decision.intent.qty if decision.intent else None,
            "stop_price": decision.intent.stop_price if decision.intent else None,
            "metrics": dict(decision.metrics),
        },
    )


def _record_order(audit: AuditSink | None, receipt: OrderReceipt) -> None:
    """Journalise un ordre exécuté, au même format que le cycle live."""
    _record(
        audit,
        "order",
        {
            "order_id": receipt.order_id,
            "client_order_id": receipt.client_order_id,
            "symbol": receipt.symbol,
            "side": receipt.side.value,
            "qty": receipt.qty,
            "filled_qty": receipt.filled_qty,
            "avg_fill_price": receipt.avg_fill_price,
            "status": receipt.status,
        },
    )


@dataclass(frozen=True, slots=True)
class WithdrawalFill:
    """Vente exécutée (ou simulée) lors d'un retrait."""

    symbol: str
    qty: float
    price: float
    reason: str
    order_id: str | None = None

    @property
    def cash_freed(self) -> float:
        """Produit de la vente, en dollars."""
        return self.qty * self.price


@dataclass(slots=True)
class WithdrawalResult:
    """Résultat observable d'un retrait, prêt à afficher."""

    fills: list[WithdrawalFill] = field(default_factory=list)
    cash: float = 0.0
    floor: float | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def cash_freed(self) -> float:
        """Somme des produits de vente."""
        return sum(fill.cash_freed for fill in self.fills)


class WithdrawalService:
    """Exécute un retrait : ventes, stops, audit et plancher de cash.

    Args:
        settings: Configuration effective (dry-run, plan d'eau, safe havens).
        broker: Passerelle Alpaca paper.
        risk: Moteur de risque ; un moteur neuf est créé au besoin.
        state_store: Persistance des stops (``StateStore``).
        audit: Journal d'audit (``AuditLog``).
        env_path: Fichier ``.env`` où écrire le plancher de cash.
        clock: Horloge injectable (tests).
    """

    def __init__(
        self,
        settings: Settings,
        broker: AlpacaPaperClient,
        *,
        risk: RiskEngine | None = None,
        state_store: StateStoreLike | None = None,
        audit: AuditSink | None = None,
        env_path: Path | None = None,
        clock: Callable[[], datetime] = get_online_now,
    ) -> None:
        self._settings = settings
        self._broker = broker
        self._risk = risk or RiskEngine(settings.risk)
        self._state_store = state_store
        self._audit = audit
        self._env_path = env_path or (PROJECT_ROOT / ".env")
        self._clock = clock

    # ------------------------------------------------------------------
    # État du compte
    # ------------------------------------------------------------------
    def load_account(self) -> AccountState:
        """Compte courant, stops persistés restaurés."""
        snapshot = self._broker.get_account()
        positions = self._broker.get_positions()
        stops = self._stored_stops()
        for symbol, position in positions.items():
            stored = stops.get(symbol)
            if stored:
                position.apply_stop_state(stored)
        return AccountState(
            equity=snapshot.equity,
            cash=snapshot.cash,
            buying_power=snapshot.buying_power,
            positions=positions,
        )

    def _stored_stops(self) -> dict[str, dict[str, Any]]:
        """Stops persistés par symbole (vide sans magasin d'état)."""
        if self._state_store is None:
            return {}
        state = self._state_store.load()
        stored = getattr(state, "stops", {})
        return dict(stored) if isinstance(stored, Mapping) else {}

    def _persist_stops(self, account: AccountState) -> None:
        """Enregistre les stops des positions restantes."""
        if self._state_store is None:
            return
        state = self._state_store.load()
        state.stops = {
            symbol: {
                key: value
                for key, value in position.stop_state().items()
                if value is not None
            }
            for symbol, position in account.positions.items()
        }
        self._state_store.save(state)

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------
    def withdraw_symbol(
        self,
        symbol: str,
        *,
        amount: float | None = None,
        pct: float | None = None,
    ) -> WithdrawalResult:
        """Vend une fraction d'une ligne, montant en dollars ou pourcentage.

        Raises:
            ValueError: symbole absent du portefeuille, montant ou pourcentage
                invalides, prix indisponible.
        """
        account = self.load_account()
        normalized = symbol.strip().upper()
        if normalized not in account.positions:
            raise ValueError(f"le symbole {normalized} n'est pas en portefeuille")
        position = account.positions[normalized]
        price = position.current_price
        if price <= 0 or not math.isfinite(price):
            raise ValueError(f"prix introuvable pour {normalized}")

        qty = self._quantity_for(position, price=price, amount=amount, pct=pct)
        result = self._sell(
            account, position, qty=qty, price=price, reason="manual_withdrawal"
        )
        if result.fills:
            self._persist_stops(account)
        result.cash = account.cash
        self._audit_summary("manual_withdrawal", normalized, result)
        return result

    def withdraw_free_cash(self, target_cash: float) -> WithdrawalResult:
        """Libère du cash jusqu'à ``target_cash`` et sanctuarise ce plancher.

        Les positions de repli sont vendues en premier, puis les lignes
        offensives les plus gagnantes. Le plancher est écrit dans ``.env`` afin
        que les cycles suivants le respectent.

        Raises:
            ValueError: montant cible négatif ou nul.
        """
        if not math.isfinite(target_cash) or target_cash <= 0:
            raise ValueError("le montant free-cash doit être strictement positif")

        account = self.load_account()
        result = WithdrawalResult(floor=target_cash)
        needed = target_cash - account.cash
        LOGGER.info(
            "retrait global : cash actuel=%.2f$, cible=%.2f$, besoin=%.2f$",
            account.cash,
            target_cash,
            max(0.0, needed),
        )
        if needed > 0:
            safe_havens = set(self._settings.safe_haven_symbols)
            for symbol in [
                s for s in self._settings.safe_haven_symbols if s in account.positions
            ]:
                if needed <= 0:
                    break
                needed -= self._drain(
                    account, symbol, needed, result, "safe_haven_withdrawal"
                )

            if needed > 0:
                offensives = [
                    position
                    for symbol, position in account.positions.items()
                    if symbol not in safe_havens and position.unrealized_pnl_pct > 0
                ]
                offensives.sort(key=lambda item: item.unrealized_pnl_pct, reverse=True)
                for position in offensives:
                    if needed <= 0:
                        break
                    needed -= self._drain(
                        account, position.symbol, needed, result, "gain_trim_withdrawal"
                    )

        if not self._settings.dry_run:
            update_env_file(
                self._env_path, {"WITHDRAWABLE_CASH_FLOOR_USD": f"{target_cash:.2f}"}
            )
            LOGGER.info(
                "plancher WITHDRAWABLE_CASH_FLOOR_USD mis à jour à %.2f$ dans %s",
                target_cash,
                self._env_path,
            )
            self._persist_stops(account)
        else:
            LOGGER.info(
                "dry-run : plancher WITHDRAWABLE_CASH_FLOOR_USD serait mis à jour à %.2f$",
                target_cash,
            )

        result.cash = account.cash
        self._audit_summary("free_cash_withdrawal", "global", result)
        return result

    # ------------------------------------------------------------------
    # Détail
    # ------------------------------------------------------------------
    @staticmethod
    def _quantity_for(
        position: Position,
        *,
        price: float,
        amount: float | None,
        pct: float | None,
    ) -> float:
        """Quantité à vendre selon le montant ou le pourcentage demandé."""
        if amount is not None:
            if amount <= 0 or not math.isfinite(amount):
                raise ValueError("le montant doit être > 0")
            return min(position.qty, round(amount / price, 4))
        if pct is not None:
            if pct <= 0 or pct > 100:
                raise ValueError("le pourcentage doit être compris entre 0 et 100")
            return min(position.qty, round(position.qty * (pct / 100.0), 4))
        raise ValueError("veuillez spécifier soit --amount soit --pct")

    def _drain(
        self,
        account: AccountState,
        symbol: str,
        needed: float,
        result: WithdrawalResult,
        reason: str,
    ) -> float:
        """Vend juste ce qu'il faut d'une ligne pour couvrir ``needed``.

        Returns:
            Le montant effectivement libéré (0 si la vente a échoué).
        """
        position = account.positions.get(symbol)
        if position is None:
            return 0.0
        price = position.current_price
        if price <= 0 or not math.isfinite(price):
            return 0.0
        qty = min(position.qty, round(min(position.qty * price, needed) / price, 4))
        if qty <= 0:
            return 0.0
        before = len(result.fills)
        self._sell(account, position, qty=qty, price=price, reason=reason)
        if len(result.fills) == before:
            return 0.0
        return result.fills[-1].cash_freed

    def _sell(
        self,
        account: AccountState,
        position: Position,
        *,
        qty: float,
        price: float,
        reason: str,
    ) -> WithdrawalResult:
        """Valide puis exécute une vente, en échouant fermé.

        La vente n'est transmise que si l'annulation des ordres ouverts est
        confirmée : sans cela, une jambe de stop pourrait se déclencher après
        la vente et laisser le compte net vendeur.
        """
        result = WithdrawalResult()
        if qty <= 0 or not math.isfinite(qty):
            result.errors.append(f"{position.symbol}: quantité de retrait invalide")
            return result

        now = self._clock()
        decision = self._risk.evaluate_manual_withdrawal(
            position, price=price, qty=qty, now=now, reason=reason
        )
        _record_decision(self._audit, decision)
        if not decision.approved or decision.intent is None:
            result.errors.append(f"{position.symbol}: {decision.reason}")
            return result

        if not cancel_open_orders_verified(
            self._broker, position.symbol, audit=self._audit, errors=result.errors
        ):
            return result

        if self._settings.dry_run:
            result.fills.append(WithdrawalFill(position.symbol, qty, price, reason))
            return result

        try:
            receipt = self._broker.submit(decision.intent, now=now)
            filled = self._broker.wait_for_fill(
                receipt.order_id, timeout_seconds=_ORDER_FILL_TIMEOUT_SECONDS
            )
        except BrokerError as exc:
            result.errors.append(f"{position.symbol}: {exc}")
            LOGGER.error("échec de la vente de retrait %s: %s", position.symbol, exc)
            return result

        _record_order(self._audit, filled)
        sold = filled.filled_qty if filled.filled_qty > 0 else qty
        fill_price = filled.avg_fill_price if filled.avg_fill_price else price
        account.cash += sold * fill_price
        account.buying_power = account.cash
        if sold >= position.qty - _QTY_EPSILON:
            account.positions.pop(position.symbol, None)
        else:
            position.qty -= sold
            self._replace_stop(position, price=fill_price, result=result)

        result.fills.append(
            WithdrawalFill(position.symbol, sold, fill_price, reason, filled.order_id)
        )
        LOGGER.info(
            "vente %s (%.4f titres @ %.2f$) : %+.2f$ de cash",
            position.symbol,
            sold,
            fill_price,
            sold * fill_price,
        )
        return result

    def _replace_stop(
        self,
        position: Position,
        *,
        price: float,
        result: WithdrawalResult,
    ) -> None:
        """Redépose le stop sur le reliquat d'une ligne partiellement vendue.

        La position a été allégée : la quantité de la jambe de protection ne
        correspond plus. Le nouveau stop est calculé par le moteur et laissé au
        statut du courtier s'il ne peut pas être posé.
        """
        if position.stop_price is None or self._settings.dry_run:
            return
        decision = self._risk.evaluate_stop_replace(
            position, price=price, now=self._clock()
        )
        if decision is None or not decision.approved or decision.intent is None:
            return
        try:
            receipt = self._broker.submit(decision.intent, now=self._clock())
        except BrokerError as exc:
            result.errors.append(f"{position.symbol}: stop non redéposé ({exc})")
            LOGGER.error("stop non redéposé pour %s: %s", position.symbol, exc)
            return
        _record(
            self._audit,
            "stop_replace",
            {
                "symbol": decision.intent.symbol,
                "stop_price": decision.intent.stop_price,
                "qty": decision.intent.qty,
                "client_order_id": receipt.client_order_id,
            },
        )

    def _audit_summary(
        self,
        kind: str,
        target: str,
        result: WithdrawalResult,
    ) -> None:
        """Résumé du retrait dans l'audit, quel que soit son issue."""
        _record(
            self._audit,
            "withdrawal",
            {
                "kind": kind,
                "target": target,
                "floor": result.floor,
                "dry_run": self._settings.dry_run,
                "fills": [
                    {
                        "symbol": fill.symbol,
                        "qty": fill.qty,
                        "price": fill.price,
                        "reason": fill.reason,
                        "order_id": fill.order_id,
                    }
                    for fill in result.fills
                ],
                "errors": list(result.errors),
            },
        )


__all__ = [
    "AuditSink",
    "StateStoreLike",
    "WithdrawalFill",
    "WithdrawalResult",
    "WithdrawalService",
    "cancel_open_orders_verified",
]
