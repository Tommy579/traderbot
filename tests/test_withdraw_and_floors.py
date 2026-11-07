"""Tests des fonctionnalités de retrait, planchers de sécurité et prise de profit partielle."""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from trading_bot.config import RiskConfig, Settings
from trading_bot.execution.alpaca_client import OrderReceipt
from trading_bot.main import PersistedState, StateStore, cmd_withdraw
from trading_bot.risk.engine import (
    AccountState,
    HaltScope,
    Position,
    RejectionCode,
    RiskEngine,
    Side,
)

NOW = datetime(2024, 1, 10, 15, 0, tzinfo=UTC)
SESSION = date(2024, 1, 10)


class FakeBroker:
    """Courtier factice pour tester les retraits et exécutions."""

    def __init__(
        self,
        *,
        equity: float = 100_000.0,
        cash: float = 100_000.0,
        buying_power: float = 100_000.0,
        positions: dict[str, Position] | None = None,
    ) -> None:
        self.equity = equity
        self.cash = cash
        self.buying_power = buying_power
        self._positions: dict[str, Position] = positions or {}
        self.orders: list[Any] = []
        self.cancelled: list[str] = []

    def get_account(self) -> Any:
        return type(
            "AccountSnapshot",
            (),
            {
                "equity": self.equity,
                "cash": self.cash,
                "buying_power": self.buying_power,
            },
        )()

    def get_positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def cancel_open_orders(self, symbol: str) -> int:
        self.cancelled.append(symbol)
        return 1

    def submit(self, intent: Any, now: datetime | None = None) -> OrderReceipt:
        self.orders.append(intent)
        price = getattr(intent, "stop_price", None) or 100.0
        return OrderReceipt(
            order_id=f"ord-{len(self.orders)}",
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            filled_qty=intent.qty,
            avg_fill_price=price,
            status="filled",
            submitted_at=now or NOW,
        )

    def wait_for_fill(self, order_id: str, timeout_seconds: float = 30.0) -> OrderReceipt:
        matching = [o for o in self.orders if f"ord-{self.orders.index(o) + 1}" == order_id]
        intent = matching[0] if matching else self.orders[-1]
        price = getattr(intent, "stop_price", None) or 100.0
        return OrderReceipt(
            order_id=order_id,
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            filled_qty=intent.qty,
            avg_fill_price=price,
            status="filled",
            submitted_at=NOW,
        )


def test_withdrawable_cash_floor_protects_cash_and_reconstitutes() -> None:
    """Vérifie la sanctuarisation du cash retirable et la reconstitution via safe havens."""
    config = RiskConfig(
        withdrawable_cash_floor_usd=5_000.0,
        cash_buffer_pct=0.10,
        safe_haven_symbols=["BRK.B", "KO"],
    )
    risk = RiskEngine(config)

    # 1. Cash sweep ne doit pas empiéter sur le plancher
    # Équité 100k, cash 4k (< 5k de plancher) -> aucun cash sweep
    account = AccountState(
        equity=100_000.0,
        cash=4_000.0,
        buying_power=4_000.0,
        positions={
            "BRK.B": Position(
                symbol="BRK.B",
                qty=20.0,
                avg_entry_price=300.0,
                current_price=300.0,
            )
        },
    )
    risk.start_session(account, session_date=SESSION)
    sweeps = risk.evaluate_cash_sweep(account, {"BRK.B": 300.0, "KO": 60.0}, now=NOW)
    assert not sweeps

    # 2. Reconstitution du plancher : cash 4k vs plancher 5k (besoin de 1k)
    reconstitutions = risk.evaluate_safe_haven_cash_floor_reconstitution(
        account, {"BRK.B": 300.0, "KO": 60.0}, now=NOW
    )
    assert len(reconstitutions) == 1
    rec = reconstitutions[0]
    assert rec.approved
    assert rec.intent is not None
    assert rec.intent.token is not None
    assert rec.intent.symbol == "BRK.B"
    assert rec.intent.side is Side.SELL
    # 1000$ / 300$ = 3.3333 actions
    assert pytest.approx(rec.intent.qty, rel=1e-2) == 3.3333

    # 3. Quand le cash réel dépasse le plancher, aucune reconstitution
    account_ok = AccountState(
        equity=100_000.0,
        cash=6_000.0,
        buying_power=6_000.0,
        positions={
            "BRK.B": Position(
                symbol="BRK.B",
                qty=20.0,
                avg_entry_price=300.0,
                current_price=300.0,
            )
        },
    )
    assert not risk.evaluate_safe_haven_cash_floor_reconstitution(
        account_ok, {"BRK.B": 300.0, "KO": 60.0}, now=NOW
    )


def test_partial_take_profit_and_break_even() -> None:
    """Vérifie la prise de profit partielle à +20% et le passage du stop à Break-Even."""
    config = RiskConfig(
        partial_take_profit_pct=0.20,
        partial_take_profit_sell_ratio=0.50,
    )
    risk = RiskEngine(config)

    pos = Position(
        symbol="NVDA",
        qty=100.0,
        avg_entry_price=100.0,
        current_price=125.0,  # +25% > +20%
        stop_price=90.0,
        initial_stop_price=90.0,
        highest_price=125.0,
    )

    dec = risk.evaluate_partial_take_profit(pos, price=125.0, now=NOW)
    assert dec is not None
    assert dec.approved
    assert dec.intent is not None
    assert dec.intent.token is not None
    assert dec.intent.side is Side.SELL
    assert dec.intent.symbol == "NVDA"
    assert dec.intent.qty == 50.0  # 50% de 100 actions
    assert dec.reason == "partial_take_profit"

    # Simulation de l'exécution du fill et passage à Break-Even
    pos.qty -= dec.intent.qty
    pos.partial_tp_taken = True
    pos.stop_price = max(pos.stop_price or 0.0, pos.avg_entry_price)
    assert pos.stop_price == 100.0
    assert pos.partial_tp_taken is True

    # Nouvelle vérification à un prix encore plus haut (+35%) : ne doit pas réémettre
    pos.current_price = 135.0
    dec_again = risk.evaluate_partial_take_profit(pos, price=135.0, now=NOW)
    assert dec_again is None

    # Vérification de la persistance de l'état
    state_dump = pos.stop_state()
    assert state_dump["partial_tp_taken"] is True
    assert state_dump["stop_price"] == 100.0

    pos_restored = Position(
        symbol="NVDA", qty=50.0, avg_entry_price=100.0, current_price=135.0
    )
    pos_restored.apply_stop_state(state_dump)
    assert pos_restored.partial_tp_taken is True
    assert pos_restored.stop_price == 100.0


def test_overweight_trimming() -> None:
    """Vérifie l'écrémage d'une ligne offensive dépassant le plafond unitaire maximal."""
    config = RiskConfig(
        max_single_position_weight=0.25,
        max_position_pct=0.20,
        safe_haven_symbols=["BRK.B"],
    )
    risk = RiskEngine(config)

    account = AccountState(
        equity=100_000.0,
        cash=70_000.0,
        buying_power=70_000.0,
        positions={},
    )

    # Position offensive pesant 30 000 $ (30% > 25%)
    pos_nvda = Position(
        symbol="NVDA",
        qty=300.0,
        avg_entry_price=80.0,
        current_price=100.0,
    )
    account.positions["NVDA"] = pos_nvda

    dec = risk.evaluate_overweight_trim(pos_nvda, price=100.0, account=account, now=NOW)
    assert dec is not None
    assert dec.approved
    assert dec.intent is not None
    assert dec.intent.token is not None
    assert dec.intent.symbol == "NVDA"
    assert dec.intent.side is Side.SELL
    # Cible : 20% de 100k = 20k$ (soit 200 actions). Excédent à vendre : 100 actions.
    assert pytest.approx(dec.intent.qty, rel=1e-3) == 100.0
    assert dec.reason == "position_trim_overweight"

    # Les safe havens ne doivent pas être écrémés par cette règle
    pos_sh = Position(
        symbol="BRK.B",
        qty=300.0,
        avg_entry_price=80.0,
        current_price=100.0,
    )
    account.positions["BRK.B"] = pos_sh
    assert (
        risk.evaluate_overweight_trim(pos_sh, price=100.0, account=account, now=NOW) is None
    )


def test_equity_hard_floor_breach() -> None:
    """Vérifie l'arrêt permanent et le refus des ordres BUY lors d'une brèche du plancher d'équité."""
    config = RiskConfig(
        equity_hard_floor_usd=80_000.0,
    )
    risk = RiskEngine(config)

    # Équité tombée à 75 000 $ <= 80 000 $
    account = AccountState(
        equity=75_000.0,
        cash=25_000.0,
        buying_power=25_000.0,
        positions={
            "NVDA": Position(
                symbol="NVDA",
                qty=500.0,
                avg_entry_price=100.0,
                current_price=100.0,
            )
        },
    )
    risk.start_session(account, session_date=SESSION)
    halt = risk.update_risk_state(account, session_date=SESSION)

    assert halt is not None
    assert halt.scope is HaltScope.PERMANENT
    assert account.halted is True
    assert account.halt_scope is HaltScope.PERMANENT
    assert "plancher de sécurité" in (account.halt_reason or "")

    # Tout nouvel achat doit être rejeté
    entry_dec = risk.evaluate_entry(
        symbol="AMD",
        price=100.0,
        atr=5.0,
        account=account,
        now=NOW,
        session_date=SESSION,
    )
    assert not entry_dec.approved
    assert entry_dec.code is RejectionCode.EQUITY_FLOOR_BREACHED

    # La liquidation globale doit être approuvée par le RiskEngine
    flattens = risk.flatten_decisions(
        account, prices={"NVDA": 100.0}, now=NOW, reason="circuit_breaker"
    )
    assert len(flattens) == 1
    assert flattens[0].approved
    assert flattens[0].intent is not None
    assert flattens[0].intent.token is not None
    assert flattens[0].intent.symbol == "NVDA"
    assert flattens[0].intent.qty == 500.0


def test_cmd_withdraw_symbol_and_pct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Vérifie la commande CLI withdraw --symbol --pct."""
    state_file = tmp_path / "state.json"
    store = StateStore(state_file)
    store.save(PersistedState())

    broker = FakeBroker(
        equity=100_000.0,
        cash=20_000.0,
        buying_power=20_000.0,
        positions={
            "NVDA": Position(
                symbol="NVDA",
                qty=50.0,
                avg_entry_price=100.0,
                current_price=100.0,
                stop_price=90.0,
            )
        },
    )

    settings = Settings(state_dir=tmp_path, dry_run=False)
    monkeypatch.setattr("trading_bot.main._settings_from_args", lambda args: settings)
    monkeypatch.setattr("trading_bot.main.AlpacaPaperClient", lambda s: broker)
    monkeypatch.setattr("trading_bot.main.StateStore", lambda p: store)

    args = argparse.Namespace(
        symbol="NVDA",
        pct=50.0,
        amount=None,
        free_cash=None,
        dry_run=False,
        log_level="INFO",
    )
    exit_code = cmd_withdraw(args)
    assert exit_code == 0
    assert (
        len(broker.orders) == 2
    )  # vente de 25 titres + remise du stop pour les 25 restants
    sell_order = broker.orders[0]
    assert sell_order.symbol == "NVDA"
    assert sell_order.side is Side.SELL
    assert sell_order.qty == 25.0


def test_cmd_withdraw_free_cash_updates_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Vérifie la commande CLI withdraw --free-cash qui vend les safe havens et met à jour .env."""
    env_file = tmp_path / ".env"
    env_file.write_text("CASH_BUFFER_PCT=0.10\n", encoding="utf-8")

    state_file = tmp_path / "state.json"
    store = StateStore(state_file)
    store.save(PersistedState())

    broker = FakeBroker(
        equity=100_000.0,
        cash=5_000.0,  # besoin de 10 000 $ au total -> 5 000 $ manquants
        buying_power=5_000.0,
        positions={
            "BRK.B": Position(
                symbol="BRK.B",
                qty=50.0,
                avg_entry_price=100.0,
                current_price=100.0,
            ),
            "NVDA": Position(
                symbol="NVDA",
                qty=20.0,
                avg_entry_price=80.0,
                current_price=100.0,
                stop_price=90.0,
            ),
        },
    )

    settings = Settings(
        state_dir=tmp_path,
        safe_haven_symbols=["BRK.B"],
        dry_run=False,
    )

    monkeypatch.setattr("trading_bot.main._settings_from_args", lambda args: settings)
    monkeypatch.setattr("trading_bot.main.AlpacaPaperClient", lambda s: broker)
    monkeypatch.setattr("trading_bot.main.StateStore", lambda p: store)
    monkeypatch.setattr("trading_bot.main.PROJECT_ROOT", tmp_path)

    args = argparse.Namespace(
        symbol=None,
        pct=None,
        amount=None,
        free_cash=10_000.0,
        dry_run=False,
        log_level="INFO",
    )
    exit_code = cmd_withdraw(args)
    assert exit_code == 0
    # On devait récupérer 5 000 $. BRK.B à 100$ : 50 titres vendus = 5 000 $.
    assert len(broker.orders) >= 1
    assert broker.orders[0].symbol == "BRK.B"
    assert broker.orders[0].qty == 50.0

    # Vérification que .env a été mis à jour avec le plancher
    env_content = env_file.read_text(encoding="utf-8")
    assert "WITHDRAWABLE_CASH_FLOOR_USD=10000.00" in env_content


def _withdrawal_settings(tmp_path: Path) -> Settings:
    """Configuration de test isolée du .env réel."""
    return Settings(
        _env_file=None,
        symbols=["AAPL"],
        state_dir=tmp_path / "state",
        dry_run=False,
        safe_haven_enabled=False,
    )


def test_withdrawal_refuses_sale_without_confirmed_cancellation(tmp_path: Path) -> None:
    """Pas de vente manuelle tant que l'annulation des stops n'est pas confirmée.

    Une jambe de stop survivante peut vendre à découvert après la vente : le
    chemin de retrait doit donc échouer fermé, exactement comme le cycle de
    trading quand il liquide une position.
    """
    from trading_bot.execution.alpaca_client import BrokerError
    from trading_bot.withdrawal import WithdrawalService

    class StubbornBroker(FakeBroker):
        def cancel_open_orders(self, symbol: str) -> int:
            raise BrokerError("annulation impossible")

    settings = _withdrawal_settings(tmp_path)
    broker = StubbornBroker(
        positions={
            "AAPL": Position(
                symbol="AAPL", qty=10.0, avg_entry_price=100.0, current_price=100.0
            )
        }
    )
    service = WithdrawalService(
        settings,
        broker,
        state_store=StateStore(settings.state_dir / "state.json"),
        env_path=tmp_path / ".env",
    )

    result = service.withdraw_symbol("AAPL", pct=50)

    assert result.fills == []
    assert broker.orders == []
    assert any("annulation" in message for message in result.errors)


def test_manual_withdrawal_is_audited(tmp_path: Path) -> None:
    """Chaque retrait laisse une trace dans audit.jsonl, comme le reste du bot."""
    from trading_bot.main import AuditLog
    from trading_bot.withdrawal import WithdrawalService

    settings = _withdrawal_settings(tmp_path)
    broker = FakeBroker(
        positions={
            "AAPL": Position(
                symbol="AAPL",
                qty=10.0,
                avg_entry_price=100.0,
                current_price=100.0,
                stop_price=90.0,
            )
        }
    )
    audit = AuditLog(settings.state_dir / "audit.jsonl")
    service = WithdrawalService(
        settings,
        broker,
        state_store=StateStore(settings.state_dir / "state.json"),
        audit=audit,
        env_path=tmp_path / ".env",
    )

    result = service.withdraw_symbol("AAPL", pct=50)

    assert result.fills, "la vente doit avoir lieu"
    events = (settings.state_dir / "audit.jsonl").read_text(encoding="utf-8")
    assert "risk_decision" in events
    assert '"withdrawal"' in events
    assert "cancel_open_orders" in events


def test_cmd_rebase_recenters_equity_after_external_cash_movement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un retrait fait chez le courtier ne doit pas ressembler à une perte de 20 %."""
    from trading_bot.main import cmd_rebase

    state_file = tmp_path / "state.json"
    store = StateStore(state_file)
    store.save(
        PersistedState(
            trading_day="2024-01-10",
            day_start_equity=100_000.0,
            peak_equity=100_000.0,
            halted=True,
            halt_scope="permanent",
            halt_reason="drawdown -20.00% au-delà de la limite -15.00%",
        )
    )
    broker = FakeBroker(equity=80_000.0, cash=80_000.0)
    settings = Settings(_env_file=None, symbols=["AAPL"], state_dir=tmp_path, dry_run=False)
    monkeypatch.setattr("trading_bot.main._settings_from_args", lambda args: settings)
    monkeypatch.setattr("trading_bot.main.AlpacaPaperClient", lambda s: broker)
    monkeypatch.setattr("trading_bot.main.StateStore", lambda p: store)

    args = argparse.Namespace(
        reason="retrait de 20 000 $ vers mon compte", dry_run=False, log_level="INFO"
    )
    assert cmd_rebase(args) == 0

    state = store.load()
    assert state.peak_equity == pytest.approx(80_000.0)
    assert state.day_start_equity == pytest.approx(80_000.0)
    assert state.halted is False
    assert state.halt_scope is None
    audit = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert "equity_rebase" in audit
    assert "retrait de 20 000" in audit


def test_cmd_rebase_requires_a_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ré-ancrer le pic est une décision humaine : elle se justifie ou n'a pas lieu."""
    from trading_bot.main import cmd_rebase

    settings = Settings(_env_file=None, symbols=["AAPL"], state_dir=tmp_path, dry_run=False)
    monkeypatch.setattr("trading_bot.main._settings_from_args", lambda args: settings)

    args = argparse.Namespace(reason="   ", dry_run=False, log_level="INFO")
    assert cmd_rebase(args) == 2
