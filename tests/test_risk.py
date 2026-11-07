"""Tests du moteur de risque : dimensionnement, stops, jetons, breakers."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest

from trading_bot.config import RiskConfig
from trading_bot.risk.engine import (
    AccountState,
    ApprovalToken,
    HaltScope,
    OrderIntent,
    Position,
    RejectionCode,
    RiskEngine,
    RiskError,
    Side,
)

NOW = datetime(2024, 1, 10, 15, 0, tzinfo=UTC)
SESSION = date(2024, 1, 10)


def make_account(equity: float = 100_000.0, cash: float | None = None) -> AccountState:
    """Compte paper de test, sans position."""
    available = equity if cash is None else cash
    return AccountState(equity=equity, cash=available, buying_power=available)


def approve(
    engine: RiskEngine,
    account: AccountState,
    *,
    price: float = 50.0,
    atr: float = 2.0,
    symbol: str = "AAPL",
) -> object:
    """Raccourci d'approbation d'entrée pour les tests."""
    return engine.evaluate_entry(
        symbol=symbol,
        price=price,
        atr=atr,
        account=account,
        now=NOW,
        session_date=SESSION,
    )


def test_position_sized_by_risk_budget() -> None:
    engine = RiskEngine()
    account = make_account()
    decision = approve(engine, account, price=50.0, atr=2.0)
    assert decision.approved
    assert decision.intent is not None
    # risque 1 % = 1000 ; stop = 2 ATR = 4 ; 1000 / 4 = 250 actions
    assert decision.intent.qty == pytest.approx(250.0)
    assert decision.intent.stop_price == pytest.approx(46.0)
    assert decision.metrics["risk_amount"] == pytest.approx(1000.0)


def test_position_capped_by_max_position_pct() -> None:
    engine = RiskEngine()
    account = make_account()
    decision = approve(engine, account, price=100.0, atr=0.5)
    assert decision.approved
    assert decision.intent is not None
    # 20 % de 100 000 = 20 000 -> 200 actions, malgré un budget de risque de 1000
    assert decision.intent.qty == pytest.approx(200.0)
    assert decision.metrics["qty_by_notional"] == pytest.approx(200.0)


def test_integer_quantities_when_fractional_disabled() -> None:
    engine = RiskEngine(RiskConfig(fractional_shares=False, qty_decimals=0))
    decision = approve(engine, make_account(), price=50.0, atr=1.3)
    assert decision.approved
    assert decision.intent is not None
    assert decision.intent.qty == int(decision.intent.qty)
    assert decision.intent.qty == 384


def test_fractional_quantities_are_floored() -> None:
    engine = RiskEngine()
    decision = approve(engine, make_account(), price=33.33, atr=1.11)
    assert decision.approved
    assert decision.intent is not None
    qty = decision.intent.qty
    assert qty == pytest.approx(round(qty, 4))
    assert qty <= 1000.0 / (1.11 * 2.0)


def test_rejects_duplicate_position() -> None:
    engine = RiskEngine()
    account = make_account()
    account.positions["AAPL"] = Position(
        symbol="AAPL", qty=10, avg_entry_price=100, current_price=101
    )
    decision = approve(engine, account)
    assert not decision.approved
    assert decision.code is RejectionCode.POSITION_ALREADY_OPEN


def test_rejects_when_max_positions_reached() -> None:
    engine = RiskEngine(RiskConfig(max_open_positions=2))
    account = make_account()
    for symbol in ("AAA", "BBB"):
        account.positions[symbol] = Position(
            symbol=symbol, qty=1, avg_entry_price=100, current_price=100
        )
    decision = approve(engine, account, symbol="CCC")
    assert not decision.approved
    assert decision.code is RejectionCode.MAX_POSITIONS_REACHED


def test_rejects_below_min_notional() -> None:
    engine = RiskEngine(RiskConfig(min_order_notional=5_000.0))
    decision = approve(engine, make_account(equity=20_000.0), price=100.0, atr=2.0)
    assert not decision.approved
    assert decision.code is RejectionCode.BELOW_MIN_NOTIONAL


def test_rejects_without_buying_power() -> None:
    engine = RiskEngine()
    account = make_account(equity=100_000.0, cash=0.0)
    decision = approve(engine, account)
    assert not decision.approved
    assert decision.code is RejectionCode.INSUFFICIENT_BUYING_POWER


@pytest.mark.parametrize("price,atr", [(0.0, 2.0), (-5.0, 2.0), (100.0, 0.0)])
def test_rejects_invalid_inputs(price: float, atr: float) -> None:
    decision = approve(RiskEngine(), make_account(), price=price, atr=atr)
    assert not decision.approved
    assert decision.code is RejectionCode.INVALID_REQUEST


def test_naive_timestamp_rejected() -> None:
    with pytest.raises(RiskError):
        RiskEngine().evaluate_entry(
            symbol="AAPL",
            price=50.0,
            atr=2.0,
            account=make_account(),
            now=datetime(2024, 1, 10, 15, 0),
            session_date=SESSION,
        )


def test_daily_circuit_breaker_halts_and_resets_next_day() -> None:
    engine = RiskEngine()
    account = make_account(equity=96_000.0)
    account.trading_day = SESSION
    account.day_start_equity = 100_000.0
    account.peak_equity = 100_000.0

    event = engine.update_risk_state(account, session_date=SESSION)
    assert event is not None
    assert event.scope is HaltScope.DAY
    assert account.halted
    assert event.daily_pnl_pct == pytest.approx(-0.04)

    decision = approve(engine, account)
    assert not decision.approved
    assert decision.code is RejectionCode.HALTED

    engine.start_session(account, session_date=date(2024, 1, 11))
    assert not account.halted
    assert account.day_start_equity == pytest.approx(96_000.0)


def test_max_drawdown_halts_permanently() -> None:
    engine = RiskEngine()
    account = make_account(equity=160_000.0)
    account.trading_day = SESSION
    account.day_start_equity = 160_000.0
    account.peak_equity = 200_000.0

    event = engine.update_risk_state(account, session_date=SESSION)
    assert event is not None
    assert event.scope is HaltScope.PERMANENT
    assert account.halted

    engine.start_session(account, session_date=date(2024, 1, 11))
    assert account.halted  # un drawdown excessif ne se réarme pas tout seul


def test_permanent_halt_requires_explicit_reason() -> None:
    engine = RiskEngine()
    account = make_account(equity=160_000.0)
    account.trading_day = SESSION
    account.day_start_equity = 160_000.0
    account.peak_equity = 200_000.0
    engine.update_risk_state(account, session_date=SESSION)
    with pytest.raises(RiskError):
        engine.reset_permanent_halt(account, reason="  ")
    engine.reset_permanent_halt(account, reason="inspection manuelle")
    assert not account.halted


def test_exit_remains_allowed_while_halted() -> None:
    engine = RiskEngine()
    position = Position(
        symbol="AAPL", qty=10, avg_entry_price=100, current_price=95, stop_price=96
    )
    account = make_account()
    account.positions["AAPL"] = position
    account.halted = True
    account.halt_reason = "test"
    decision = engine.evaluate_exit(
        position=position, price=95.0, now=NOW, reason="circuit_breaker"
    )
    assert decision.approved
    assert decision.intent is not None
    assert decision.intent.side is Side.SELL


def test_stop_exit_triggered_below_stop() -> None:
    engine = RiskEngine()
    position = Position(
        symbol="AAPL",
        qty=10,
        avg_entry_price=100,
        current_price=95,
        stop_price=96,
        initial_stop_price=96,
    )
    assert engine.evaluate_stop_exit(position, price=97.0, now=NOW) is None
    decision = engine.evaluate_stop_exit(position, price=95.0, now=NOW)
    assert decision is not None
    assert decision.reason == "stop_loss"
    assert decision.intent is not None
    assert decision.intent.qty == 10


def test_trailing_stop_ratchets_only_up() -> None:
    engine = RiskEngine(RiskConfig(trailing_atr_multiple=3.0, trailing_activation_r=1.0))
    account = make_account()
    position = Position(
        symbol="AAPL",
        qty=100,
        avg_entry_price=100.0,
        current_price=110.0,
        stop_price=96.0,
        initial_stop_price=96.0,
    )
    account.positions["AAPL"] = position

    update = engine.update_trailing_stop(position, price=110.0, atr=2.0)
    assert update is not None
    assert update.new_stop == pytest.approx(104.0)
    assert engine.apply_trailing_stop(account, update)
    assert position.stop_price == pytest.approx(104.0)

    assert engine.update_trailing_stop(position, price=108.0, atr=2.0) is None


def test_trailing_moves_to_breakeven_at_one_r() -> None:
    engine = RiskEngine(RiskConfig(trailing_atr_multiple=3.0, trailing_activation_r=1.0))
    position = Position(
        symbol="AAPL",
        qty=100,
        avg_entry_price=100.0,
        current_price=104.5,
        stop_price=96.0,
        initial_stop_price=96.0,
    )
    update = engine.update_trailing_stop(position, price=104.5, atr=2.0)
    assert update is not None
    assert update.new_stop == pytest.approx(100.0)


def test_trailing_inactive_below_activation_threshold() -> None:
    engine = RiskEngine()
    position = Position(
        symbol="AAPL",
        qty=100,
        avg_entry_price=100.0,
        current_price=103.0,
        stop_price=96.0,
        initial_stop_price=96.0,
    )
    assert engine.update_trailing_stop(position, price=103.0, atr=2.0) is None


def test_trailing_disabled_by_zero_multiple() -> None:
    engine = RiskEngine(RiskConfig(trailing_atr_multiple=0.0))
    position = Position(
        symbol="AAPL",
        qty=100,
        avg_entry_price=100.0,
        current_price=110.0,
        stop_price=96.0,
        initial_stop_price=96.0,
    )
    assert engine.update_trailing_stop(position, price=110.0, atr=2.0) is None


def test_intent_rejects_quantity_tampering() -> None:
    decision = approve(RiskEngine(), make_account())
    assert decision.intent is not None
    intent = decision.intent
    with pytest.raises(ValueError):
        OrderIntent(
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty * 2,
            token=intent.token,
            stop_price=intent.stop_price,
        )


def test_intent_rejects_forged_token() -> None:
    decision = approve(RiskEngine(), make_account())
    assert decision.intent is not None
    real = decision.intent
    forged = ApprovalToken(
        decision_id="forged",
        symbol=real.symbol,
        side=real.side,
        qty=real.qty,
        issued_at=NOW,
        expires_at=NOW + timedelta(minutes=5),
        issuer=object(),
    )
    with pytest.raises(ValueError, match="RiskEngine"):
        OrderIntent(
            symbol=real.symbol,
            side=real.side,
            qty=real.qty,
            token=forged,
        )


def test_token_expires() -> None:
    engine = RiskEngine(RiskConfig(order_ttl_seconds=30))
    decision = approve(engine, make_account())
    assert decision.intent is not None
    token = decision.intent.token
    assert token.is_valid(NOW + timedelta(seconds=29))
    assert not token.is_valid(NOW + timedelta(seconds=30))
    assert not token.is_valid(NOW - timedelta(seconds=1))


def test_decision_id_is_deterministic_for_same_inputs() -> None:
    first = approve(RiskEngine(), make_account())
    second = approve(RiskEngine(), make_account())
    assert first.decision_id == second.decision_id


def test_flatten_decisions_cover_all_positions() -> None:
    engine = RiskEngine()
    account = make_account()
    for symbol, price in (("AAA", 10.0), ("BBB", 20.0)):
        account.positions[symbol] = Position(
            symbol=symbol, qty=5, avg_entry_price=price, current_price=price
        )
    decisions = engine.flatten_decisions(
        account,
        prices={"AAA": 10.0, "BBB": 20.0},
        now=NOW,
        reason="circuit_breaker",
    )
    assert len(decisions) == 2
    assert all(decision.approved for decision in decisions)
    assert {decision.intent.symbol for decision in decisions if decision.intent} == {
        "AAA",
        "BBB",
    }


def test_rejections_are_audited() -> None:
    engine = RiskEngine()
    account = make_account()
    account.halted = True
    account.halt_reason = "test"
    decision = approve(engine, account)
    assert not decision.approved
    assert len(engine.audit) == 1
    assert engine.audit[0].code is RejectionCode.HALTED


def test_get_position_metrics() -> None:
    position = Position(
        symbol="AAPL",
        qty=10,
        avg_entry_price=100.0,
        current_price=110.0,
        stop_price=96.0,
        initial_stop_price=96.0,
    )
    assert position.market_value == pytest.approx(1100.0)
    assert position.unrealized_pnl == pytest.approx(100.0)
    assert position.unrealized_pnl_pct == pytest.approx(0.10)
    assert position.risk_per_share == pytest.approx(4.0)
    assert position.r_multiple == pytest.approx(2.5)


def test_account_daily_and_drawdown_metrics() -> None:
    account = make_account(equity=95_000.0)
    account.day_start_equity = 100_000.0
    account.peak_equity = 110_000.0
    assert account.daily_pnl_pct == pytest.approx(-0.05)
    assert account.drawdown_pct == pytest.approx(95_000 / 110_000 - 1)


def test_position_state_roundtrip() -> None:
    position = Position(
        symbol="AAPL",
        qty=10,
        avg_entry_price=100.0,
        current_price=105.0,
        stop_price=97.5,
        initial_stop_price=96.0,
        highest_price=106.0,
    )
    restored = Position(symbol="AAPL", qty=10, avg_entry_price=100.0, current_price=101.0)
    restored.apply_stop_state(position.stop_state())
    assert restored.stop_price == pytest.approx(97.5)
    assert restored.initial_stop_price == pytest.approx(96.0)
    assert restored.highest_price == pytest.approx(106.0)
    assert replace(restored, current_price=107.0).current_price == pytest.approx(107.0)


def test_evaluate_stop_replace_seals_stop_order() -> None:
    position = Position(
        symbol="AAPL",
        qty=10,
        avg_entry_price=100.0,
        current_price=110.0,
        stop_price=104.5,
        initial_stop_price=96.0,
    )
    decision = RiskEngine().evaluate_stop_replace(position, price=110.0, now=NOW)
    assert decision is not None and decision.approved
    intent = decision.intent
    assert intent is not None
    assert intent.symbol == "AAPL"
    assert intent.side.value == "sell"
    assert intent.order_type.value == "stop"
    assert intent.stop_price == pytest.approx(104.5)
    assert intent.qty == pytest.approx(10.0)
    assert intent.is_executable(NOW)


def test_evaluate_stop_replace_rejects_invalid_state() -> None:
    engine = RiskEngine()
    # Stop au-dessus ou égal au prix courant : aucun ordre scellé.
    high_stop = Position(
        symbol="AAPL",
        qty=10,
        avg_entry_price=100.0,
        current_price=100.0,
        stop_price=101.0,
    )
    assert engine.evaluate_stop_replace(high_stop, price=100.0, now=NOW) is None
    # Position sans stop : rien à matérialiser.
    no_stop = Position(
        symbol="AAPL",
        qty=10,
        avg_entry_price=100.0,
        current_price=100.0,
    )
    assert engine.evaluate_stop_replace(no_stop, price=100.0, now=NOW) is None


def test_rank_candidates_orders_by_score_desc() -> None:
    engine = RiskEngine()
    ranked = engine.rank_candidates({"AAPL": 1.2, "MSFT": 3.4, "NVDA": 2.0})
    assert ranked == ["MSFT", "NVDA", "AAPL"]
    # Scores non finis relégués en fin, tri alphabétique entre eux.
    ranked = engine.rank_candidates({"AAA": float("nan"), "ZZZ": 1.0, "BBB": float("nan")})
    assert ranked == ["ZZZ", "AAA", "BBB"]
    assert engine.rank_candidates({}) == []


def test_bull_regime_loses_and_tightens_caps() -> None:
    engine = RiskEngine()
    assert engine.bull_market is False
    assert engine.effective_position_cap_pct() == pytest.approx(0.20)
    assert engine.effective_gross_cap_pct() == pytest.approx(0.80)

    engine.set_market_regime(bull_market=True)
    assert engine.effective_position_cap_pct() == pytest.approx(0.30)
    assert engine.effective_gross_cap_pct() == pytest.approx(0.90)

    engine.set_market_regime(bull_market=False)
    assert engine.effective_position_cap_pct() == pytest.approx(0.20)


def test_bull_regime_raises_entry_notional_cap() -> None:
    config = RiskConfig(atr_stop_multiple=2.0)
    engine = RiskEngine(config)
    account = make_account()
    decision = approve(engine, account, price=100.0, atr=1.0)
    assert decision.intent is not None
    base_qty = decision.intent.qty

    engine.set_market_regime(bull_market=True)
    decision_bull = approve(engine, make_account(), price=100.0, atr=1.0)
    assert decision_bull.intent is not None
    # Plafond de position relevé de 20 % à 30 % : quantité 1,5x (plafond
    # notionnel contraignant, l'ATR volontairement large neutralise le
    # budget de risque).
    assert decision_bull.intent.qty == pytest.approx(base_qty * 1.5)
    # Le régime est journalisé dans les métriques d'audit.
    assert decision_bull.metrics["bull_market"] == 1.0
    assert decision_bull.metrics["gross_cap"] == pytest.approx(0.90)


def test_bull_cap_validation_rejects_incoherent_profile() -> None:
    with pytest.raises(ValueError):
        RiskConfig(bull_max_position_pct=0.5, bull_gross_exposure_pct=0.4)


def test_custom_shield_symbol_is_treated_as_defensive() -> None:
    """Le symbole du Bouclier vient de la configuration, pas d'une constante « SGOV »."""
    engine = RiskEngine(RiskConfig(shield_symbol="SHV"))
    position = Position(
        symbol="SHV",
        qty=100.0,
        avg_entry_price=100.0,
        current_price=130.0,
        stop_price=90.0,
        initial_stop_price=90.0,
        highest_price=130.0,
    )
    account = AccountState(
        equity=10_000.0, cash=0.0, buying_power=0.0, positions={"SHV": position}
    )

    assert engine.update_trailing_stop(position, price=130.0, atr=1.0) is None
    assert engine.evaluate_partial_take_profit(position, price=130.0, now=NOW) is None
    assert (
        engine.evaluate_overweight_trim(position, price=130.0, account=account, now=NOW)
        is None
    )


def test_custom_shield_symbol_does_not_block_offensive_position_limit() -> None:
    """Le Bouclier ne consomme pas un emplacement de la limite de positions offensives."""
    engine = RiskEngine(RiskConfig(shield_symbol="SHV", max_open_positions=1))
    account = make_account()
    account.positions["SHV"] = Position(
        symbol="SHV", qty=100.0, avg_entry_price=100.0, current_price=100.0
    )
    decision = engine.evaluate_entry(
        symbol="AAPL",
        price=50.0,
        atr=2.0,
        account=account,
        now=NOW,
        session_date=SESSION,
    )
    assert decision.approved


def test_shield_allocation_buys_up_to_target_weight() -> None:
    """Le Bouclier se finance jusqu'à son poids cible, sans stop ni trailing."""
    engine = RiskEngine(RiskConfig(shield_symbol="SHV", cash_buffer_pct=0.10))
    account = make_account(100_000.0)

    decision = engine.evaluate_shield_allocation(
        account, symbol="SHV", price=100.0, target_weight=0.70, now=NOW
    )

    assert decision.approved
    assert decision.intent is not None
    assert decision.intent.qty == pytest.approx(700.0)
    assert decision.intent.stop_price is None
    assert decision.intent.reason == "shield_allocation"


def test_shield_allocation_tops_up_only_the_gap() -> None:
    """Une poche déjà partiellement financée ne reçoit que le complément."""
    engine = RiskEngine(RiskConfig(shield_symbol="SHV", cash_buffer_pct=0.10))
    account = make_account(100_000.0, cash=70_000.0)
    account.positions["SHV"] = Position(
        symbol="SHV", qty=300.0, avg_entry_price=100.0, current_price=100.0
    )

    decision = engine.evaluate_shield_allocation(
        account, symbol="SHV", price=100.0, target_weight=0.70, now=NOW
    )

    assert decision.approved
    assert decision.intent is not None
    assert decision.intent.qty == pytest.approx(400.0)


def test_shield_allocation_is_silent_when_already_funded() -> None:
    """Aucun ordre quand la poche a atteint son poids cible."""
    engine = RiskEngine(RiskConfig(shield_symbol="SHV"))
    account = make_account(100_000.0, cash=30_000.0)
    account.positions["SHV"] = Position(
        symbol="SHV", qty=700.0, avg_entry_price=100.0, current_price=100.0
    )

    assert (
        engine.evaluate_shield_allocation(
            account, symbol="SHV", price=100.0, target_weight=0.70, now=NOW
        )
        is None
    )


def test_shield_allocation_respects_the_cash_buffer() -> None:
    """Le plancher de cash reste sanctuarisé : le Bouclier ne boit pas la réserve."""
    engine = RiskEngine(RiskConfig(shield_symbol="SHV", cash_buffer_pct=0.10))
    account = make_account(100_000.0, cash=20_000.0)

    decision = engine.evaluate_shield_allocation(
        account, symbol="SHV", price=100.0, target_weight=0.70, now=NOW
    )

    assert decision.approved
    assert decision.intent is not None
    # 20 000 de cash - 10 000 de buffer (10 % de 100 000) = 10 000 investissables
    assert decision.intent.qty == pytest.approx(100.0)


def test_shield_allocation_ignores_tiny_gaps() -> None:
    """Un écart de quelques dollars ne justifie pas un ordre à chaque séance."""
    engine = RiskEngine(RiskConfig(shield_symbol="SHV"))
    account = make_account(100_000.0, cash=30_100.0)
    account.positions["SHV"] = Position(
        symbol="SHV", qty=699.0, avg_entry_price=100.0, current_price=100.0
    )

    # Écart de 100 $ sur 100 000 $ d'équité : sous le seuil de réajustement.
    assert (
        engine.evaluate_shield_allocation(
            account, symbol="SHV", price=100.0, target_weight=0.70, now=NOW
        )
        is None
    )


def test_cash_sweep_never_targets_the_shield_symbol() -> None:
    """Même listé dans les safe havens, le Bouclier garde son financement propre."""
    engine = RiskEngine(
        RiskConfig(
            shield_symbol="SHV",
            safe_haven_symbols=["SHV", "KO"],
            cash_buffer_pct=0.10,
            max_safe_haven_total_pct=0.50,
        )
    )
    account = make_account(100_000.0)
    prices = {"SHV": 100.0, "KO": 50.0}

    decisions = engine.evaluate_cash_sweep(account, prices, now=NOW)

    assert decisions, "la poche de repli doit tout de même être financée"
    assert {decision.intent.symbol for decision in decisions if decision.intent} == {"KO"}
