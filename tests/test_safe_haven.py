"""Tests de la poche de repli défensive : Cash Sweep, Auto-Unwind et BEAR_REGIME_VETO.

Vérifie l'invariant absolu : seul RiskEngine émet des ApprovalToken valides
pour sceller les ordres d'achat (cash sweep) ou de vente (auto-unwind).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from trading_bot.config import RiskConfig, Settings
from trading_bot.data.news_sentiment import NewsAssessment
from trading_bot.execution.alpaca_client import OrderReceipt
from trading_bot.main import TradingBot
from trading_bot.risk.engine import (
    AccountState,
    Position,
    RejectionCode,
    RiskEngine,
    Side,
)
from trading_bot.strategies.base import SignalType, Strategy

NOW = datetime(2024, 1, 10, 15, 0, tzinfo=UTC)
SESSION = date(2024, 1, 10)


class FakeBroker:
    """Courtier de test en mémoire."""

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

    def is_market_open(self) -> bool:
        return True

    def market_status(self) -> Any:
        return type("Status", (), {"is_open": True, "closure": None, "next_open": None})()

    def get_latest_prices(self, symbols: Any) -> dict[str, float]:
        return {sym: 100.0 for sym in symbols}

    def cancel_open_orders(self, symbol: str) -> int:
        return 0

    def submit(self, intent: Any, now: datetime | None = None) -> OrderReceipt:
        self.orders.append(intent)
        receipt = OrderReceipt(
            order_id=f"ord-{len(self.orders)}",
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            filled_qty=intent.qty,
            avg_fill_price=100.0,
            status="filled",
            submitted_at=now or NOW,
        )
        return receipt

    def wait_for_fill(self, order_id: str, timeout_seconds: float = 30.0) -> OrderReceipt:
        matching = [o for o in self.orders if f"ord-{self.orders.index(o) + 1}" == order_id]
        intent = matching[0] if matching else self.orders[-1]
        return OrderReceipt(
            order_id=order_id,
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            filled_qty=intent.qty,
            avg_fill_price=100.0,
            status="filled",
            submitted_at=NOW,
        )


class FakeBars:
    """Source de barres synthétiques."""

    def __init__(self, bars: pd.DataFrame) -> None:
        self._bars = bars

    def fetch_bars(self, symbol: str, **kw: Any) -> pd.DataFrame:
        return self._bars.copy()


class ConstantSignalStrategy(Strategy):
    """Stratégie émettant toujours le même signal."""

    name = "constant_signal"

    def __init__(self, action: SignalType = SignalType.BUY) -> None:
        self._action = action

    @property
    def warmup_bars(self) -> int:
        return 2

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        out = bars.copy()
        out["atr"] = 1.0
        return out

    def compute_signals(self, prepared: pd.DataFrame) -> pd.Series:
        return pd.Series(self._action, index=prepared.index, dtype=object)


def _make_bars(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=len(closes), freq="h", tz="UTC")
    close = pd.Series(closes, index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(len(closes), 1000.0),
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# 1. Configuration & Initialisation
# ---------------------------------------------------------------------------
def test_safe_haven_defaults_and_auto_exclusion() -> None:
    """Les safe havens par défaut sont exclus automatiquement des offensifs."""
    settings = Settings(
        _env_file=None,
        symbols="NVDA,AMD,BRK.B,WMT,KO,SPY",
    )
    assert settings.timeframe == "1Day"
    assert settings.safe_haven_enabled is True
    assert settings.safe_haven_symbols == ["BRK.B", "WMT", "KO"]
    assert settings.risk.cash_buffer_pct == pytest.approx(0.10)
    assert settings.risk.max_safe_haven_total_pct == pytest.approx(0.50)
    assert settings.risk.regime_sma_period == 200

    # SPY et safe havens retirés des symboles offensifs tradables
    assert "SPY" not in settings.symbols
    assert "BRK.B" not in settings.symbols
    assert "WMT" not in settings.symbols
    assert "KO" not in settings.symbols
    assert settings.symbols == ["NVDA", "AMD"]


def test_safe_haven_csv_parsing_and_validation() -> None:
    """La liste safe_haven_symbols accepte un format CSV et rejette les tickers invalides."""
    settings = Settings(
        _env_file=None,
        symbols=["NVDA"],
        safe_haven_symbols="brk.b, wmt, ko;brk.b",
    )
    assert settings.safe_haven_symbols == ["BRK.B", "WMT", "KO"]

    with pytest.raises(ValidationError):
        Settings(_env_file=None, safe_haven_symbols="")

    with pytest.raises(ValidationError):
        Settings(_env_file=None, safe_haven_symbols="INVALID$TICKER")


def test_safe_haven_risk_config_coherence() -> None:
    """cash_buffer_pct + max_safe_haven_total_pct ne peut pas dépasser 1.0."""
    with pytest.raises(ValidationError):
        RiskConfig(cash_buffer_pct=0.60, max_safe_haven_total_pct=0.50)


# ---------------------------------------------------------------------------
# 2. Détection du régime & BEAR_REGIME_VETO
# ---------------------------------------------------------------------------
def test_bear_regime_veto_blocks_offensive_entries() -> None:
    """En régime BEAR (SPY < SMA 200), les achats offensifs sont refusés avec BEAR_REGIME_VETO."""
    engine = RiskEngine(RiskConfig())
    engine.set_market_regime(bull_market=False, bear_market=True)
    assert engine.bear_market is True
    assert engine.bull_market is False

    account = AccountState(equity=100_000.0, cash=50_000.0, buying_power=50_000.0)

    # Entrée offensive sur NVDA refusée
    decision = engine.evaluate_entry(
        symbol="NVDA",
        price=100.0,
        atr=2.0,
        account=account,
        now=NOW,
        session_date=SESSION,
    )
    assert not decision.approved
    assert decision.code is RejectionCode.BEAR_REGIME_VETO
    assert "[BEAR_REGIME_VETO]" in decision.reason

    # Entrée sur safe haven autorisée en régime baissier
    sh_decision = engine.evaluate_entry(
        symbol="BRK.B",
        price=100.0,
        atr=2.0,
        account=account,
        now=NOW,
        session_date=SESSION,
    )
    assert sh_decision.approved
    assert sh_decision.code is RejectionCode.APPROVED


def test_bull_regime_allows_offensive_entries() -> None:
    """En régime BULL (SPY > SMA 200), les achats offensifs sont acceptés."""
    engine = RiskEngine(RiskConfig())
    engine.set_market_regime(bull_market=True, bear_market=False)
    assert engine.bull_market is True
    assert engine.bear_market is False

    account = AccountState(equity=100_000.0, cash=50_000.0, buying_power=50_000.0)
    decision = engine.evaluate_entry(
        symbol="NVDA",
        price=100.0,
        atr=2.0,
        account=account,
        now=NOW,
        session_date=SESSION,
    )
    assert decision.approved
    assert decision.code is RejectionCode.APPROVED


# ---------------------------------------------------------------------------
# 3. Cash Sweep (Réinvestissement défensif)
# ---------------------------------------------------------------------------
def test_cash_sweep_allocates_excess_above_buffer() -> None:
    """Le cash sweep n'alloue que l'excédent au-dessus du buffer de 10 %."""
    engine = RiskEngine(
        RiskConfig(
            safe_haven_enabled=True,
            safe_haven_symbols=["BRK.B", "WMT", "KO"],
            cash_buffer_pct=0.10,
            max_safe_haven_total_pct=0.50,
        )
    )
    # Équité 100 000 $, cash 40 000 $.
    # Buffer requis : 10 % de 100 000 = 10 000 $.
    # Excédent sweepable : 40 000 - 10 000 = 30 000 $.
    # Plafond safe haven (50 %) = 50 000 $.
    # Répartition sur 3 symboles : 30 000 / 3 = 10 000 $ par symbole.
    account = AccountState(equity=100_000.0, cash=40_000.0, buying_power=40_000.0)
    prices = {"BRK.B": 100.0, "WMT": 50.0, "KO": 25.0}

    decisions = engine.evaluate_cash_sweep(account, prices, now=NOW)
    assert len(decisions) == 3

    for dec in decisions:
        assert dec.approved
        assert dec.intent is not None
        assert dec.intent.side is Side.BUY
        assert dec.intent.reason == "cash_sweep"
        # Invariant : seul RiskEngine émet le jeton scellé
        assert dec.intent.token.is_valid(NOW)

    by_sym = {d.intent.symbol: d.intent for d in decisions if d.intent}
    assert by_sym["BRK.B"].qty == pytest.approx(100.0)  # 10 000 / 100
    assert by_sym["WMT"].qty == pytest.approx(200.0)  # 10 000 / 50
    assert by_sym["KO"].qty == pytest.approx(400.0)  # 10 000 / 25


def test_cash_sweep_capped_by_max_safe_haven_pct() -> None:
    """Le cash sweep respecte le plafond global MAX_SAFE_HAVEN_TOTAL_PCT (50 %)."""
    engine = RiskEngine(
        RiskConfig(
            safe_haven_enabled=True,
            safe_haven_symbols=["BRK.B", "WMT"],
            cash_buffer_pct=0.10,
            max_safe_haven_total_pct=0.50,
        )
    )
    # Déjà 40 000 $ de BRK.B détenu sur 100 000 $ d'équité.
    # Capacité restante : 50 000 - 40 000 = 10 000 $.
    pos_brk = Position(
        symbol="BRK.B", qty=400.0, avg_entry_price=100.0, current_price=100.0
    )
    account = AccountState(
        equity=100_000.0,
        cash=50_000.0,
        buying_power=50_000.0,
        positions={"BRK.B": pos_brk},
    )
    prices = {"BRK.B": 100.0, "WMT": 100.0}

    decisions = engine.evaluate_cash_sweep(account, prices, now=NOW)
    assert len(decisions) == 2
    total_swept = sum(d.intent.qty * prices[d.intent.symbol] for d in decisions if d.intent)
    assert total_swept == pytest.approx(10_000.0)


def test_cash_sweep_inactive_when_cash_below_buffer() -> None:
    """Aucun sweep n'est produit si le cash est sous les 10 % requis."""
    engine = RiskEngine(RiskConfig(cash_buffer_pct=0.10))
    account = AccountState(equity=100_000.0, cash=8_000.0, buying_power=8_000.0)
    prices = {"BRK.B": 100.0, "WMT": 100.0, "KO": 100.0}
    decisions = engine.evaluate_cash_sweep(account, prices, now=NOW)
    assert decisions == []


def test_cash_sweep_skips_news_vetoed_safe_haven() -> None:
    """Un symbole safe haven sous le coup d'une alerte news est exclu du sweep."""
    engine = RiskEngine(RiskConfig(news_veto_threshold=-0.60))
    account = AccountState(equity=100_000.0, cash=30_000.0, buying_power=30_000.0)
    prices = {"BRK.B": 100.0, "WMT": 100.0, "KO": 100.0}

    bad_news = NewsAssessment(
        symbol="KO",
        sentiment_score=-0.75,
        confidence=0.90,
        is_critical_threat=False,
        rationale="scandale",
        articles_count=5,
        updated_at=NOW,
    )
    news_map = {"KO": bad_news}
    decisions = engine.evaluate_cash_sweep(account, prices, now=NOW, news=news_map)

    # KO exclu, sweep réparti sur BRK.B et WMT
    symbols = [d.intent.symbol for d in decisions if d.intent]
    assert "KO" not in symbols
    assert "BRK.B" in symbols
    assert "WMT" in symbols


# ---------------------------------------------------------------------------
# 4. Auto-Unwind (Revente défensive pour financer l'offensif)
# ---------------------------------------------------------------------------
def test_auto_unwind_sells_only_necessary_portion() -> None:
    """L'auto-unwind ne vend que la portion nécessaire pour combler le déficit de cash."""
    engine = RiskEngine(
        RiskConfig(
            safe_haven_enabled=True,
            safe_haven_symbols=["BRK.B", "WMT"],
        )
    )
    # 20 000 $ de BRK.B (200 parts à 100 $)
    pos_brk = Position(
        symbol="BRK.B", qty=200.0, avg_entry_price=100.0, current_price=100.0
    )
    account = AccountState(
        equity=100_000.0,
        cash=5_000.0,
        buying_power=5_000.0,
        positions={"BRK.B": pos_brk},
    )
    prices = {"BRK.B": 100.0}

    # On a besoin de 8 000 $ supplémentaires
    decisions = engine.evaluate_safe_haven_unwind(
        account, needed_cash=8_000.0, prices=prices, now=NOW
    )
    assert len(decisions) == 1
    dec = decisions[0]
    assert dec.approved
    assert dec.intent is not None
    assert dec.intent.symbol == "BRK.B"
    assert dec.intent.side is Side.SELL
    assert dec.intent.qty == pytest.approx(80.0)  # 80 * 100 $ = 8 000 $
    assert dec.intent.reason == "safe_haven_unwind"
    assert dec.intent.token.is_valid(NOW)


def test_auto_unwind_across_multiple_safe_havens() -> None:
    """L'auto-unwind vend la totalité du premier puis entame le second si nécessaire."""
    engine = RiskEngine(
        RiskConfig(
            safe_haven_enabled=True,
            safe_haven_symbols=["BRK.B", "WMT"],
        )
    )
    pos_brk = Position(
        symbol="BRK.B", qty=50.0, avg_entry_price=100.0, current_price=100.0
    )  # 5 000 $
    pos_wmt = Position(
        symbol="WMT", qty=100.0, avg_entry_price=50.0, current_price=50.0
    )  # 5 000 $
    account = AccountState(
        equity=100_000.0,
        cash=2_000.0,
        buying_power=2_000.0,
        positions={"BRK.B": pos_brk, "WMT": pos_wmt},
    )
    prices = {"BRK.B": 100.0, "WMT": 50.0}

    # Besoin de 7 000 $ : 5 000 $ de BRK.B (totalité) + 2 000 $ de WMT (partiel)
    decisions = engine.evaluate_safe_haven_unwind(
        account, needed_cash=7_000.0, prices=prices, now=NOW
    )
    assert len(decisions) == 2

    by_sym = {d.intent.symbol: d.intent for d in decisions if d.intent}
    assert by_sym["BRK.B"].qty == pytest.approx(50.0)  # 50 * 100 = 5 000 $
    assert by_sym["WMT"].qty == pytest.approx(40.0)  # 40 * 50 = 2 000 $


# ---------------------------------------------------------------------------
# 5. Cycle end-to-end TradingBot
# ---------------------------------------------------------------------------
def test_bot_cycle_executes_cash_sweep_at_end_of_hourly_cycle(tmp_path: Path) -> None:
    """À la fin d'un cycle horaire, le cash au-dessus du buffer de 10 % est investi en safe havens."""
    settings = Settings(
        _env_file=None,
        symbols=["NVDA"],
        safe_haven_symbols=["BRK.B", "WMT", "KO"],
        safe_haven_enabled=True,
        cash_buffer_pct=0.10,
        max_safe_haven_total_pct=0.50,
        state_dir=tmp_path / "state",
        timeframe="1Hour",
    )
    bars = _make_bars([100.0] * 30)
    broker = FakeBroker(equity=100_000.0, cash=50_000.0, buying_power=50_000.0)

    bot = TradingBot(
        settings,
        ConstantSignalStrategy(SignalType.HOLD),
        broker,  # type: ignore[arg-type]
        bars=FakeBars(bars),  # type: ignore[arg-type]
        clock=lambda: NOW,
    )
    report = bot.run_once()
    assert not report.skipped
    assert report.errors == []

    # Le cash sweep doit avoir généré des ordres d'achat sur les 3 safe havens
    sweep_orders = [o for o in broker.orders if o.reason == "cash_sweep"]
    assert len(sweep_orders) == 3
    symbols_bought = {o.symbol for o in sweep_orders}
    assert symbols_bought == {"BRK.B", "WMT", "KO"}


def test_bot_cycle_executes_auto_unwind_to_fund_offensive_buy(tmp_path: Path) -> None:
    """Un ordre offensif qui manque de cash pur provoque l'unwind automatique des safe havens."""
    settings = Settings(
        _env_file=None,
        symbols=["NVDA"],
        safe_haven_symbols=["BRK.B"],
        safe_haven_enabled=True,
        cash_buffer_pct=0.10,
        state_dir=tmp_path / "state",
        timeframe="1Hour",
        regime_sma_period=20,
    )
    # Barres de SPY en hausse pour assurer le régime haussier
    bars = _make_bars([90.0 + i for i in range(30)])

    # 30 000 $ de BRK.B et seulement 10 500 $ de cash (buffer 10 000 $, donc 500 $ utilisable)
    pos_brk = Position(
        symbol="BRK.B", qty=300.0, avg_entry_price=100.0, current_price=100.0
    )
    broker = FakeBroker(
        equity=100_000.0,
        cash=10_500.0,
        buying_power=10_500.0,
        positions={"BRK.B": pos_brk},
    )

    bot = TradingBot(
        settings,
        ConstantSignalStrategy(SignalType.BUY),
        broker,  # type: ignore[arg-type]
        bars=FakeBars(bars),  # type: ignore[arg-type]
        clock=lambda: NOW,
    )
    report = bot.run_once()
    assert not report.skipped

    # On doit retrouver un ordre de vente auto-unwind sur BRK.B
    unwinds = [
        o for o in broker.orders if o.side is Side.SELL and o.reason == "safe_haven_unwind"
    ]
    assert len(unwinds) >= 1
    assert unwinds[0].symbol == "BRK.B"

    # Et l'ordre d'achat offensif sur NVDA a pu être exécuté grâce au cash libéré
    buys = [o for o in broker.orders if o.side is Side.BUY and o.symbol == "NVDA"]
    assert len(buys) >= 1
