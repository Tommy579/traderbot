"""Tests de la passerelle Alpaca Paper (clients SDK simulés, aucun réseau)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
import pytest
from alpaca.data.enums import DataFeed
from alpaca.trading.enums import AssetClass, OrderSide, OrderStatus

from trading_bot.config import Settings
from trading_bot.execution.alpaca_client import (
    AlpacaPaperClient,
    BrokerError,
    MarketClosure,
    MarketStatus,
    OrderRejectedError,
    OrderTimeoutError,
    PaperTradingViolation,
)
from trading_bot.risk.engine import AccountState, Position, RiskEngine, Side

NOW = datetime(2024, 1, 10, 15, 0, tzinfo=UTC)


class FakeTradingClient:
    """Doublure minimale de ``TradingClient``."""

    def __init__(self, *, status: OrderStatus = OrderStatus.FILLED) -> None:
        self.orders: list[object] = []
        self.status = status
        self.cancelled = 0
        self.closed = 0
        self._by_id: dict[str, object] = {}
        self.clock: object = SimpleNamespace(is_open=True)
        self.calendar: list[object] = []
        self.calendar_error: Exception | None = None

    def get_account(self) -> object:
        return SimpleNamespace(
            equity="100000.50", cash="50000.25", buying_power="150000.00"
        )

    def get_all_positions(self) -> list[object]:
        return [
            SimpleNamespace(
                symbol="AAPL",
                qty="10",
                avg_entry_price="95.00",
                current_price="101.00",
            )
        ]

    def get_clock(self) -> object:
        return self.clock

    def get_calendar(self, filters: object = None) -> list[object]:
        if self.calendar_error is not None:
            raise self.calendar_error
        return self.calendar

    def submit_order(self, order_data: object) -> object:
        self.orders.append(order_data)
        order = SimpleNamespace(
            id=str(uuid4()),
            client_order_id=order_data.client_order_id,
            symbol=order_data.symbol,
            qty=str(order_data.qty),
            side=order_data.side,
            status=self.status,
            filled_qty=str(order_data.qty) if self.status == OrderStatus.FILLED else "0",
            filled_avg_price="100.00" if self.status == OrderStatus.FILLED else None,
            submitted_at=NOW,
        )
        self._by_id[str(order.id)] = order
        return order

    def get_order_by_id(self, order_id: str) -> object:
        return self._by_id[str(order_id)]

    def cancel_orders(self) -> None:
        self.cancelled += 1

    def close_all_positions(self, cancel_orders: bool = True) -> None:
        self.closed += 1


class FakeDataClient:
    """Doublure minimale de ``StockHistoricalDataClient``."""

    def __init__(self) -> None:
        self.last_request: object | None = None

    def get_stock_latest_trade(self, request: object) -> dict[str, object]:
        return {
            "AAPL": SimpleNamespace(price="101.50"),
            "MSFT": SimpleNamespace(price="420.00"),
        }

    def get_stock_bars(self, request: object) -> object:
        self.last_request = request
        index = pd.MultiIndex.from_product(
            [["AAPL"], pd.date_range("2024-01-02", periods=10, freq="D", tz="UTC")],
            names=["symbol", "timestamp"],
        )
        values = pd.DataFrame(
            {
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 1000.0,
                "trade_count": 10.0,
                "vwap": 100.5,
            },
            index=index,
        )
        return SimpleNamespace(df=values)


@pytest.fixture
def fake_trading() -> FakeTradingClient:
    return FakeTradingClient()


@pytest.fixture
def client(settings: Settings, fake_trading: FakeTradingClient) -> AlpacaPaperClient:
    return AlpacaPaperClient(
        settings,
        trading_client=fake_trading,
        data_client=FakeDataClient(),
    )


def make_intent(*, price: float = 100.0, atr: float = 1.0) -> object:
    """Construit un ordre réellement approuvé par le RiskEngine."""
    decision = RiskEngine().evaluate_entry(
        symbol="AAPL",
        price=price,
        atr=atr,
        account=AccountState(equity=100_000.0, cash=100_000.0, buying_power=100_000.0),
        now=NOW,
        session_date=date(2024, 1, 10),
    )
    assert decision.intent is not None
    return decision.intent


def test_constructor_rejects_live_endpoint(settings: Settings) -> None:
    object.__setattr__(settings, "alpaca_trading_url", "https://api.alpaca.markets")
    with pytest.raises(PaperTradingViolation):
        AlpacaPaperClient(
            settings,
            trading_client=FakeTradingClient(),
            data_client=FakeDataClient(),
        )


def test_submit_builds_market_order(client: AlpacaPaperClient, fake_trading) -> None:
    intent = make_intent()
    receipt = client.submit(intent, now=NOW)
    assert len(fake_trading.orders) == 1
    request = fake_trading.orders[0]
    assert request.symbol == "AAPL"
    assert request.side is OrderSide.BUY
    assert float(request.qty) == pytest.approx(intent.qty)
    assert request.client_order_id == intent.client_order_id
    assert receipt.is_filled
    assert receipt.status == "filled"
    assert receipt.side is Side.BUY


def test_submit_rejects_replay(client: AlpacaPaperClient, fake_trading) -> None:
    intent = make_intent()
    client.submit(intent, now=NOW)
    with pytest.raises(BrokerError, match="idempotence"):
        client.submit(intent, now=NOW)
    assert len(fake_trading.orders) == 1


def test_submit_rejects_expired_token(client: AlpacaPaperClient) -> None:
    intent = make_intent()
    with pytest.raises(BrokerError, match="expiré"):
        client.submit(intent, now=NOW + timedelta(minutes=5))


def test_account_and_positions_normalized(client: AlpacaPaperClient) -> None:
    account = client.get_account()
    assert account.equity == pytest.approx(100_000.50)
    assert account.cash == pytest.approx(50_000.25)
    positions = client.get_positions()
    assert positions["AAPL"].qty == pytest.approx(10.0)
    assert positions["AAPL"].avg_entry_price == pytest.approx(95.0)
    assert positions["AAPL"].current_price == pytest.approx(101.0)


def test_latest_prices(client: AlpacaPaperClient) -> None:
    prices = client.get_latest_prices(["AAPL", "MSFT", "UNKNOWN"])
    assert prices == {"AAPL": 101.5, "MSFT": 420.0}


def test_fetch_bars_normalizes_multiindex(client: AlpacaPaperClient) -> None:
    bars = client.fetch_bars("AAPL", timeframe="1Day", lookback_days=30, now=NOW)
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert str(bars.index.tz) == "UTC"
    assert len(bars) == 10


def test_wait_for_fill_returns_filled(client: AlpacaPaperClient) -> None:
    receipt = client.submit(make_intent(), now=NOW)
    filled = client.wait_for_fill(receipt.order_id, timeout_seconds=1.0)
    assert filled.is_filled
    assert filled.avg_fill_price == pytest.approx(100.0)


def test_wait_for_fill_rejects_terminal_failure(
    settings: Settings, fake_trading: FakeTradingClient
) -> None:
    fake_trading.status = OrderStatus.REJECTED
    client = AlpacaPaperClient(
        settings, trading_client=fake_trading, data_client=FakeDataClient()
    )
    receipt = client.submit(make_intent(), now=NOW)
    with pytest.raises(OrderRejectedError):
        client.wait_for_fill(receipt.order_id, timeout_seconds=1.0)


def test_wait_for_fill_times_out(
    settings: Settings, fake_trading: FakeTradingClient
) -> None:
    fake_trading.status = OrderStatus.NEW
    client = AlpacaPaperClient(
        settings, trading_client=fake_trading, data_client=FakeDataClient()
    )
    receipt = client.submit(make_intent(), now=NOW)
    with pytest.raises(OrderTimeoutError):
        client.wait_for_fill(receipt.order_id, timeout_seconds=0.05, poll_seconds=0.01)


def test_market_clock_and_emergency_methods(
    client: AlpacaPaperClient, fake_trading: FakeTradingClient
) -> None:
    assert client.is_market_open() is True
    client.cancel_all_orders()
    client.close_all_positions()
    assert fake_trading.cancelled == 1
    assert fake_trading.closed == 1


def test_market_status_open(client: AlpacaPaperClient) -> None:
    """Une horloge ouverte : marché ouvert, aucun motif de fermeture."""
    assert client.market_status() == MarketStatus(is_open=True)


def test_market_status_weekend(
    client: AlpacaPaperClient, fake_trading: FakeTradingClient
) -> None:
    """Dimanche 1er nov. 2026 : hors horaires ET hors semaine → week-end.

    C'est le cas qui trompait l'utilisateur : 17 h à Paris = 11 h à New York,
    dans les heures de séance, mais l'horloge Alpaca est fermée.
    """
    next_open = datetime(2026, 11, 2, 14, 30, tzinfo=UTC)
    fake_trading.clock = SimpleNamespace(
        is_open=False,
        timestamp=datetime(2026, 11, 1, 16, 0, tzinfo=UTC),
        next_open=next_open,
    )
    fake_trading.calendar = []
    status = client.market_status()
    assert status.is_open is False
    assert status.closure is MarketClosure.WEEKEND
    assert status.next_open == next_open


def test_market_status_holiday(
    client: AlpacaPaperClient, fake_trading: FakeTradingClient
) -> None:
    """Un jour de semaine sans séance au calendrier : jour férié."""
    fake_trading.clock = SimpleNamespace(
        is_open=False, timestamp=datetime(2026, 12, 25, 15, 0, tzinfo=UTC)
    )
    fake_trading.calendar = []
    assert client.market_status().closure is MarketClosure.HOLIDAY


def test_market_status_outside_hours(
    client: AlpacaPaperClient, fake_trading: FakeTradingClient
) -> None:
    """Une séance existe ce jour-là mais l'horloge est fermée : hors séance."""
    fake_trading.clock = SimpleNamespace(
        is_open=False, timestamp=datetime(2024, 1, 10, 22, 0, tzinfo=UTC)
    )
    fake_trading.calendar = [SimpleNamespace(date=date(2024, 1, 10))]
    assert client.market_status().closure is MarketClosure.OUTSIDE_HOURS


def test_market_status_calendar_failure_degrades(
    client: AlpacaPaperClient, fake_trading: FakeTradingClient
) -> None:
    """Calendrier illisible : on retombe sur la seule information locale."""
    fake_trading.clock = SimpleNamespace(
        is_open=False, timestamp=datetime(2026, 11, 1, 16, 0, tzinfo=UTC)
    )
    fake_trading.calendar_error = RuntimeError("calendrier indisponible")
    assert client.market_status().closure is MarketClosure.WEEKEND


def test_market_status_clock_failure(
    client: AlpacaPaperClient, fake_trading: FakeTradingClient
) -> None:
    """Horloge illisible : erreur de courtier explicite, jamais silencieuse."""

    def boom() -> object:
        raise RuntimeError("horloge indisponible")

    fake_trading.get_clock = boom  # type: ignore[method-assign]
    with pytest.raises(BrokerError):
        client.market_status()


def test_data_feed_uses_settings(settings: Settings) -> None:
    settings = settings.model_copy(update={"data_feed": "sip"})
    client = AlpacaPaperClient(
        settings,
        trading_client=FakeTradingClient(),
        data_client=FakeDataClient(),
    )
    assert client._feed is DataFeed.SIP


def test_repr_does_not_leak_secrets(client: AlpacaPaperClient) -> None:
    text = repr(client)
    assert "SECRET_TEST" not in text
    assert "PK_TEST" not in text
    assert "paper-api.alpaca.markets" in text


def test_submit_stop_intent_builds_stop_order(
    settings: Settings, fake_trading: FakeTradingClient
) -> None:
    client = AlpacaPaperClient(
        settings, trading_client=fake_trading, data_client=FakeDataClient()
    )
    position = Position(
        symbol="AAPL",
        qty=10,
        avg_entry_price=100.0,
        current_price=110.0,
        stop_price=104.5,
        initial_stop_price=96.0,
    )
    decision = RiskEngine().evaluate_stop_replace(position, price=110.0, now=NOW)
    assert decision is not None and decision.intent is not None
    receipt = client.submit(decision.intent, now=NOW)
    assert len(fake_trading.orders) == 1
    request = fake_trading.orders[0]
    assert request.side is OrderSide.SELL
    assert float(request.stop_price) == pytest.approx(104.5)
    assert request.client_order_id == decision.intent.client_order_id
    assert receipt.is_filled


def test_submit_stop_rounds_fractional_qty_down_to_whole_shares(
    settings: Settings, fake_trading: FakeTradingClient
) -> None:
    """Alpaca refuse un stop fractionnaire en GTC : on protège les actions entières."""
    client = AlpacaPaperClient(
        settings, trading_client=fake_trading, data_client=FakeDataClient()
    )
    position = Position(
        symbol="KO",
        qty=57.268,
        avg_entry_price=94.69,
        current_price=94.71,
        stop_price=91.47,
        initial_stop_price=91.47,
    )
    decision = RiskEngine().evaluate_stop_replace(position, price=94.71, now=NOW)
    assert decision is not None and decision.intent is not None
    client.submit(decision.intent, now=NOW)

    request = fake_trading.orders[0]
    assert float(request.qty) == pytest.approx(57.0)
    assert float(request.stop_price) == pytest.approx(91.47)


def test_submit_stop_keeps_whole_qty_untouched(
    settings: Settings, fake_trading: FakeTradingClient
) -> None:
    """Une position déjà entière n'est pas rognée."""
    client = AlpacaPaperClient(
        settings, trading_client=fake_trading, data_client=FakeDataClient()
    )
    position = Position(
        symbol="XOM",
        qty=128.0,
        avg_entry_price=182.45,
        current_price=180.29,
        stop_price=174.6,
        initial_stop_price=174.6,
    )
    decision = RiskEngine().evaluate_stop_replace(position, price=180.29, now=NOW)
    assert decision is not None and decision.intent is not None
    client.submit(decision.intent, now=NOW)

    assert float(fake_trading.orders[0].qty) == pytest.approx(128.0)


def test_submit_stop_below_one_share_is_refused_with_a_clear_message(
    settings: Settings, fake_trading: FakeTradingClient
) -> None:
    """Moins d'une action entière : aucun stop possible, le refus doit le dire."""
    client = AlpacaPaperClient(
        settings, trading_client=fake_trading, data_client=FakeDataClient()
    )
    position = Position(
        symbol="PLTR",
        qty=0.4,
        avg_entry_price=184.33,
        current_price=178.88,
        stop_price=168.8,
        initial_stop_price=168.8,
    )
    decision = RiskEngine().evaluate_stop_replace(position, price=178.88, now=NOW)
    assert decision is not None and decision.intent is not None

    with pytest.raises(BrokerError, match="au moins une action entière"):
        client.submit(decision.intent, now=NOW)
    assert fake_trading.orders == []


def test_broker_rejection_message_carries_the_cause(
    settings: Settings, fake_trading: FakeTradingClient
) -> None:
    """« rejet de l'ordre X » seul ne dit jamais pourquoi : le motif doit suivre."""
    client = AlpacaPaperClient(
        settings, trading_client=fake_trading, data_client=FakeDataClient()
    )

    def boom(order_data: object) -> None:
        raise RuntimeError("insufficient qty available for order")

    fake_trading.submit_order = boom  # type: ignore[method-assign]
    position = Position(
        symbol="AAPL",
        qty=10,
        avg_entry_price=100.0,
        current_price=110.0,
        stop_price=104.5,
        initial_stop_price=96.0,
    )
    decision = RiskEngine().evaluate_stop_replace(position, price=110.0, now=NOW)
    assert decision is not None and decision.intent is not None

    with pytest.raises(BrokerError, match="insufficient qty available"):
        client.submit(decision.intent, now=NOW)


def test_cancel_open_orders_cancels_and_verifies_each_open_order(
    settings: Settings, fake_trading: FakeTradingClient
) -> None:
    open_orders = [SimpleNamespace(id="order-1"), SimpleNamespace(id="order-2")]
    reads = 0

    def get_orders(filter: object) -> list[object]:
        nonlocal reads
        reads += 1
        return open_orders if reads == 1 else []

    fake_trading.get_orders = get_orders  # type: ignore[attr-defined]
    fake_trading.cancelled_ids: list[str] = []
    fake_trading.cancel_order_by_id = lambda order_id: fake_trading.cancelled_ids.append(
        order_id
    )
    client = AlpacaPaperClient(
        settings, trading_client=fake_trading, data_client=FakeDataClient()
    )

    assert client.cancel_open_orders("AAPL") == 2
    assert fake_trading.cancelled_ids == ["order-1", "order-2"]
    assert reads == 2


def test_cancel_open_orders_fails_when_order_remains_open(
    settings: Settings, fake_trading: FakeTradingClient
) -> None:
    order = SimpleNamespace(id="protective-stop")
    reads = 0

    def get_orders(filter: object) -> list[object]:
        nonlocal reads
        reads += 1
        return [order]

    fake_trading.get_orders = get_orders  # type: ignore[attr-defined]
    fake_trading.cancel_order_by_id = lambda order_id: None
    client = AlpacaPaperClient(
        settings, trading_client=fake_trading, data_client=FakeDataClient()
    )

    with pytest.raises(BrokerError, match="encore ouverts"):
        client.cancel_open_orders("AAPL")
    assert reads == 2


class _AssetNotFound(Exception):
    """Erreur Alpaca « asset not found » (code 40410000)."""

    code = 40410000


def _client_with_asset(
    settings: Settings, fake_trading: FakeTradingClient, responder: object
) -> AlpacaPaperClient:
    fake_trading.get_asset = responder  # type: ignore[attr-defined]
    return AlpacaPaperClient(
        settings, trading_client=fake_trading, data_client=FakeDataClient()
    )


def test_describe_asset_reports_the_real_instrument(
    settings: Settings, fake_trading: FakeTradingClient
) -> None:
    """« BTC » chez Alpaca est un ETF Grayscale, pas le Bitcoin."""
    responder = lambda symbol: SimpleNamespace(  # noqa: E731
        symbol="BTC",
        name="Grayscale Bitcoin Mini Trust ETF",
        asset_class=AssetClass.US_EQUITY,
        tradable=True,
    )
    client = _client_with_asset(settings, fake_trading, responder)

    info = client.describe_asset("BTC")
    assert info is not None
    assert info.name == "Grayscale Bitcoin Mini Trust ETF"
    assert info.asset_class == "us_equity"
    assert info.is_crypto is False


def test_describe_asset_flags_a_crypto_pair(
    settings: Settings, fake_trading: FakeTradingClient
) -> None:
    responder = lambda symbol: SimpleNamespace(  # noqa: E731
        symbol="BTC/USD",
        name="Bitcoin  / US Dollar",
        asset_class=AssetClass.CRYPTO,
        tradable=True,
    )
    client = _client_with_asset(settings, fake_trading, responder)

    info = client.describe_asset("BTC/USD")
    assert info is not None
    assert info.is_crypto is True


def test_describe_asset_returns_none_when_alpaca_does_not_know_it(
    settings: Settings, fake_trading: FakeTradingClient
) -> None:
    def responder(symbol: str) -> object:
        raise _AssetNotFound(f"asset not found for {symbol}")

    client = _client_with_asset(settings, fake_trading, responder)
    assert client.describe_asset("ZZZZZZZZ") is None


def test_describe_asset_raises_on_transport_failure(
    settings: Settings, fake_trading: FakeTradingClient
) -> None:
    """Une panne réseau ne doit pas passer pour un symbole inconnu."""

    def responder(symbol: str) -> object:
        raise RuntimeError("connexion perdue")

    client = _client_with_asset(settings, fake_trading, responder)
    with pytest.raises(BrokerError):
        client.describe_asset("AAPL")
