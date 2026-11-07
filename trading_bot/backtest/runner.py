"""Moteur de backtest déterministe, sans lookahead.

Modèle d'exécution :

* les signaux sont calculés de façon **vectorisée** sur les clôtures ;
* un signal issu de la barre ``t`` n'est exécuté qu'à l'ouverture de la barre
  ``t + execution_delay_bars`` (1 par défaut) ;
* chaque exécution paie un slippage et une commission (0,1 % par défaut,
  paramétrables) ;
* les stops sont vérifiés sur le ``low`` de la barre suivante, avec exécution au
  prix du stop (ou à l'ouverture en cas de gap défavorable) ;
* le trailing stop n'est relevé qu'après la clôture, donc jamais appliqué
  rétroactivement à la barre qui l'a déclenché.

Le dimensionnement passe par le même :class:`~trading_bot.risk.engine.RiskEngine`
que le live : un backtest ne peut pas exécuter une taille que le risque
refuserait en production. Aucun aléa n'est utilisé : deux exécutions sur les
mêmes données produisent le même résultat au centime près.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Final
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from trading_bot.config import BacktestConfig, RiskConfig
from trading_bot.data.loader import (
    BarFrame,
    DataError,
    momentum_zscores,
)
from trading_bot.risk.engine import (
    AccountState,
    HaltEvent,
    Position,
    RiskEngine,
)
from trading_bot.strategies.base import SignalType, Strategy, StrategyError

_EPSILON: Final[float] = 1e-9


class BacktestError(RuntimeError):
    """Données ou configuration incompatibles avec le moteur de backtest."""


@dataclass(frozen=True, slots=True)
class Trade:
    """Aller-retour complet d'une position, frais inclus."""

    symbol: str
    qty: float
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    fees: float
    pnl: float
    return_pct: float
    exit_reason: str
    bars_held: int

    @property
    def is_win(self) -> bool:
        """``True`` si l'aller-retour est profitable après frais."""
        return self.pnl > 0.0


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Résultat complet et sérialisable d'un backtest."""

    equity_curve: pd.DataFrame
    trades: tuple[Trade, ...]
    metrics: Mapping[str, float | int | None]
    halts: tuple[HaltEvent, ...]
    initial_equity: float
    symbols: tuple[str, ...]
    strategy_name: str

    def trades_frame(self) -> pd.DataFrame:
        """Retourne les trades sous forme de DataFrame (une ligne par trade)."""
        columns = [
            "symbol",
            "qty",
            "entry_time",
            "entry_price",
            "exit_time",
            "exit_price",
            "fees",
            "pnl",
            "return_pct",
            "exit_reason",
            "bars_held",
        ]
        if not self.trades:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame([self._trade_row(trade) for trade in self.trades])[columns]

    @staticmethod
    def _trade_row(trade: Trade) -> dict[str, object]:
        """Convertit un trade en dictionnaire plat."""
        return {
            "symbol": trade.symbol,
            "qty": trade.qty,
            "entry_time": trade.entry_time,
            "entry_price": trade.entry_price,
            "exit_time": trade.exit_time,
            "exit_price": trade.exit_price,
            "fees": trade.fees,
            "pnl": trade.pnl,
            "return_pct": trade.return_pct,
            "exit_reason": trade.exit_reason,
            "bars_held": trade.bars_held,
        }

    def to_dict(self) -> dict[str, object]:
        """Représentation JSON-compatible (``NaN``/``inf`` deviennent ``None``)."""
        return {
            "strategy": self.strategy_name,
            "symbols": list(self.symbols),
            "initial_equity": self.initial_equity,
            "metrics": {key: _json_number(value) for key, value in self.metrics.items()},
            "trades": [
                {
                    "symbol": trade.symbol,
                    "qty": trade.qty,
                    "entry_time": trade.entry_time.isoformat(),
                    "entry_price": trade.entry_price,
                    "exit_time": trade.exit_time.isoformat(),
                    "exit_price": trade.exit_price,
                    "fees": trade.fees,
                    "pnl": trade.pnl,
                    "return_pct": trade.return_pct,
                    "exit_reason": trade.exit_reason,
                    "bars_held": trade.bars_held,
                }
                for trade in self.trades
            ],
            "halts": [
                {
                    "scope": halt.scope.value,
                    "reason": halt.reason,
                    "session_date": halt.session_date.isoformat(),
                    "equity": halt.equity,
                    "daily_pnl_pct": halt.daily_pnl_pct,
                    "drawdown_pct": halt.drawdown_pct,
                }
                for halt in self.halts
            ],
            "equity_curve": [
                {
                    "timestamp": index.isoformat(),
                    **{
                        column: _json_number(value)
                        for column, value in row.items()
                        if column != "timestamp"
                    },
                }
                for index, row in self.equity_curve.iterrows()
            ],
        }


@dataclass(slots=True)
class _OpenTrade:
    """Métadonnées internes d'une position encore ouverte."""

    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    entry_fee: float
    bars_held: int = 1


@dataclass
class _Simulation:
    """État mutable d'une exécution de backtest (jamais partagé entre runs)."""

    config: BacktestConfig
    risk: RiskEngine
    account: AccountState
    trades: list[Trade] = field(default_factory=list)
    open_trades: dict[str, _OpenTrade] = field(default_factory=dict)
    marks: dict[str, float] = field(default_factory=dict)
    halts: list[HaltEvent] = field(default_factory=list)
    equity_rows: list[dict[str, object]] = field(default_factory=list)
    entry_candidates: int = 0
    entry_rejections: int = 0
    cash_skips: int = 0
    exit_count: int = 0
    is_shield_sword: bool = False
    shield_symbol: str = "SGOV"
    sword_target_weight: float = 0.10
    sword_stop_atr_multiplier: float = 5.0
    sword_max_loss_pct: float = 0.12
    risk_free_rate_annual: float = 0.045

    def process_bar(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        row: pd.Series,
        signal_prev: SignalType,
        atr_prev: float,
    ) -> None:
        """Traite une barre pour un symbole : sorties, entrée, trailing."""
        session_date = _session_date(timestamp, self.config.market_timezone)
        self.risk.start_session(self.account, session_date=session_date)

        position = self.account.positions.get(symbol)
        exited_this_bar = False
        if position is not None:
            if signal_prev is SignalType.SELL:
                self._exit_at(symbol, timestamp, float(row["open"]), "strategy_exit")
                exited_this_bar = True
            else:
                is_shield_asset = self.is_shield_sword and (
                    symbol == self.shield_symbol
                    or symbol in self.risk.config.safe_haven_symbols
                )
                if not is_shield_asset and position.stop_price is not None:
                    stop = position.stop_price
                    trailing = (
                        position.initial_stop_price is not None
                        and stop > position.initial_stop_price
                    )
                    label = "trailing_stop" if trailing else "stop_loss"
                    if float(row["open"]) <= stop:
                        self._exit_at(symbol, timestamp, float(row["open"]), "stop_gap")
                        exited_this_bar = True
                    elif float(row["low"]) <= stop:
                        self._exit_at(symbol, timestamp, stop, label)
                        exited_this_bar = True

        if (
            signal_prev is SignalType.BUY
            and symbol not in self.account.positions
            and not exited_this_bar
            and ((math.isfinite(atr_prev) and atr_prev > 0.0) or self.is_shield_sword)
        ):
            self._enter(symbol, timestamp, row, atr_prev, session_date)

        position = self.account.positions.get(symbol)
        if position is not None:
            meta = self.open_trades.get(symbol)
            if meta is not None and meta.entry_time != timestamp:
                meta.bars_held += 1
            if not self.is_shield_sword:
                trailing_update = self.risk.update_trailing_stop(
                    position,
                    price=float(row["close"]),
                    atr=float(row["atr"]) if math.isfinite(float(row["atr"])) else 0.0,
                )
                if trailing_update is not None:
                    self.risk.apply_trailing_stop(self.account, trailing_update)

        self.marks[symbol] = float(row["close"])
        self.risk.mark_to_market(self.account, self.marks)
        self._recompute_equity()

    def close_bar(self, timestamp: pd.Timestamp) -> None:
        """Contrôle les circuit breakers, liquide si nécessaire, prend un snapshot."""
        session_date = _session_date(timestamp, self.config.market_timezone)
        event = self.risk.update_risk_state(self.account, session_date=session_date)
        if event is not None:
            self.halts.append(event)
            for symbol in sorted(list(self.account.positions)):
                reference = self.marks.get(
                    symbol, self.account.positions[symbol].current_price
                )
                self._exit_at(symbol, timestamp, reference, "circuit_breaker")
            self._recompute_equity()

        drawdown = self.account.drawdown_pct
        if self.is_shield_sword and self.account.cash > 0:
            daily_rate = self.risk_free_rate_annual / 252.0
            daily_interest = self.account.cash * daily_rate
            self.account.cash += daily_interest
            self.account.buying_power = self.account.cash
            self._recompute_equity()
            drawdown = self.account.drawdown_pct

        self.equity_rows.append(
            {
                "timestamp": timestamp,
                "equity": self.account.equity,
                "cash": self.account.cash,
                "gross_exposure": self.account.gross_exposure,
                "positions": len(self.account.positions),
                "drawdown_pct": drawdown,
                "halted": self.account.halted,
            }
        )

    def _enter(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        row: pd.Series,
        atr: float,
        session_date: date,
    ) -> None:
        """Demande l'approbation du risque puis simule le fill à l'ouverture."""
        self.entry_candidates += 1
        open_price = float(row["open"])
        decision_price = open_price * (1.0 + self.config.slippage_pct)
        is_shield_asset = self.is_shield_sword and (
            symbol == self.shield_symbol
            or symbol in self.risk.config.safe_haven_symbols
        )
        if self.is_shield_sword:
            decision = self.risk.evaluate_entry(
                symbol=symbol,
                price=decision_price,
                atr=atr,
                account=self.account,
                now=timestamp.to_pydatetime(),
                session_date=session_date,
                target_weight=self.sword_target_weight if not is_shield_asset else 0.70,
                atr_stop_multiple=self.sword_stop_atr_multiplier if not is_shield_asset else None,
                max_loss_pct=self.sword_max_loss_pct if not is_shield_asset else None,
                is_shield=is_shield_asset,
            )
        else:
            decision = self.risk.evaluate_entry(
                symbol=symbol,
                price=decision_price,
                atr=atr,
                account=self.account,
                now=timestamp.to_pydatetime(),
                session_date=session_date,
            )
        if not decision.approved or decision.intent is None:
            self.entry_rejections += 1
            return

        intent = decision.intent
        qty = intent.qty
        notional = qty * decision_price
        fee = notional * self.config.commission_pct if not is_shield_asset else 0.0
        cost = notional + fee
        if cost > self.account.cash + _EPSILON:
            # Le plafond de liquidités du risque ne tient pas compte des frais.
            self.cash_skips += 1
            return
        stop_price = intent.stop_price
        if stop_price is None and not is_shield_asset:
            raise BacktestError("entrée approuvée sans stop : invariant violé")

        self.account.cash -= cost
        self.account.buying_power = self.account.cash
        self.account.positions[symbol] = Position(
            symbol=symbol,
            qty=qty,
            avg_entry_price=decision_price,
            current_price=decision_price,
            stop_price=stop_price,
            initial_stop_price=stop_price,
            highest_price=decision_price,
            opened_at=timestamp.to_pydatetime(),
        )
        self.open_trades[symbol] = _OpenTrade(
            entry_time=timestamp,
            entry_price=decision_price,
            qty=qty,
            entry_fee=fee,
            bars_held=1,
        )

        # Un stop peut être touché dès la barre d'entrée : on le simule.
        if stop_price is not None and float(row["low"]) <= stop_price:
            self._exit_at(symbol, timestamp, stop_price, "stop_loss")

    def _exit_at(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        reference_price: float,
        reason: str,
    ) -> None:
        """Sortie au marché simulée, via une décision du moteur de risque."""
        position = self.account.positions.get(symbol)
        if position is None or position.qty <= 0:
            return
        if not math.isfinite(reference_price) or reference_price <= 0:
            raise BacktestError(
                f"prix de sortie invalide pour {symbol}: {reference_price!r}"
            )
        decision = self.risk.evaluate_exit(
            position=position,
            price=reference_price,
            now=timestamp.to_pydatetime(),
            reason=reason,
        )
        if not decision.approved or decision.intent is None:
            raise BacktestError(f"sortie refusée par le risque: {decision.reason}")

        fill_price = reference_price * (1.0 - self.config.slippage_pct)
        proceeds = position.qty * fill_price
        exit_fee = proceeds * self.config.commission_pct
        self.account.cash += proceeds - exit_fee
        self.account.buying_power = self.account.cash

        meta = self.open_trades.pop(symbol, None)
        if meta is None:
            raise BacktestError(f"métadonnées de trade absentes pour {symbol}")
        pnl = (fill_price - meta.entry_price) * position.qty - meta.entry_fee - exit_fee
        entry_notional = meta.entry_price * position.qty
        self.trades.append(
            Trade(
                symbol=symbol,
                qty=position.qty,
                entry_time=meta.entry_time,
                entry_price=meta.entry_price,
                exit_time=timestamp,
                exit_price=fill_price,
                fees=meta.entry_fee + exit_fee,
                pnl=pnl,
                return_pct=0.0 if entry_notional <= 0 else pnl / entry_notional,
                exit_reason=reason,
                bars_held=meta.bars_held,
            )
        )
        self.exit_count += 1
        del self.account.positions[symbol]

    def _recompute_equity(self) -> None:
        """Équity = liquidités + valeur de marché des positions."""
        self.account.equity = self.account.cash + sum(
            position.market_value for position in self.account.positions.values()
        )
        self.account.buying_power = self.account.cash


class BacktestRunner:
    """Exécute une stratégie sur un ou plusieurs symboles.

    Args:
        strategy: Stratégie d'analyse (même code qu'en live).
        config: Hypothèses d'exécution et de frais.
        risk_config: Profil de risque (dimensionnement, stops, breakers).
    """

    def __init__(
        self,
        strategy: Strategy,
        *,
        config: BacktestConfig | None = None,
        risk_config: RiskConfig | None = None,
    ) -> None:
        self._strategy = strategy
        self._config = config or BacktestConfig()
        self._risk_config = risk_config or RiskConfig()

    @property
    def config(self) -> BacktestConfig:
        """Hypothèses d'exécution effectives."""
        return self._config

    def run(
        self,
        data: Mapping[str, BarFrame],
        *,
        benchmark: BarFrame | None = None,
        start_date: datetime | date | str | None = None,
    ) -> BacktestResult:
        """Backteste ``data`` (un DataFrame OHLCV normalisé par symbole).

        Args:
            data: Barres par symbole à trader (peut inclure une période de
                warm-up antérieure à ``start_date``).
            benchmark: Barres du benchmark (SPY), jamais tradées : elles
                pilotent le régime de marché (SMA longue) qui desserre les
                plafonds d'exposition. Sans benchmark, le régime est
                systématiquement prudent.
            start_date: Début effectif de la simulation. Si spécifié, les
                barres antérieures servent de warm-up pour stabiliser la
                SMA(200) et les indicateurs sans déclencher d'ordres.

        Raises:
            BacktestError: données vides, index incohérents ou stratégie en
                erreur.
        """
        if not data:
            raise BacktestError("aucune donnée fournie au backtest")
        if hasattr(self._strategy, "set_start_date"):
            self._strategy.set_start_date(start_date)
        if hasattr(self._strategy, "set_benchmark") and benchmark is not None:
            self._strategy.set_benchmark(benchmark)
        if hasattr(self._strategy, "set_universe"):
            self._strategy.set_universe(data)
        symbols = sorted(data)
        prepared: dict[str, BarFrame] = {}
        for symbol in symbols:
            bars = data[symbol]
            if bars.empty:
                raise BacktestError(f"{symbol}: données vides")
            if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
                raise BacktestError(f"{symbol}: index UTC requis")
            try:
                enriched = self._strategy.generate_signals(bars, symbol=symbol)
            except (StrategyError, DataError) as exc:
                raise BacktestError(f"{symbol}: {exc}") from exc
            prepared[symbol] = self._execution_frame(enriched)

        index = pd.DatetimeIndex(
            sorted(set().union(*(frame.index for frame in prepared.values())))
        )
        if len(index) < 2:
            raise BacktestError("au moins deux barres sont nécessaires")

        start_ts: pd.Timestamp | None = None
        if start_date is not None:
            start_ts = pd.Timestamp(start_date)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            else:
                start_ts = start_ts.tz_convert("UTC")

        sim_index = (
            pd.DatetimeIndex([ts for ts in index if ts >= start_ts])
            if start_ts is not None
            else index
        )
        if len(sim_index) < 2:
            sim_index = index

        risk = RiskEngine(self._risk_config)
        is_shield_sword = getattr(self._strategy, "name", "") == "shield_sword"
        simulation = _Simulation(
            config=self._config,
            risk=risk,
            account=AccountState(
                equity=self._config.initial_equity,
                cash=self._config.initial_equity,
                buying_power=self._config.initial_equity,
            ),
            is_shield_sword=is_shield_sword,
        )

        # Régime de marché par séance : benchmark au-dessus de sa SMA longue.
        # La valeur utilisée à la barre t est celle de la barre de signal
        # (t - execution_delay_bars), comme pour les signaux eux-mêmes.
        regime_by_date: dict[pd.Timestamp, tuple[bool, bool]] = {}
        if benchmark is not None and not benchmark.empty and "close" in benchmark.columns:
            close = benchmark["close"].astype(float)
            sma_period = self._risk_config.regime_sma_period
            if len(close) >= sma_period:
                sma = close.rolling(sma_period).mean()
                bull = (close > sma).where(sma.notna(), False).astype(bool).shift(self._config.execution_delay_bars)
                bear = (close <= sma).where(sma.notna(), False).astype(bool).shift(self._config.execution_delay_bars)
                for ts in benchmark.index:
                    regime_by_date[ts] = (bool(bull.get(ts, False)), bool(bear.get(ts, False)))

        # Score de classement par symbole (z-score de momentum, même mesure que
        # le live) : à chaque barre d'exécution, le budget va d'abord au plus
        # vigoureux.
        scores_by_symbol: dict[str, pd.Series] = {}
        for symbol in symbols:
            z_fast, z_slow = momentum_zscores(data[symbol]["close"])
            scores_by_symbol[symbol] = (z_fast + z_slow).shift(
                self._config.execution_delay_bars
            )

        for timestamp in sim_index:
            bull_market, bear_market = regime_by_date.get(timestamp, (False, False))
            risk.set_market_regime(bull_market=bull_market, bear_market=bear_market)
            scores = {
                symbol: float(scores_by_symbol[symbol].get(timestamp, float("nan")))
                for symbol in symbols
                if timestamp in prepared[symbol].index
            }
            for symbol in risk.rank_candidates(scores):
                frame = prepared[symbol]
                if timestamp not in frame.index:
                    continue
                row = frame.loc[timestamp]
                signal_prev = row["signal_prev"]
                if not isinstance(signal_prev, SignalType):
                    signal_prev = SignalType.HOLD
                atr_prev = float(row["atr_prev"])
                simulation.process_bar(
                    symbol,
                    timestamp,
                    row,
                    signal_prev,
                    atr_prev,
                )
            simulation.close_bar(timestamp)

        curve = pd.DataFrame(simulation.equity_rows).set_index("timestamp")
        curve.index = pd.DatetimeIndex(curve.index)
        curve = curve.astype(
            {
                "equity": float,
                "cash": float,
                "gross_exposure": float,
                "positions": int,
                "drawdown_pct": float,
                "halted": bool,
            }
        )

        metrics = self._metrics(
            curve=curve,
            trades=simulation.trades,
            prepared=prepared,
            simulation=simulation,
            benchmark=benchmark,
        )
        return BacktestResult(
            equity_curve=curve,
            trades=tuple(sorted(simulation.trades, key=lambda t: (t.exit_time, t.symbol))),
            metrics=metrics,
            halts=tuple(simulation.halts),
            initial_equity=self._config.initial_equity,
            symbols=tuple(symbols),
            strategy_name=self._strategy.name,
        )

    def _execution_frame(self, enriched: BarFrame) -> BarFrame:
        """Ajoute les colonnes exécutables décalées (signal et ATR de t-1)."""
        frame = enriched.copy()
        delay = self._config.execution_delay_bars
        shifted_signal = frame["signal"].shift(delay)
        frame["signal_prev"] = shifted_signal.where(shifted_signal.notna(), SignalType.HOLD)
        frame["atr_prev"] = frame["atr"].shift(delay)
        return frame

    def _metrics(
        self,
        *,
        curve: pd.DataFrame,
        trades: list[Trade],
        prepared: Mapping[str, BarFrame],
        simulation: _Simulation,
        benchmark: BarFrame | None = None,
    ) -> dict[str, float | int | None]:
        """Calcule les métriques de performance et de risque.

        Args:
            curve: Courbe d'équity du backtest.
            trades: Allers-retours clôturés.
            prepared: Barres enrichies par symbole tradé.
            simulation: État interne de la simulation.
            benchmark: Barres du benchmark (SPY), pour la référence de marché.
        """
        equity = curve["equity"].astype(float)
        initial = self._config.initial_equity
        final = float(equity.iloc[-1])
        total_return = final / initial - 1.0 if initial > 0 else None

        timestamps = curve.index
        days = max((timestamps[-1] - timestamps[0]).total_seconds() / 86_400.0, 0.0)
        years = days / 365.25
        if years > 0 and final > 0 and initial > 0:
            cagr = (final / initial) ** (1.0 / years) - 1.0
        else:
            cagr = None

        returns = equity.pct_change().dropna()
        periods = self._config.annualization_factor
        risk_free_per_period = self._config.risk_free_rate / periods
        sharpe: float | None = None
        sortino: float | None = None
        if len(returns) >= 2:
            excess = returns - risk_free_per_period
            std = float(excess.std(ddof=1))
            if std > 0 and math.isfinite(std):
                sharpe = float(excess.mean() / std * math.sqrt(periods))
            downside = excess[excess < 0.0]
            if len(downside) >= 2:
                downside_std = float(downside.std(ddof=1))
                if downside_std > 0 and math.isfinite(downside_std):
                    sortino = float(excess.mean() / downside_std * math.sqrt(periods))

        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
        calmar: float | None = None
        if cagr is not None and max_drawdown < 0:
            calmar = cagr / abs(max_drawdown)

        wins = [trade for trade in trades if trade.pnl > 0]
        losses = [trade for trade in trades if trade.pnl < 0]
        gross_profit = sum(trade.pnl for trade in wins)
        gross_loss = -sum(trade.pnl for trade in losses)
        profit_factor: float | None = None
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = None  # aucun perdant : ratio non défini, pas infini

        total_fees = sum(trade.fees for trade in trades)
        expectancy = sum(trade.pnl for trade in trades) / len(trades) if trades else None
        avg_bars = (
            sum(trade.bars_held for trade in trades) / len(trades) if trades else None
        )

        # Panier équipondéré : ce qu'aurait rapporté « tout acheter et ne rien
        # toucher » sur les actifs tradés, frais non déduits. Ce n'est PAS la
        # référence de marché : c'est la performance du panier lui-même.
        basket_returns = [
            float(frame["close"].iloc[-1] / frame["close"].iloc[0] - 1.0)
            for frame in prepared.values()
            if len(frame) > 1
        ]
        basket_buy_hold = (
            sum(basket_returns) / len(basket_returns) if basket_returns else None
        )
        # Référence : performance de l'indice (SPY) sur la même fenêtre. C'est
        # la seule comparaison qui dise si la stratégie a battu le marché.
        benchmark_buy_hold = _buy_and_hold(benchmark)

        return {
            "initial_equity": initial,
            "final_equity": final,
            "total_return": total_return,
            "cagr": cagr,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_drawdown,
            "calmar": calmar,
            "trades": len(trades),
            "win_rate": (len(wins) / len(trades)) if trades else None,
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "expectancy": expectancy,
            "avg_bars_held": avg_bars,
            "total_fees": total_fees,
            "exposure_pct": float((curve["gross_exposure"] > 0).mean()),
            "halt_events": len(simulation.halts),
            "entry_candidates": simulation.entry_candidates,
            "entry_rejections": simulation.entry_rejections,
            "cash_skips": simulation.cash_skips,
            "open_positions_at_end": len(simulation.account.positions),
            "benchmark_buy_hold": benchmark_buy_hold,
            "basket_buy_hold": basket_buy_hold,
        }


def _buy_and_hold(frame: BarFrame | None) -> float | None:
    """Performance « acheter puis ne plus toucher » sur une fenêtre de barres.

    Args:
        frame: Barres OHLCV du benchmark, ou ``None``.

    Returns:
        Le rendement total sur la fenêtre couverte par les barres, ou ``None``
        si la fenêtre est trop courte pour être exploitable.
    """
    if frame is None or len(frame) < 2:
        return None
    first = float(frame["close"].iloc[0])
    if not math.isfinite(first) or first <= 0:
        return None
    return float(frame["close"].iloc[-1] / first - 1.0)


def _session_date(timestamp: pd.Timestamp, market_timezone: str) -> date:
    """Date de séance d'une barre (gère les labels journaliers à minuit UTC)."""
    if timestamp.hour == 0 and timestamp.minute == 0 and timestamp.second == 0:
        return date(timestamp.year, timestamp.month, timestamp.day)
    local = timestamp.tz_convert(ZoneInfo(market_timezone))
    return date(local.year, local.month, local.day)


def _json_number(value: object) -> float | int | bool | None:
    """Convertit un scalaire numpy/pandas en type JSON, ``NaN``/``inf`` -> None."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


__all__ = ["BacktestError", "BacktestResult", "BacktestRunner", "Trade"]
