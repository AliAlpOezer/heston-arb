"""
Broker abstraction layer.

Defines the Broker protocol that any execution backend must implement.
Concrete implementations:
  PaperBroker  — simulates fills at mid-price; no real orders sent (default)
  IBKRBroker   — Interactive Brokers via ib_insync (stub, wires up the interface)

Order model:
  - Options are entered as limit orders at mid-price.
  - Delta hedge is a market order on the underlying (shares or SPX futures).
  - All quantities are signed: positive = long, negative = short.

Fill assumptions in PaperBroker:
  - Options: filled at mid-price instantly (optimistic — add slippage for realism).
  - Underlying: filled at last_price (no slippage model).
  - No partial fills, no margin checks, no position limits beyond the strategy layer.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Quote:
    """Current market quote for a security."""
    symbol: str
    bid: float
    ask: float
    last: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0 if self.bid and self.ask else self.last


@dataclass
class OptionContract:
    """Identifies a specific option contract."""
    underlying: str       # e.g. "SPY"
    expiry: str           # "YYYY-MM-DD"
    strike: float
    right: str            # "C" or "P"
    exchange: str = "SMART"
    currency: str = "USD"

    @property
    def symbol(self) -> str:
        return f"{self.underlying}_{self.expiry}_{self.strike:.0f}{self.right}"


@dataclass
class Order:
    """A submitted order."""
    order_id: str
    contract: OptionContract | str     # OptionContract or underlying ticker
    action: str                        # "BUY" or "SELL"
    qty: int                           # always positive; action determines direction
    order_type: str                    # "MKT" or "LMT"
    limit_price: Optional[float]
    submitted_at: str                  # ISO timestamp
    filled_at: Optional[str] = None
    fill_price: Optional[float] = None
    status: str = "pending"            # "pending" | "filled" | "cancelled" | "rejected"


@dataclass
class AccountSnapshot:
    """Account state at a point in time."""
    net_liquidation: float
    cash: float
    margin_used: float
    unrealized_pnl: float
    timestamp: str


# ── Broker protocol ───────────────────────────────────────────────────────────

@runtime_checkable
class Broker(Protocol):
    """Minimal interface any broker backend must implement."""

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Current bid/ask/last for the underlying or an option symbol."""
        ...

    def get_account(self) -> AccountSnapshot:
        """Account balance, margin, and unrealized P&L."""
        ...

    def submit_option_order(
        self,
        contract: OptionContract,
        action: str,
        qty: int,
        order_type: str = "LMT",
        limit_price: Optional[float] = None,
    ) -> Order:
        """Submit an option buy or sell. Returns Order with assigned order_id."""
        ...

    def submit_hedge_order(
        self,
        underlying: str,
        qty: int,
        order_type: str = "MKT",
        limit_price: Optional[float] = None,
    ) -> Order:
        """Submit a hedge trade on the underlying (shares). qty is signed."""
        ...

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if cancelled successfully."""
        ...

    def get_open_orders(self) -> list[Order]:
        """Return all pending (unfilled) orders."""
        ...


# ── Paper broker ──────────────────────────────────────────────────────────────

class PaperBroker:
    """Paper trading broker — simulates fills at mid-price.

    Writes an execution log to paper_log.json so you can review every order.
    Useful for strategy validation before connecting a real broker.

    Slippage model: none by default. Set slippage_bps to add a cost:
        fill_price = mid * (1 + slippage_bps/10000 * sign(action))
    """

    def __init__(
        self,
        log_path: Optional[Path] = None,
        slippage_bps: float = 0.0,
    ):
        self._log_path = log_path or Path("data/paper_log.json")
        self._slippage_bps = slippage_bps
        self._orders: dict[str, Order] = {}
        self._order_counter = 0
        self._hedge_qty: int = 0           # net shares of underlying held
        self._option_positions: dict[str, int] = {}  # symbol -> signed qty
        self._load_log()

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Returns None — paper broker has no real-time feed.

        The live loop must supply prices from Polygon or another source.
        """
        return None

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(
            net_liquidation=1_000_000.0,
            cash=1_000_000.0,
            margin_used=0.0,
            unrealized_pnl=0.0,
            timestamp=datetime.utcnow().isoformat(),
        )

    def is_market_open(self) -> bool:
        """Paper broker has no market calendar — always tradeable."""
        return True

    def submit_option_order(
        self,
        contract: OptionContract,
        action: str,
        qty: int,
        order_type: str = "LMT",
        limit_price: Optional[float] = None,
    ) -> Order:
        self._order_counter += 1
        oid = f"PAPER-{self._order_counter:06d}"
        now = datetime.utcnow().isoformat()

        fill = limit_price or 0.0
        slippage = fill * self._slippage_bps / 10_000 * (1 if action == "BUY" else -1)
        fill_price = fill + slippage

        signed_qty = qty if action == "BUY" else -qty
        self._option_positions[contract.symbol] = (
            self._option_positions.get(contract.symbol, 0) + signed_qty
        )

        order = Order(
            order_id=oid,
            contract=contract,
            action=action,
            qty=qty,
            order_type=order_type,
            limit_price=limit_price,
            submitted_at=now,
            filled_at=now,
            fill_price=fill_price,
            status="filled",
        )
        self._orders[oid] = order
        self._append_log(order)
        return order

    def submit_hedge_order(
        self,
        underlying: str,
        qty: int,
        order_type: str = "MKT",
        limit_price: Optional[float] = None,
    ) -> Order:
        self._order_counter += 1
        oid = f"PAPER-{self._order_counter:06d}"
        now = datetime.utcnow().isoformat()
        action = "BUY" if qty > 0 else "SELL"
        fill_price = limit_price or 0.0

        self._hedge_qty += qty

        order = Order(
            order_id=oid,
            contract=underlying,
            action=action,
            qty=abs(qty),
            order_type=order_type,
            limit_price=limit_price,
            submitted_at=now,
            filled_at=now,
            fill_price=fill_price,
            status="filled",
        )
        self._orders[oid] = order
        self._append_log(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders and self._orders[order_id].status == "pending":
            self._orders[order_id].status = "cancelled"
            return True
        return False

    def get_open_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if o.status == "pending"]

    @property
    def hedge_qty(self) -> int:
        """Net shares of underlying currently held as hedge."""
        return self._hedge_qty

    # ── Log I/O ───────────────────────────────────────────────────────────

    def _load_log(self) -> None:
        if self._log_path.exists():
            try:
                with self._log_path.open() as f:
                    raw = json.load(f)
                self._order_counter = len(raw)
            except Exception:
                pass

    def _append_log(self, order: Order) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = []
            if self._log_path.exists():
                with self._log_path.open() as f:
                    existing = json.load(f)
            record = {k: v for k, v in asdict(order).items()}
            if isinstance(order.contract, OptionContract):
                record["contract"] = asdict(order.contract)
            existing.append(record)
            with self._log_path.open("w") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            warnings.warn(f"[paper] Failed to write order log: {e}")


# ── Alpaca broker ─────────────────────────────────────────────────────────────

def _occ_symbol(contract: OptionContract) -> str:
    """Convert OptionContract to OCC option symbol used by Alpaca.

    Format: {underlying}{YYMMDD}{C|P}{strike*1000 zero-padded to 8 digits}
    Example: SPY 2024-03-15 Call $415.00 -> SPY240315C00415000
    """
    exp = contract.expiry.replace("-", "")   # YYYYMMDD
    exp_short = exp[2:]                       # YYMMDD
    opt_type = contract.right.upper()         # C or P
    strike_int = round(contract.strike * 1000)
    return f"{contract.underlying}{exp_short}{opt_type}{strike_int:08d}"


class AlpacaBroker:
    """Alpaca Markets broker via alpaca-py.

    Supports both paper and live trading.
    Requires:
        pip install alpaca-py

    API keys are read from environment variables:
        ALPACA_API_KEY    (or pass api_key=)
        ALPACA_SECRET_KEY (or pass secret_key=)

    Paper trading endpoint is used when paper=True (default).
    Switch to paper=False only for live capital.

    Example:
        broker = AlpacaBroker()            # paper, keys from env
        broker = AlpacaBroker(paper=False) # live
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
    ):
        import os
        try:
            from alpaca.trading.client import TradingClient
        except ImportError:
            raise ImportError(
                "alpaca-py not installed. Run: pip install alpaca-py"
            )

        key = api_key or os.environ.get("ALPACA_API_KEY")
        secret = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise ValueError(
                "Alpaca API keys required. Pass api_key=/secret_key= or set "
                "ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
            )

        self._paper = paper
        self._client = TradingClient(key, secret, paper=paper)
        self._api_key = key
        self._secret_key = secret

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Fetch real-time NBBO quote.

        Detects OCC option symbols (e.g. SPY240315C00415000) vs equity symbols
        and routes to the appropriate Alpaca data endpoint.
        """
        import re
        is_option = bool(re.match(r'^[A-Z1-9]{1,6}\d{6}[CP]\d{8}$', symbol))
        try:
            if is_option:
                from alpaca.data.historical.option import OptionHistoricalDataClient
                from alpaca.data.requests import OptionLatestQuoteRequest
                from alpaca.data.enums import OptionsFeed
                dc = OptionHistoricalDataClient(self._api_key, self._secret_key)
                resp = dc.get_option_latest_quote(
                    OptionLatestQuoteRequest(
                        symbol_or_symbols=symbol,
                        feed=OptionsFeed.INDICATIVE,
                    )
                )
                q = resp[symbol]
            else:
                from alpaca.data.historical.stock import StockHistoricalDataClient
                from alpaca.data.requests import StockLatestQuoteRequest
                dc = StockHistoricalDataClient(self._api_key, self._secret_key)
                resp = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
                q = resp[symbol]
            bid = float(q.bid_price)
            ask = float(q.ask_price)
            return Quote(symbol=symbol, bid=bid, ask=ask, last=(bid + ask) / 2.0)
        except Exception:
            return None

    def get_account(self) -> AccountSnapshot:
        acct = self._client.get_account()
        return AccountSnapshot(
            net_liquidation=float(acct.equity),
            cash=float(acct.cash),
            margin_used=float(acct.maintenance_margin or 0),
            unrealized_pnl=float(acct.unrealized_pl or 0),
            timestamp=datetime.utcnow().isoformat(),
        )

    def is_market_open(self) -> bool:
        """True if the US equity/option market is currently open (Alpaca clock)."""
        try:
            return bool(self._client.get_clock().is_open)
        except Exception:
            return False

    def submit_option_order(
        self,
        contract: OptionContract,
        action: str,
        qty: int,
        order_type: str = "LMT",
        limit_price: Optional[float] = None,
        close: bool = False,
    ) -> Order:
        """Submit an option order.

        Args:
            close: If True, uses BTC/STC position_intent (closing an existing position).
                   If False (default), uses BTO/STO (opening a new position).
        """
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, PositionIntent

        symbol = _occ_symbol(contract)
        side = OrderSide.BUY if action == "BUY" else OrderSide.SELL
        now = datetime.utcnow().isoformat()

        if action == "BUY":
            intent = PositionIntent.BUY_TO_CLOSE if close else PositionIntent.BUY_TO_OPEN
        else:
            intent = PositionIntent.SELL_TO_CLOSE if close else PositionIntent.SELL_TO_OPEN
        common = dict(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            position_intent=intent,
        )

        if order_type == "LMT" and limit_price:
            req = LimitOrderRequest(**common, limit_price=round(limit_price, 2))
        else:
            req = MarketOrderRequest(**common)

        trade = self._client.submit_order(req)
        return Order(
            order_id=str(trade.id),
            contract=contract,
            action=action,
            qty=qty,
            order_type=order_type,
            limit_price=limit_price,
            submitted_at=now,
            status="pending",
        )

    def submit_hedge_order(
        self,
        underlying: str,
        qty: int,
        order_type: str = "MKT",
        limit_price: Optional[float] = None,
    ) -> Order:
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        action = "BUY" if qty > 0 else "SELL"
        abs_qty = abs(qty)
        side = OrderSide.BUY if qty > 0 else OrderSide.SELL
        now = datetime.utcnow().isoformat()

        if order_type == "LMT" and limit_price:
            req = LimitOrderRequest(
                symbol=underlying,
                qty=abs_qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 2),
            )
        else:
            req = MarketOrderRequest(
                symbol=underlying,
                qty=abs_qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )

        trade = self._client.submit_order(req)
        return Order(
            order_id=str(trade.id),
            contract=underlying,
            action=action,
            qty=abs_qty,
            order_type=order_type,
            limit_price=limit_price,
            submitted_at=now,
            status="pending",
        )

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._client.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False

    def get_open_orders(self) -> list[Order]:
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            orders = self._client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
            return [
                Order(
                    order_id=str(o.id),
                    contract=str(o.symbol),
                    action=o.side.value.upper(),
                    qty=int(o.qty),
                    order_type=o.order_type.value.upper(),
                    limit_price=float(o.limit_price) if o.limit_price else None,
                    submitted_at=str(o.submitted_at),
                    status="pending",
                )
                for o in orders
            ]
        except Exception:
            return []


# ── IBKR stub ─────────────────────────────────────────────────────────────────

class IBKRBroker:
    """Interactive Brokers broker via ib_insync.

    Requires:
        pip install ib_insync
        TWS or IB Gateway running on localhost:7497 (paper) or 7496 (live)

    This stub wires up the interface. Call connect() before use.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,        # 7497 = paper, 7496 = live
        client_id: int = 1,
    ):
        self._host = host
        self._port = port
        self._client_id = client_id
        self._ib = None

    def connect(self) -> None:
        try:
            from ib_insync import IB
        except ImportError:
            raise ImportError(
                "ib_insync not installed. Run: pip install ib_insync\n"
                "Also requires TWS or IB Gateway running on "
                f"{self._host}:{self._port}"
            )
        self._ib = IB()
        self._ib.connect(self._host, self._port, clientId=self._client_id)

    def disconnect(self) -> None:
        if self._ib:
            self._ib.disconnect()

    def get_quote(self, symbol: str) -> Optional[Quote]:
        if not self._ib:
            raise RuntimeError("Not connected. Call connect() first.")
        from ib_insync import Stock
        contract = Stock(symbol, "SMART", "USD")
        self._ib.qualifyContracts(contract)
        ticker = self._ib.reqMktData(contract, "", False, False)
        self._ib.sleep(0.5)
        if ticker.bid and ticker.ask:
            return Quote(symbol=symbol, bid=ticker.bid, ask=ticker.ask, last=ticker.last or ticker.close)
        return None

    def get_account(self) -> AccountSnapshot:
        if not self._ib:
            raise RuntimeError("Not connected.")
        vals = {v.tag: v.value for v in self._ib.accountValues()}
        return AccountSnapshot(
            net_liquidation=float(vals.get("NetLiquidation", 0)),
            cash=float(vals.get("TotalCashValue", 0)),
            margin_used=float(vals.get("MaintMarginReq", 0)),
            unrealized_pnl=float(vals.get("UnrealizedPnL", 0)),
            timestamp=datetime.utcnow().isoformat(),
        )

    def submit_option_order(
        self,
        contract: OptionContract,
        action: str,
        qty: int,
        order_type: str = "LMT",
        limit_price: Optional[float] = None,
    ) -> Order:
        if not self._ib:
            raise RuntimeError("Not connected.")
        from ib_insync import Option, LimitOrder, MarketOrder

        ib_contract = Option(
            contract.underlying,
            contract.expiry.replace("-", ""),
            contract.strike,
            contract.right,
            contract.exchange,
        )
        self._ib.qualifyContracts(ib_contract)

        if order_type == "LMT" and limit_price:
            ib_order = LimitOrder(action, qty, limit_price)
        else:
            ib_order = MarketOrder(action, qty)

        trade = self._ib.placeOrder(ib_contract, ib_order)
        now = datetime.utcnow().isoformat()

        return Order(
            order_id=str(trade.order.orderId),
            contract=contract,
            action=action,
            qty=qty,
            order_type=order_type,
            limit_price=limit_price,
            submitted_at=now,
            status="pending",
        )

    def submit_hedge_order(
        self,
        underlying: str,
        qty: int,
        order_type: str = "MKT",
        limit_price: Optional[float] = None,
    ) -> Order:
        if not self._ib:
            raise RuntimeError("Not connected.")
        from ib_insync import Stock, MarketOrder, LimitOrder

        action = "BUY" if qty > 0 else "SELL"
        abs_qty = abs(qty)

        ib_contract = Stock(underlying, "SMART", "USD")
        self._ib.qualifyContracts(ib_contract)

        if order_type == "LMT" and limit_price:
            ib_order = LimitOrder(action, abs_qty, limit_price)
        else:
            ib_order = MarketOrder(action, abs_qty)

        trade = self._ib.placeOrder(ib_contract, ib_order)
        now = datetime.utcnow().isoformat()

        return Order(
            order_id=str(trade.order.orderId),
            contract=underlying,
            action=action,
            qty=abs_qty,
            order_type=order_type,
            limit_price=limit_price,
            submitted_at=now,
            status="pending",
        )

    def cancel_order(self, order_id: str) -> bool:
        if not self._ib:
            return False
        # ib_insync requires the Trade object; simplified stub
        warnings.warn("[IBKR] cancel_order stub — not fully implemented")
        return False

    def get_open_orders(self) -> list[Order]:
        if not self._ib:
            return []
        trades = self._ib.openTrades()
        return [
            Order(
                order_id=str(t.order.orderId),
                contract=str(t.contract.symbol),
                action=t.order.action,
                qty=t.order.totalQuantity,
                order_type=t.order.orderType,
                limit_price=getattr(t.order, "lmtPrice", None),
                submitted_at="",
                status="pending",
            )
            for t in trades
        ]
