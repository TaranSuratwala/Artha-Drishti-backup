import os
import json
import uuid
import random
import logging
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict

from BrokerAdapter import (
    BrokerAdapter, OrderRequest, OrderResponse, Position, Holding,
    OrderSide, BrokerOrderType, OrderStatus, ProductType
)

logger = logging.getLogger(__name__)

class PaperBroker(BrokerAdapter):
    """
    A simulated broker for paper trading. 
    Executes orders immediately against live prices with small random slippage.
    State is persisted to disk so paper trades survive restarts.
    """
    
    def __init__(self, initial_capital: float = 1000000.0, data_pipeline=None, data_dir: str = 'data'):
        self.initial_capital = initial_capital
        self.data_pipeline = data_pipeline
        self.data_dir = data_dir
        
        self.state_file = os.path.join(self.data_dir, 'paper_portfolio.json')
        self.audit_file = os.path.join(self.data_dir, 'paper_trades.json')
        
        # State variables
        self.orders: Dict[str, OrderResponse] = {}
        self.positions: Dict[str, Position] = {}
        self.capital = initial_capital
        self.used_margin = 0.0
        
        self._connected = False
        self._load_state()

    def connect(self, credentials: dict) -> bool:
        logger.info("✅ Paper Broker connected. Simulated execution enabled.")
        self._connected = True
        return True

    def is_connected(self) -> bool:
        return self._connected

    def _get_current_price(self, symbol: str) -> float:
        """Fetch current price using data pipeline, fallback to yfinance."""
        if self.data_pipeline:
            try:
                hist = self.data_pipeline.get_ticker_history(symbol)
                if hist:
                    last_row = hist[-1] if isinstance(hist[-1], dict) else dict(hist[-1])
                    return float(last_row.get("close", 0.0))
            except Exception as e:
                logger.warning(f"Failed to get price from pipeline for {symbol}: {e}")
        
        # Fallback to yfinance
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}.NS")
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception as e:
            logger.warning(f"Failed to get price from yfinance for {symbol}: {e}")
            
        # Last resort fallback to a random plausible price just so the backtest/demo works
        raise ValueError(f"Could not fetch live price for {symbol}. Order rejected.")

    def place_order(self, order: OrderRequest) -> OrderResponse:
        if not self._connected:
            raise ConnectionError("Broker not connected.")
            
        order_id = f"PAPER-{uuid.uuid4().hex[:8].upper()}"
        placed_at = datetime.now()
        
        # Simulate price lookup and slippage
        try:
            current_price = self._get_current_price(order.symbol)
        except ValueError as e:
            return OrderResponse(order_id, OrderStatus.REJECTED, str(e), order.symbol, order.side.value, order.quantity)
        
        if order.order_type == BrokerOrderType.LIMIT:
            if order.price is None:
                return OrderResponse(order_id, OrderStatus.REJECTED, "Limit price required", order.symbol, order.side.value, order.quantity)
                
            # Naive fill logic: if BUY limit is above current price, fill it. Else PENDING.
            if order.side == OrderSide.BUY and order.price >= current_price:
                fill_price = order.price
            elif order.side == OrderSide.SELL and order.price <= current_price:
                fill_price = order.price
            else:
                # Leaves order pending - in a real paper broker we'd need a background task to check prices
                # For simplicity here, we reject it for immediate feedback to agent
                return OrderResponse(order_id, OrderStatus.REJECTED, "Limit price not met (simulated restriction)", order.symbol, order.side.value, order.quantity)
        else:
            # Market order: Add realistic slippage for paper trading (0.1% to 0.5%)
            slippage_pct = random.uniform(0.001, 0.005)
            if order.side == OrderSide.BUY:
                fill_price = current_price * (1 + slippage_pct)
            else:
                fill_price = current_price * (1 - slippage_pct)
        
        # Calculate NSE transaction costs
        try:
            from nse_cost_model import NSECostModel, Segment
            cost_model = NSECostModel()
            is_buy = order.side == OrderSide.BUY
            segment = Segment.CASH_INTRADAY if getattr(order, 'product', None) and order.product.value == "MIS" else Segment.CASH_DELIVERY
            txn_cost_data = cost_model.compute("BUY" if is_buy else "SELL", fill_price, order.quantity, segment)
            txn_cost = txn_cost_data['total']
        except ImportError:
            txn_cost = 0.0

        # Check funds
        cost = (fill_price * order.quantity) + txn_cost
        if order.side == OrderSide.BUY and cost > self.capital:
            return OrderResponse(order_id, OrderStatus.REJECTED, f"Insufficient funds. Required: ₹{cost:.2f}, Available: ₹{self.capital:.2f}", order.symbol, order.side.value, order.quantity)
            
        # Update capital
        if order.side == OrderSide.BUY:
            self.capital -= cost
        else:
            self.capital += (fill_price * order.quantity) - txn_cost
            
        # Create response
        resp = OrderResponse(
            order_id=order_id,
            status=OrderStatus.COMPLETE,
            message="Simulated order filled successfully",
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            filled_quantity=order.quantity,
            average_price=fill_price,
            placed_at=placed_at
        )
        
        self.orders[order_id] = resp
        self._update_position(order, fill_price)
        self._save_state()
        self._log_audit(resp)
        
        return resp

    def _update_position(self, order: OrderRequest, fill_price: float):
        """Update positions tracking."""
        pos_key = f"{order.symbol}_{order.product.value}"
        
        if pos_key not in self.positions:
            qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
            self.positions[pos_key] = Position(
                symbol=order.symbol,
                exchange=order.exchange,
                quantity=qty,
                average_price=fill_price,
                last_price=fill_price,
                pnl=0.0,
                product=order.product.value
            )
        else:
            pos = self.positions[pos_key]
            
            # Simple average price calculation
            if order.side == OrderSide.BUY:
                if pos.quantity >= 0:
                    total_value = (pos.quantity * pos.average_price) + (order.quantity * fill_price)
                    pos.quantity += order.quantity
                    pos.average_price = total_value / pos.quantity
                else:
                    # Covering short
                    pos.quantity += order.quantity
                    if pos.quantity > 0:
                        pos.average_price = fill_price # Flipped to long
            else:
                if pos.quantity <= 0:
                    total_value = (abs(pos.quantity) * pos.average_price) + (order.quantity * fill_price)
                    pos.quantity -= order.quantity
                    pos.average_price = total_value / abs(pos.quantity)
                else:
                    # Selling long
                    pos.quantity -= order.quantity
                    if pos.quantity < 0:
                        pos.average_price = fill_price # Flipped to short
            
            # Remove empty positions
            if pos.quantity == 0:
                del self.positions[pos_key]

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders and self.orders[order_id].status == OrderStatus.PENDING:
            self.orders[order_id].status = OrderStatus.CANCELLED
            self._save_state()
            return True
        return False

    def get_order_status(self, order_id: str) -> OrderResponse:
        if order_id in self.orders:
            return self.orders[order_id]
        raise ValueError(f"Order {order_id} not found")

    def get_positions(self) -> List[Position]:
        # Update PnL before returning
        for pos in self.positions.values():
            pos.last_price = self._get_current_price(pos.symbol)
            if pos.quantity > 0:
                pos.pnl = (pos.last_price - pos.average_price) * pos.quantity
            else:
                pos.pnl = (pos.average_price - pos.last_price) * abs(pos.quantity)
        return list(self.positions.values())

    def get_holdings(self) -> List[Holding]:
        # Simplified: treat all CNC positions as holdings
        positions = self.get_positions()
        holdings = []
        for p in positions:
            if p.product == "CNC" and p.quantity > 0:
                holdings.append(Holding(
                    symbol=p.symbol,
                    quantity=p.quantity,
                    average_price=p.average_price,
                    last_price=p.last_price,
                    pnl=p.pnl,
                    day_change_pct=0.0 # Simplify
                ))
        return holdings

    def get_funds(self) -> dict:
        return {
            "available_capital": self.capital,
            "used_margin": self.used_margin,
            "total_capital": self.capital + self.used_margin,
            "daily_realized_pnl": self.capital - self.initial_capital  # Simplified for paper broker
        }

    def get_ltp(self, symbols: List[str]) -> Dict[str, float]:
        return {s: self._get_current_price(s) for s in symbols}

    def get_order_history(self) -> List[OrderResponse]:
        # Return today's orders (simplified to all for paper broker)
        return list(self.orders.values())
        
    def reset(self):
        """Reset paper broker state."""
        self.orders = {}
        self.positions = {}
        self.capital = self.initial_capital
        self._save_state()

    def _save_state(self):
        """Save paper trading state to JSON."""
        os.makedirs(self.data_dir, exist_ok=True)
        try:
            state = {
                "capital": self.capital,
                "positions": [
                    {
                        "symbol": p.symbol, "exchange": p.exchange, "quantity": p.quantity,
                        "average_price": p.average_price, "product": p.product
                    } for p in self.positions.values()
                ]
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save paper broker state: {e}")

    def _load_state(self):
        """Load paper trading state from JSON."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.capital = state.get("capital", self.initial_capital)
                    for p in state.get("positions", []):
                        pos_key = f"{p['symbol']}_{p['product']}"
                        self.positions[pos_key] = Position(
                            symbol=p['symbol'], exchange=p['exchange'], quantity=p['quantity'],
                            average_price=p['average_price'], last_price=p['average_price'],
                            pnl=0.0, product=p['product']
                        )
                logger.info(f"Loaded paper trading state. Capital: ₹{self.capital:.2f}")
            except Exception as e:
                logger.error(f"Failed to load paper broker state: {e}")

    def _log_audit(self, order: OrderResponse):
        """Append to audit log."""
        try:
            entry = {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "price": order.average_price,
                "status": order.status.value,
                "time": order.placed_at.isoformat() if order.placed_at else datetime.now().isoformat()
            }
            # Append atomically using jsonlines
            with open(self.audit_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write paper trade audit log: {e}")
