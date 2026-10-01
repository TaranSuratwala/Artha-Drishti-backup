import os
import json
import logging
from datetime import datetime
from typing import List, Dict

from BrokerAdapter import BrokerAdapter, OrderRequest, OrderResponse, OrderStatus

logger = logging.getLogger(__name__)

class OrderManager:
    """
    Safety layer sitting between the AI Agent and the Broker.
    Enforces risk limits, daily loss limits, order count limits, and provides a kill switch.
    """
    
    def __init__(self, broker: BrokerAdapter, config, data_dir: str = 'data'):
        self.broker = broker
        self.config = config
        self.data_dir = data_dir
        self.audit_file = os.path.join(self.data_dir, 'order_audit.json')
        
        # Daily tracking state
        self.daily_orders = []
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        self.is_killed = False
        
        # Load risk limits from config
        self.max_order_value = getattr(self.config, 'MAX_ORDER_VALUE', 100000.0)
        self.max_daily_loss = getattr(self.config, 'MAX_DAILY_LOSS', 10000.0)
        self.max_position_pct = getattr(self.config, 'MAX_POSITION_SIZE_PCT', 20.0)
        self.max_orders_day = getattr(self.config, 'MAX_ORDERS_PER_DAY', 20)
        self.confirm_threshold = getattr(self.config, 'REQUIRE_CONFIRMATION_ABOVE', 50000.0)

    def _check_daily_reset(self):
        """Reset daily counters if a new day has started."""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_orders = []
            self.daily_pnl = 0.0
            self.last_reset_date = today
            logger.info("Daily order limits have been reset.")

    def validate_and_place(self, order: OrderRequest, user_confirmed: bool = False) -> OrderResponse:
        """
        Pre-trade validation and execution wrapper.
        Applies all safety checks before sending to the broker.
        """
        self._check_daily_reset()
        
        # 1. Kill switch check
        if self.is_killed:
            msg = "Trading is currently HALTED by the emergency kill switch."
            self._log_rejection(order, msg)
            return self._create_reject(order, msg)
            
        # 2. Daily order count limit
        if len(self.daily_orders) >= self.max_orders_day:
            msg = f"Daily order limit reached ({self.max_orders_day})."
            self._log_rejection(order, msg)
            return self._create_reject(order, msg)
            
        # 3. Get current price estimate for value checks
        try:
            ltp_map = self.broker.get_ltp([order.symbol])
            est_price = order.price if order.price else ltp_map.get(order.symbol, 0.0)
            
            if est_price <= 0:
                logger.warning(f"Could not get reliable price for {order.symbol}. Passing value checks for now.")
                est_value = 0
            else:
                est_value = est_price * order.quantity
        except Exception as e:
            logger.warning(f"Error fetching price for validation: {e}")
            est_value = 0
            
        # 4. Single order value limit
        if est_value > self.max_order_value:
            msg = f"Order value (₹{est_value:,.2f}) exceeds maximum allowed (₹{self.max_order_value:,.2f})."
            self._log_rejection(order, msg)
            return self._create_reject(order, msg)
            
        # 5. Position size check (vs total capital)
        if est_value > 0:
            funds = self.broker.get_funds()
            total_cap = funds.get("total_capital", 1.0)
            if (est_value / total_cap) * 100 > self.max_position_pct:
                msg = f"Order value exceeds max position size limit ({self.max_position_pct}% of capital)."
                self._log_rejection(order, msg)
                return self._create_reject(order, msg)
                
        # 6. Daily loss limit
        # This requires tracking realized PnL throughout the day. Simplified here.
        if self.daily_pnl <= -self.max_daily_loss:
            msg = f"Max daily loss limit reached (₹{self.max_daily_loss:,.2f}). Trading halted for the day."
            self._log_rejection(order, msg)
            return self._create_reject(order, msg)
            
        # 7. Confirmation gate
        if est_value > self.confirm_threshold and not user_confirmed:
            msg = f"High value order (₹{est_value:,.2f}) requires explicit confirmation."
            # We don't log this as a strict rejection, it's just a prompt back to the user/agent
            return self._create_reject(order, msg)
            
        # --- All Checks Passed. Execute ---
        logger.info(f"Order passed safety checks. Routing to broker: {order.side.value} {order.quantity} {order.symbol}")
        try:
            resp = self.broker.place_order(order)
            self.daily_orders.append(resp.order_id)
            self._log_audit(resp, "ACCEPTED")
            return resp
        except Exception as e:
            logger.error(f"Broker execution failed: {e}")
            msg = f"Broker execution failed: {str(e)}"
            self._log_rejection(order, msg)
            return self._create_reject(order, msg)

    def _create_reject(self, order: OrderRequest, reason: str) -> OrderResponse:
        return OrderResponse(
            order_id="REJECTED",
            status=OrderStatus.REJECTED,
            message=reason,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity
        )

    def _log_rejection(self, order: OrderRequest, reason: str):
        logger.warning(f"ORDER REJECTED [{order.symbol}]: {reason}")
        self._log_audit(self._create_reject(order, reason), "REJECTED")

    def _log_audit(self, response: OrderResponse, action: str):
        """Append to the central order audit log file."""
        os.makedirs(self.data_dir, exist_ok=True)
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "order_id": response.order_id,
                "symbol": response.symbol,
                "side": response.side,
                "quantity": response.quantity,
                "price": response.average_price,
                "status": response.status.value,
                "message": response.message
            }
            log = []
            if os.path.exists(self.audit_file):
                with open(self.audit_file, 'r') as f:
                    try:
                        log = json.load(f)
                    except:
                        pass
            log.append(entry)
            with open(self.audit_file, 'w') as f:
                json.dump(log, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def kill_switch(self) -> dict:
        """Emergency stop all trading."""
        self.is_killed = True
        logger.critical("🚨 TRADING KILL SWITCH ACTIVATED 🚨")
        
        # Try to cancel all open orders
        cancelled = 0
        try:
            for order in self.broker.get_order_history():
                if order.status == OrderStatus.PENDING:
                    self.broker.cancel_order(order.order_id)
                    cancelled += 1
        except Exception as e:
            logger.error(f"Error cancelling open orders during kill switch: {e}")
            
        return {
            "status": "killed", 
            "message": f"All trading halted. Cancelled {cancelled} pending orders."
        }

    def resume_trading(self) -> dict:
        """Resume trading after kill switch."""
        self.is_killed = False
        logger.warning("Trading resumed. Kill switch deactivated.")
        return {"status": "resumed", "message": "Trading resumed"}

    def get_audit_trail(self, limit=50) -> List[dict]:
        """Fetch recent audit log entries."""
        if not os.path.exists(self.audit_file):
            return []
        try:
            with open(self.audit_file, 'r') as f:
                log = json.load(f)
                return log[-limit:]
        except Exception:
            return []

    def get_daily_stats(self) -> dict:
        self._check_daily_reset()
        return {
            "orders_placed_today": len(self.daily_orders),
            "max_orders_allowed": self.max_orders_day,
            "daily_pnl": self.daily_pnl,
            "is_killed": self.is_killed
        }
