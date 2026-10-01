from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict
from datetime import datetime

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class BrokerOrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"

class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class ProductType(Enum):
    CNC = "CNC"       # Delivery
    MIS = "MIS"       # Intraday
    NRML = "NRML"     # F&O normal

@dataclass
class OrderRequest:
    symbol: str
    exchange: str = "NSE"
    side: OrderSide = OrderSide.BUY
    order_type: BrokerOrderType = BrokerOrderType.MARKET
    quantity: int = 0
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    product: ProductType = ProductType.CNC
    tag: str = ""  # user-defined order tag

@dataclass
class OrderResponse:
    order_id: str
    status: OrderStatus
    message: str
    symbol: str = ""
    side: str = ""
    quantity: int = 0
    filled_quantity: int = 0
    average_price: float = 0.0
    placed_at: Optional[datetime] = None

@dataclass
class Position:
    symbol: str
    exchange: str
    quantity: int
    average_price: float
    last_price: float
    pnl: float
    product: str

@dataclass
class Holding:
    symbol: str
    quantity: int
    average_price: float
    last_price: float
    pnl: float
    day_change_pct: float

class BrokerAdapter(ABC):
    """Abstract base class for all broker integrations."""
    
    @abstractmethod
    def connect(self, credentials: dict) -> bool:
        """Connect to the broker API and establish session."""
        pass
        
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connection is active."""
        pass
        
    @abstractmethod
    def place_order(self, order: OrderRequest) -> OrderResponse:
        """Place a new order."""
        pass
        
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        pass
        
    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderResponse:
        """Get the status of a specific order."""
        pass
        
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        pass
        
    @abstractmethod
    def get_holdings(self) -> List[Holding]:
        """Get long-term holdings (CNC)."""
        pass
        
    @abstractmethod
    def get_funds(self) -> dict:
        """Get available funds and margin information."""
        pass
        
    @abstractmethod
    def get_ltp(self, symbols: List[str]) -> Dict[str, float]:
        """Get the Last Traded Price for a list of symbols."""
        pass
        
    @abstractmethod
    def get_order_history(self) -> List[OrderResponse]:
        """Get history of all orders for the current day."""
        pass
