"""
Reinforcement Learning Trading Agent
Bridging ML Predictions with Paper Trading feedback using PyTorch (Deep Q-Network).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import logging

logger = logging.getLogger(__name__)

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class RLTradingAgent:
    def __init__(self, state_size=6, action_size=3):
        # State: [prob_3d, prob_5d, prob_7d, prob_15d, current_position, unrealized_pnl]
        # Action: 0 = HOLD, 1 = BUY, 2 = SELL
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # Replay memory
        self.memory = deque(maxlen=2000)
        
        # Neural Nets
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state_tensor)
        return torch.argmax(act_values[0]).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).to(self.device)

        # Get current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        # Train model
        loss = self.criterion(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class TradingEnvironment:
    """
    Simulates a gym-like environment connecting MLPredictor with PaperBroker.
    """
    def __init__(self, broker, order_manager, ml_predictor, ticker="RELIANCE"):
        self.broker = broker
        self.order_manager = order_manager
        self.predictor = ml_predictor
        self.ticker = ticker
        
        # For simulation, we would iterate through a dataframe.
        # In live mode, step() is called on daily cron tick.
        self.current_step = 0
        self.historical_data = None
        self.initial_capital = broker.get_funds()['total_capital']

    def reset(self, df):
        self.historical_data = df
        self.current_step = 0
        self.broker.reset() # Reset paper broker
        return self._get_state()

    def _get_state(self):
        # 1. Fetch current market row
        row = self.historical_data.iloc[self.current_step]
        
        # 2. Get ML Predictions for multiple timeframes
        # (Assuming ml_predictor has a predict_row or similar method)
        # Mocking the 4 timeframes for demonstration since unified model 
        # may require complex tensor building
        prob_3d = getattr(row, 'pred_3d', 0.5)
        prob_5d = getattr(row, 'pred_5d', 0.5)
        prob_7d = getattr(row, 'pred_7d', 0.5)
        prob_15d = getattr(row, 'pred_15d', 0.5)
        
        # 3. Get Portfolio State
        positions = self.broker.get_positions()
        pos = next((p for p in positions if p.symbol == self.ticker and p.quantity > 0), None)
        
        current_pos = 1 if pos else 0
        unrealized_pnl = pos.pnl if pos else 0.0
        
        # Normalize PnL
        cap = self.broker.get_funds()['total_capital']
        norm_pnl = unrealized_pnl / cap if cap > 0 else 0
        
        return np.array([prob_3d, prob_5d, prob_7d, prob_15d, current_pos, norm_pnl])

    def step(self, action):
        from BrokerAdapter import OrderRequest, OrderSide, OrderType, Segment
        
        row = self.historical_data.iloc[self.current_step]
        price = row['close']
        
        # Previous capital
        prev_cap = self.broker.get_funds()['total_capital']
        
        # Execute Action
        if action == 1: # BUY
            req = OrderRequest(
                symbol=self.ticker, side=OrderSide.BUY,
                quantity=10, order_type=OrderType.MARKET,
                price=price, product=Segment.CASH_DELIVERY
            )
            # Submit to safety layer
            self.order_manager.validate_and_place(req)
            
        elif action == 2: # SELL
            req = OrderRequest(
                symbol=self.ticker, side=OrderSide.SELL,
                quantity=10, order_type=OrderType.MARKET,
                price=price, product=Segment.CASH_DELIVERY
            )
            self.order_manager.validate_and_place(req)
            
        # Move forward in time (Simulate market closing and filling orders)
        # The PaperBroker would normally fill these via L1 ticks, but for 
        # this RL simulation we assume fill at close.
        
        self.current_step += 1
        done = self.current_step >= len(self.historical_data) - 1
        
        # Update positions with new price
        self.broker._get_current_price = lambda x: self.historical_data.iloc[self.current_step]['close']
        
        # Reward = Change in Total Portfolio Capital (Realized + Unrealized)
        new_cap = self.broker.get_funds()['total_capital']
        
        # We calculate reward as the delta in total capital
        reward = new_cap - prev_cap
        
        next_state = self._get_state() if not done else np.zeros(6)
        
        return next_state, reward, done, {}

# Example Training Loop
def train_rl_agent(env, agent, episodes=50):
    for e in range(episodes):
        state = env.reset(env.historical_data)
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.replay() # Train network
            
            if done:
                agent.update_target_network()
                logger.info(f"Episode {e+1}/{episodes} - Total Profit: ₹{total_reward:.2f} - Epsilon: {agent.epsilon:.2f}")
                break
