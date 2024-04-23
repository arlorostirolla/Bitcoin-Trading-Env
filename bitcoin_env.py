import numpy as np
import requests
import json
import talib
import time
from abc import ABC, abstractmethod

class TradingPolicy(ABC):
    def __init__(self):
        self.env = None

    @abstractmethod
    def set_env(self, env):
        self.env = env

    def get_action(self, obs):
        # Implement your trading strategy here
        # Return 0 for hold, 1 for buy, 2 for sell
        return 0  # Replace with your trading logic

class BitcoinMarketEnv:
    def __init__(self, is_live=True, market_data=None, window_size=20, api_delay=1):
        self.is_live = is_live
        self.market_data = market_data
        self.current_step = 0
        self.current_features = None
        self.holdings = 0
        self.balance = 10000  # Initial balance in USD
        self.simulation_data = []
        self.window_size = window_size
        self.api_delay = api_delay
        self.last_api_call_time = 0

    def reset(self):
        self.current_step = 0
        if self.is_live:
            self.current_features = self._get_live_data()
        else:
            if self.market_data is None:
                raise ValueError("Market data is required for non-live environment.")
            self.current_features = self.market_data[0]
        self.holdings = 0
        self.balance = 10000
        self.simulation_data = []
        return self._get_obs()

    def step(self, action):
        if self.is_live:
            self.current_features = self._get_live_data()
        else:
            self.current_features = self.market_data[self.current_step]
        self.simulation_data.append(self.current_features)

        # Execute the action (buy, sell, or hold)
        if action == 1:  # Buy
            price = float(self.current_features["price"])
            amount = self.balance // price
            self.holdings += amount
            self.balance -= amount * price
            print(f"Bought {amount} BTC at price {price}, balance: {self.balance}, holdings: {self.holdings}")
        elif action == 2:  # Sell
            price = float(self.current_features["price"])
            self.balance += self.holdings * price
            print(f"Sold {self.holdings} BTC at price {price}, balance: {self.balance}, holdings: 0")
            self.holdings = 0

        # Calculate reward based on the action and market movement
        price = float(self.current_features["price"])
        portfolio_value = self.holdings * price + self.balance
        reward = portfolio_value - 10000

        # Move to the next step
        self.current_step += 1
        if not self.is_live and self.current_step >= len(self.market_data):
            done = True
        else:
            done = False
        obs = self._get_obs()

        return obs, reward, done, {}

    def _get_obs(self):
        # Extract additional features from the simulation data
        if len(self.simulation_data) >= self.window_size:
            closing_prices = np.array([float(data["price"]) for data in self.simulation_data[-self.window_size:]])
            volume = np.array([float(data["volume"]) for data in self.simulation_data[-self.window_size:]])

            # Calculate technical indicators
            rsi = talib.RSI(closing_prices)[-1]
            macd, _, _ = talib.MACD(closing_prices)
            macd = macd[-1]
            bollinger_upper, bollinger_middle, bollinger_lower = talib.BBANDS(closing_prices)
            bollinger_upper = bollinger_upper[-1]
            bollinger_lower = bollinger_lower[-1]

            # Return the current features as observation
            return np.array([
                float(self.current_features["price"]),
                volume[-1],
                rsi,
                macd,
                bollinger_upper,
                bollinger_lower,
                self.holdings,
                self.balance
            ])
        else:
            return np.array([
                float(self.current_features["price"]),
                0,
                50,
                0,
                float(self.current_features["price"]),
                float(self.current_features["price"]),
                self.holdings,
                self.balance
            ])

    def _get_live_data(self):
        # Check if enough time has passed since the last API call
        elapsed_time = time.time() - self.last_api_call_time
        if elapsed_time < self.api_delay:
            time.sleep(self.api_delay - elapsed_time)

        # requesting data from url
        key = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        data = requests.get(key)
        data = data.json()

        # Get volume data
        volume_key = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
        volume_data = requests.get(volume_key)
        volume_data = volume_data.json()
        data["volume"] = volume_data["volume"]

        self.last_api_call_time = time.time()  # Update the last API call time

        return data

class SimpleMovingAveragePolicy(TradingPolicy):
    def __init__(self, window_size=10):
        super().__init__()
        self.window_size = window_size

    def set_env(self, env):
        self.env = env

    def get_action(self, obs):
        if len(self.env.simulation_data) >= self.window_size:
            closing_prices = np.array([float(data["price"]) for data in self.env.simulation_data[-self.window_size:]])
            sma = np.mean(closing_prices)
            current_price = float(self.env.current_features["price"])

            if current_price > sma:
                return 1  # Buy
            elif current_price < sma:
                return 2  # Sell
        return 0  # Hold

# Use historical market data
historical_data = []

print("Running with historical market data...")
env = BitcoinMarketEnv(is_live=False, market_data=historical_data)
policy = SimpleMovingAveragePolicy(window_size=10)
policy.set_env(env)

obs = env.reset()
done = False
while not done:
    action = policy.get_action(obs)
    obs, reward, done, _ = env.step(action)
    print(f"Step: {env.current_step}, Action: {action}, Reward: {reward}, Done: {done}")

print("Simulation completed.")

# Use live market data
print("Running with live market data...")
env = BitcoinMarketEnv(is_live=True, api_delay=1)  # Set api_delay to 1 second
policy = SimpleMovingAveragePolicy(window_size=10)
policy.set_env(env)

obs = env.reset()
done = False
while not done:
    action = policy.get_action(obs)
    obs, reward, done, _ = env.step(action)
    print(f"Step: {env.current_step}, Action: {action}, Reward: {reward}, Done: {done}")

print("Live trading completed.")