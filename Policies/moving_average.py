from bitcoin_trading.agents.trading_policy import TradingPolicy
import talib

class MAPolicy(TradingPolicy):
    def __init__(self, short_period=10, long_period=20):
        self.short_period = short_period
        self.long_period = long_period
        self.prices = []

    def get_action(self, obs):
        self.prices.append(obs[0])  # Append the current price to the prices list

        if len(self.prices) < self.long_period:
            return 0  # Hold if not enough data points

        short_ma = talib.SMA(self.prices, timeperiod=self.short_period)[-1]
        long_ma = talib.SMA(self.prices, timeperiod=self.long_period)[-1]

        if short_ma > long_ma:
            return 1  # Buy if short MA crosses above long MA
        elif short_ma < long_ma:
            return 2  # Sell if short MA crosses below long MA
        else:
            return 0  # Hold if no crossover