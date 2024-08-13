import gym
from gym import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(df.columns),), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self.df.iloc[self.current_step].values

    def step(self, action):
        self.current_step += 1
        reward = self._take_action(action)
        done = self.current_step >= len(self.df) - 1
        obs = self.df.iloc[self.current_step].values
        return obs, reward, done, {}

    def _take_action(self, action):
        # Implement the logic for taking an action (buy, sell, hold)
        # For simplicity, we return a random reward
        return np.random.randn()
