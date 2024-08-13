import numpy as np
from src.env.trading_env import TradingEnv
from src.agents.dqn_agent import DQNAgent
import pandas as pd

def train_dqn_agent(csv_file, episodes=1000):
    df = pd.read_csv(csv_file)
    env = TradingEnv(df)
    agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
        agent.replay(32)

    agent.model.save('models/dqn_trading_agent.h5')

if __name__ == '__main__':
    train_dqn_agent('data/processed/stock_data.csv')
