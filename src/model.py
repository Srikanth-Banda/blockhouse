import numpy as np
from stable_baselines3 import DQN
from gym import spaces, Env

class TradingEnv(Env):
    def __init__(self, data, initial_inventory=1000):
        super(TradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        print(f'Headers of self.data: {self.data.head()}')
        self.current_step = 0
        self.inventory = initial_inventory
        self.total_steps = len(data)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(11)  # Actions: Sell 0% to 100% in increments of 10%
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)

    def step(self, action):
        sell_amount = (action / 10) * self.inventory
        current_price = self.data.iloc[self.current_step]['close']
        transaction_cost = (sell_amount * current_price) * 0.001  # Example cost

        self.inventory -= sell_amount
        reward = -transaction_cost  # Negative reward to minimize cost

        self.current_step += 1
        done = self.current_step >= self.total_steps or self.inventory <= 0

        obs = self.data.iloc[self.current_step][['close', 'volume']].values
        obs = np.append(obs, [self.inventory, self.current_step / self.total_steps, sell_amount])

        return obs, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.inventory = 1000
        obs = self.data.iloc[self.current_step][['close', 'volume']].values
        obs = np.append(obs, [self.inventory, 0, 0])
        return obs

def create_model(env):
    model = DQN('MlpPolicy', env, verbose=1)
    return model
