import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO

class RateEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=np.log10(1e-5), high=np.log10(1), shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.last_loss = None
    def step(self, action):
        reward = -self.last_loss if self.last_loss is not None else 0.0
        self.last_loss = None
        return np.zeros(1, dtype=np.float32), reward, True, {}
    def reset(self):
        return np.zeros(1, dtype=np.float32)
    def set_loss(self, loss):
        self.last_loss = loss

def make_agent(env):
    model = PPO('MlpPolicy', env, verbose=0)
    return PPOWrapper(model, env)

class PPOWrapper:
    def __init__(self, model, env):
        self.model = model
        self.env = env
    def sample(self):
        obs = np.zeros(1, dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=False)
        return 10 ** action[0]
    def observe(self, loss, throughput):
        self.env.set_loss(loss)
        self.model.learn(total_timesteps=1)
