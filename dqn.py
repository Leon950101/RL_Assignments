import gym

from stable_baselines3 import DQN 
from gridworld import Gridworld

env = Gridworld()

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000, log_interval=1000)
model.save("model/dqn_gridworld")

