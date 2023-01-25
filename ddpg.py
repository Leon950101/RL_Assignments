import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import time
import torch as th
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env

from gridworld import Gridworld

# Can only singel env
env = Gridworld()

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[128, 128])
model = DDPG("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1) # action_noise=action_noise, 
model.learn(total_timesteps=1000000, log_interval=100)
model.save("model/ddpg_gridworld")

