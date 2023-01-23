# Agent Part
# cd Documents/New/Courses/RL/Project/code
import json
import torch as th
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from gridworld import Gridworld
from stable_baselines3.common.logger import configure
from evaluation import val

# env = Gridworld()
env = make_vec_env(lambda: Gridworld(), n_envs=4)
obs = env.reset()

# Custom actor (pi) and value function (vf) networks
# of two layers of size 128 (default is 64) each with Relu activation function
# Note: an extra linear layer will be added on top of the pi and the vf nets, respectively
policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[128, 128])
# Create the agent
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
# Set new logger for plot
log_path = "log/"
new_logger = configure(log_path, ["stdout", "json"]) # ["stdout", "csv", "log", "tensorboard", "json"]
model.set_logger(new_logger)
start_time = time.time()
model.learn(total_timesteps=4000000)
end_time = time.time()
run_time = round((end_time - start_time)/60)
print("Run Time: ", end="")
print(run_time, end="")
print("min")
model.save("ppo_gridworld")

# del model # remove to demonstrate saving and loading
# model = PPO.load("ppo_cartpole")
val(0, 24000, "train")
val(100000, 102400, "val", out=True)
# val(0, 0, "test")