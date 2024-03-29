import time
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env

from gridworld import Gridworld
# 1011_22 best 0011_38 medium 1011_38 worse 0011_22 worst
# env = Gridworld()
# Paralell multiple envs, single env will act worse
env = make_vec_env(lambda: Gridworld(), n_envs=4)

# Custom actor (pi) and value function (vf) networks of two layers of size 128 (default is 64) each with Relu activation function
# Note: an extra linear layer will be added on top of the pi and the vf nets, respectively
policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[128, 128]) # 64 64 4m 84 | 128 128 4m 91 | 128 128 6m 93
# Create the agent
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
# Set new logger for plot
log_path = "log/1011_38/"
new_logger = configure(log_path, ["stdout", "json"]) # ["stdout", "csv", "log", "tensorboard", "json"]
model.set_logger(new_logger)
# Record the running time
start_time = time.time()
# Train the agent
model.learn(total_timesteps=4000000)
# Print the running time
end_time = time.time()
run_time = round((end_time - start_time)/60)
print("Run Time: ", end="")
print(run_time, end="")
print("min")
# Save the model
model.save("model/1011_38/ppo")

