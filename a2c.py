from stable_baselines3 import A2C
import time
import torch as th
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

from gridworld import Gridworld

# env = Gridworld()
env = make_vec_env(lambda: Gridworld(), n_envs=4)

policy_kwargs=dict(activation_fn=th.nn.ReLU, net_arch=[128, 128], optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)) 
model = A2C("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
# Set new logger
log_path = "log/"
new_logger = configure(log_path, ["stdout", "json"]) # ["stdout", "csv", "log", "tensorboard", "json"]
model.set_logger(new_logger)
start_time = time.time()
model.learn(total_timesteps=10000000)
end_time = time.time()
run_time = round((end_time - start_time)/60)
print("Run Time: ", end="")
print(run_time, end="")
print("min")
model.save("model/a2c_gridworld")

