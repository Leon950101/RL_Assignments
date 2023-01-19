# Agent Part
# cd Documents/New/Courses/RL/Project/code
import json
import torch as th
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from gridworld import Gridworld # 22
# from gridworld_38 import Gridworld

# env = Gridworld()
env = make_vec_env(lambda: Gridworld(), n_envs=4)
obs = env.reset()
# Custom actor (pi) and value function (vf) networks
# of two layers of size 128 (default is 64) each with Relu activation function
# Note: an extra linear layer will be added on top of the pi and the vf nets, respectively
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[128, 128])
# Create the agent
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
start_time = time.time()
model.learn(total_timesteps=4000000)
end_time = time.time()
run_time = round(end_time - start_time)
print("Run Time: ", end="")
print(run_time, end="")
print("s")
model.save("Gridworld")

# del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_cartpole")

# Evaluation
# vec_env = model.get_env()
# obs = vec_env.reset()
env = Gridworld()
actions_real = ["move", "turnLeft", "turnRight", "pickMarker", "putMarker", "finish"]
T = 0
solved_optimal = 0
solved_not_optimal = 0
not_solved = 0

for i in range(0, 24000): # 24000
    with open('data/train/seq/'+str(i)+'_seq.json', 'r') as fcc_file_seq:
        fcc_data_seq = json.load(fcc_file_seq)
    optimal_seq = fcc_data_seq
    
    obs = env.reset_with_map_index(i)
    done = False
    output = {
        "sequence":[] 
        }

    while not done:
        T += 1
        action, _states = model.predict(obs, deterministic=True)
        output["sequence"].append(actions_real[action])
        obs, reward, done, info = env.step(action) # vec_env
        # env.render() # vec_env
        if T > 30: done = True # Control the length of output
        if done:
            # print("Env No." + str(i) + " is ", end="")
            if reward == 1: # solved
                # print("Solved", end="")
                if len(optimal_seq["sequence"]) == len(output["sequence"]):
                    # print(" - Optimal")
                    solved_optimal += 1
                else:
                    # print(" - Not Optimal")
                    solved_not_optimal += 1
            else: # not solved
                # print("Not Solved") 
                not_solved += 1
print("In Total: Solved Optimal: " + str(solved_optimal) + " | Solved Not Optimal: " + 
      str(solved_not_optimal) +" | Not Solved: " + str(not_solved))
                           
        # VecEnv resets automatically
        # if done:
        #   obs = env.reset()

    # Write as a json file
    # seq = json.dumps(output)
    # # Writing to sample.json
    # with open("/content/drive/My Drive/"+str(i)+"_sep.json", "w") as outfile:
    #     outfile.write(seq)

env.close()