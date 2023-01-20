import json
from stable_baselines3 import PPO
from gridworld import Gridworld
from stable_baselines3.common.env_util import make_vec_env

env = Gridworld()
actions_real = ["move", "turnLeft", "turnRight", "pickMarker", "putMarker", "finish"]

# Load the trained model
model = PPO.load("Gridworld")

solved_optimal = 0
solved_not_optimal = 0
not_solved = 0

for i in range(100000, 102400): # val
    with open('data/val/seq/'+str(i)+'_seq.json', 'r') as fcc_file_seq:
        fcc_data_seq = json.load(fcc_file_seq)
    optimal_seq = fcc_data_seq
    
    obs = env.reset_val(i)
    done = False
    output = {
        "sequence":[] 
        }

    T = 0
    while not done:
        T += 1
        action, _states = model.predict(obs, deterministic=True)
        output["sequence"].append(actions_real[action])
        obs, reward, done, info = env.step(action)
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
                           
    # Write as a json file
    # seq = json.dumps(output)
    # # Writing to sample.json
    # with open("/content/drive/My Drive/"+str(i)+"_sep.json", "w") as outfile:
    #     outfile.write(seq)

env.close()