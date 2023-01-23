import json
from stable_baselines3 import PPO
from gridworld import Gridworld
# from stable_baselines3.common.env_util import make_vec_env

# Evaluation
# vec_env = model.get_env()
# obs = vec_env.reset()
def val(low, high, label, out=False):
    env = Gridworld()
    actions_real = ["move", "turnLeft", "turnRight", "pickMarker", "putMarker", "finish"]
    model = PPO.load("ppo_gridworld")
    solved_optimal = 0
    solved_not_optimal = 0
    not_solved = 0

    for i in range(low, high):
        with open('data/'+label+'/seq/'+str(i)+'_seq.json', 'r') as fcc_file_seq:
            fcc_data_seq = json.load(fcc_file_seq)
        optimal_seq = fcc_data_seq
        
        obs = env.reset_with_map_index(i)
        done = False
        output = {
            "sequence":[] 
            }

        T = 0
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
        # Write as a json file
        if out:
            seq = json.dumps(output)
            # Writing to sample.json
            with open('data/my/'+str(i)+'_sep.json', 'w') as outfile:
                outfile.write(seq)

    print("In Total: Solved Optimal: " + str(solved_optimal) + " | Solved Not Optimal: " + 
        str(solved_not_optimal) +" | Not Solved: " + str(not_solved))
    print("Solved Optimal: {}%".format(round(solved_optimal/2400*100, 2)))
    
    env.close()

# I.5
# import json
# from stable_baselines3 import PPO
# from gridworld import Gridworld

# env = Gridworld()
# actions_real = ["move", "turnLeft", "turnRight", "pickMarker", "putMarker", "finish"]

# # Load the trained model
# model = PPO.load("ppo_gridworld")

# a = input()
# obs = env.reset_file(a)
# done = False
# output = {
#     "sequence":[] 
#     }

# while not done:
#     action, _states = model.predict(obs, deterministic=True)
#     output["sequence"].append(actions_real[action])
#     obs, reward, done, info = env.step(action)
#     if done:
#         if reward == 1: # solved
#             print("Solved")
#         else: # not solved
#             print("Not Solved")

# seq = json.dumps(output)
# print(seq)
# env.close()