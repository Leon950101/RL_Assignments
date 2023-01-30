import json
from stable_baselines3 import PPO
from gridworld import Gridworld

# Evaluation
def val(low, high, label, out=False):
    env = Gridworld()
    actions_real = ["move", "turnLeft", "turnRight", "pickMarker", "putMarker", "finish"]
    model = PPO.load("model/ppo_gridworld")
    solved_optimal = 0
    solved_not_optimal = 0
    not_solved = 0

    for i in range(low, high):
        with open('data/'+label+'/seq/'+str(i)+'_seq.json', 'r') as fcc_file_seq:
            fcc_data_seq = json.load(fcc_file_seq)
        optimal_seq = fcc_data_seq
        
        obs = env.reset_val(i, label)
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
            # env.render()
            if T > 100: done = True # Control the length of output
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
            with open('data/my/'+str(i)+'_sep.json', 'w') as outfile:
                outfile.write(seq)

    print("In Total: Solved Optimal: " + str(solved_optimal) + " | Solved Not Optimal: " + 
        str(solved_not_optimal) +" | Not Solved: " + str(not_solved))
    print("Solved Optimal: {}%".format(round(solved_optimal/(high-low)*100, 2)))
    
    env.close()

# val(0, 24000, "train")
val(100000, 102400, "val") # , out=True
# val(0, 0, "test") # TODO: no optimal seq to read

# I.5
# a = input()
# obs = env.reset_file(a)

# while not done:
#     action, _states = model.predict(obs, deterministic=True)
#     output["sequence"].append(actions_real[action])
#     obs, reward, done, info = env.step(action)
#     if done:
#         if reward == 1: # solved
#             print("Solved")
#         else: # not solved
#             print("Not Solved")

# print(output)
# env.close()