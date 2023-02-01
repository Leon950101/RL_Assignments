import json
from stable_baselines3 import PPO, A2C
from gridworld import Gridworld

# Evaluation
def val(low, high, label, out=False):
    env = Gridworld()
    actions_real = ["move", "turnLeft", "turnRight", "pickMarker", "putMarker", "finish"]
    model = PPO.load("model/ppo") # Maybe multiple models to vote (least solved sequence wins)
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
            if T > 100: done = True # Control the length of output
            if done:
                if reward == 1: # solved
                    if len(optimal_seq["sequence"]) == len(output["sequence"]):
                        solved_optimal += 1
                    else:
                        solved_not_optimal += 1
                else: # not solved
                    not_solved += 1
        # Write as a json file
        if out:
            seq = json.dumps(output)
            with open('data/my/'+str(i)+'_sep.json', 'w') as outfile:
                outfile.write(seq)

    print("In Total: Solved Optimal: " + str(solved_optimal) + " | Solved Not Optimal: " + 
        str(solved_not_optimal) +" | Not Solved: " + str(not_solved))
    print("Solved Optimal: {}%".format(round(solved_optimal/(high-low)*100, 2)), end="")
    print(" | Solved: {}%".format(round((solved_optimal+solved_not_optimal)/(high-low)*100, 2)))
    
    env.close()

val(100000, 102400, "val", out=False)