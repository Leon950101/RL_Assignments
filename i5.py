import json
import argparse
from stable_baselines3 import PPO
from gridworld import Gridworld

parser = argparse.ArgumentParser(description='I5')
parser.add_argument('file_path', type=str, metavar='F', help='file path')
args = parser.parse_args()

def val(file_path):
    env = Gridworld()
    actions_real = ["move", "turnLeft", "turnRight", "pickMarker", "putMarker", "finish"]
    model = PPO.load("model/ppo_gridworld")
        
    obs = env.reset_file(file_path)
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
        if T > 500: done = True # Control the length of output
        if done:
            if reward == 1: 
                print("Solved")
            else:
                print("Not Solved") 
    
    print(output)   
    env.close()

val(args.file_path)

