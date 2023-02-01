# 1011_38:best 0011_38:medium 0011_22:worse 1011_22:worst
import numpy as np
import time
import json
from itertools import count
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from gridworld import Gridworld
import matplotlib.pyplot as plt

env = Gridworld()
env.reset()
input = env.observation_space.shape[0]
# torch.manual_seed(543)

class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input, 128)
        self.affine2 = nn.Linear(128, 128)
        self.output_l = nn.Linear(128, 6)
        
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        action_prob = F.softmax(self.output_l(x), dim=1)
        return action_prob


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-4)

def select_action(state, t, op_seq, total_step, il_period, il_percent):
    state = torch.from_numpy(state).float().unsqueeze(0)
    # Normal Learning
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    # Imitation Learning
    if (total_step - 1) % il_period < int(il_period * il_percent): # as least 5%
        action = torch.tensor([env.action_dir[op_seq[t]]])
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()

def select_action_val(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item()

def finish_episode():
    R = 0
    policy_loss = []
    returns = deque()

    for r in policy.rewards[::-1]:
        R = r + env.gamma * R
        returns.appendleft(R)

    returns = torch.tensor(returns)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    return policy_loss

def main():
    
    running_reward = 0
    total_step = 0
    total_reward = []
    loss = 0
    max_step = 4000000
    log_interval = 500
    il_period = 5000 # < steady, > worse performance
    il_percent = 1.0
    # 0.0 | 0.01 | 0.05 | 0.1 | 0.2 | 0.5 | 1.0
    # 5 | 59 | 78 | 81 | 85 | 84 | 92
    l_total_step = 0
    start_time = time.time()
    for i_episode in count(1):
        state, op_seq = env.reset_il()
        ep_reward = 0
        for t in range(0, 100): # Don't infinite loop while learning (500)
            total_step += 1
            action = select_action(state, t, op_seq, total_step, il_period, il_percent)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            ep_reward += reward
            if done or total_step % il_period == 0:
                break

        running_reward = 0.001 * ep_reward + (1 - 0.001) * running_reward
        loss += finish_episode() / log_interval
        if i_episode % log_interval == 0:
            print('Total Step {}| Episode: {}| Ep Length: {:.2f}| Average reward: {:.2f}| Loss: {:.4f}'.format(
                  total_step, i_episode, (total_step-l_total_step)/log_interval, running_reward, loss))
            total_reward.append(running_reward)
            l_total_step = total_step
            loss = 0
            t_r = json.dumps(total_reward)
            with open('reward/reinforce_reward_1011_38_1.0.json', 'w') as outfile:
                outfile.write(t_r)
        if total_step > max_step:
            break
    
    end_time = time.time()
    run_time = round((end_time - start_time)/60)
    print("Run Time: ", end="")
    print(run_time, end="")
    print("min")
     
    # Plot
    x = np.arange(log_interval, i_episode, log_interval, dtype=int)
    y = total_reward
    fig, axs = plt.subplots()
    axs.plot(x, y, label="reinforce_average_reward")

    plt.legend()
    plt.show()

    # Evaluation
    actions_real = ["move", "turnLeft", "turnRight", "pickMarker", "putMarker", "finish"]

    solved_optimal = 0
    solved_not_optimal = 0
    not_solved = 0

    for i in range(100000, 102400):
        with open('data/val/seq/'+str(i)+'_seq.json', 'r') as fcc_file_seq:
            fcc_data_seq = json.load(fcc_file_seq)
        optimal_seq = fcc_data_seq
        
        obs = env.reset_val(i, 'val')
        done = False
        output = {
            "sequence":[] 
            }

        T = 0
        while not done:
            T += 1
            action = select_action_val(obs)
            output["sequence"].append(actions_real[action])
            obs, reward, done, info = env.step(action)
            if T > 500: done = True # Control the length of output
            if done:
                if reward == 1: # solved
                    if len(optimal_seq["sequence"]) == len(output["sequence"]):
                        solved_optimal += 1
                    else:
                        solved_not_optimal += 1
                else: # not solved
                    not_solved += 1

    print("In Total: Solved Optimal: " + str(solved_optimal) + " | Solved Not Optimal: " + 
        str(solved_not_optimal) +" | Not Solved: " + str(not_solved))
    print("Solved Optimal: {}%".format(round(solved_optimal/2400*100, 2)), end="")
    print(" | Solved: {}%".format(round((solved_optimal+solved_not_optimal)/2400*100, 2)))
    
    env.close()

if __name__ == '__main__':
    main()