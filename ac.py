# 1011 38
import numpy as np
import time
import json
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from gridworld import Gridworld
import matplotlib.pyplot as plt

env = Gridworld()
input = env.observation_space.shape[0]
# torch.manual_seed(543)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input, 128)
        self.affine2 = nn.Linear(128, 128)
        self.action_head = nn.Linear(128, 6) # Actor
        self.value_head = nn.Linear(128, 1) # Critic

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        action_prob = F.softmax(self.action_head(x), dim=-1) # Actor: Probability of each action over the action space
        state_values = self.value_head(x) # Critic: evaluates value being in the state s_t
        return action_prob, state_values


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=5e-4)

def select_action(state, t, op_seq, total_step, il_period, il_percent):
    state = torch.from_numpy(state).float()
    # Normal Learning
    probs, state_value = policy(state)
    m = Categorical(probs)
    action = m.sample()
    # Imitation Learning
    if (total_step - 1) % il_period < int(il_period * il_percent):
        action = torch.tensor(env.action_dir[op_seq[t]])
        if state_value > 0: # Must hack
            state_value = torch.tensor([state_value.item() * 2])
        else:
            state_value = torch.tensor([state_value.item() / 2])
    policy.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()

def select_action_val(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs, state_value = policy(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item()

def finish_episode():
    R = 0
    saved_actions = policy.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in policy.rewards[::-1]:
        R = r + env.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    optimizer.zero_grad()
    all_loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    all_loss.backward()
    optimizer.step()

    del policy.rewards[:]
    del policy.saved_actions[:]
    return all_loss

def main():
    
    running_reward = 0
    total_step = 0
    total_reward = []
    loss = 0
    l_total_step = 0
    max_step = 8000000
    log_interval = 500
    il_period = 5000
    il_percent = 0.0
    # 0.0       | 0.01  | 0.05 | 0.1(Base)  | 0.2   | 0.5   | 1.0   | divide    | divide_0.05
    # 0/0(100)  | 47/83 | 45/75| 46/75      | 36/50 | 53    | 0     | 48/77     ｜ 6/71 
    start_time = time.time()
    for i_episode in count(1):
        # 0.01 0.1  | 0.03 0.3  | 0.03 0.15 ｜ 0.03 0.15 8m
        # 54/78     | 59/81     ||
        state, op_seq = env.reset_cd(0.015, 0.075, max_step)
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
            with open('reward/ac_reward_cd_0.03_0.15_8m.json', 'w') as outfile:
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
    axs.plot(x, y, label="ac_average_reward")

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

