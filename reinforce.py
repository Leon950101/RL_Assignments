import argparse
import gym
import numpy as np
import random
import time
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from gridworld import Gridworld
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = Gridworld() # gym.make('CartPole-v1') # 
env.reset()
# torch.manual_seed(args.seed)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.flatten = nn.Flatten()
        self.affine1 = nn.Linear(22, 128)
        # self.dropout1 = nn.Dropout(p=0.1)
        self.relu1 = nn.ReLU()
        self.affine2 = nn.Linear(128, 128)
        # self.dropout2 = nn.Dropout(p=0.1)
        self.relu2 = nn.ReLU()
        self.affine3 = nn.Linear(128, 6)
        
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.flatten(x)
        x = self.affine1(x)
        # x = self.dropout1(x)
        x = self.relu1(x)
        x = self.affine2(x)
        # x = self.dropout2(x)
        x = self.relu2(x)
        action_scores = self.affine3(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-4)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()

def finish_episode():
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.appendleft(R)
    # if len(returns) == 1: returns.append(0)
    returns = torch.tensor(returns)
    # returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def main():
    running_reward = 0
    total_step = 0
    total_reward = []
    max_step = 2000000
    start_time = time.time()
    for i_episode in count(1):
        state = env.reset()
        ep_reward = 0
        for t in range(1, 100):  # Don't infinite loop while learning (500)
            total_step += 1
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tTotal Step: {}\tAverage reward: {:.2f}'.format(
                  i_episode, total_step, running_reward)) # tLast reward: {:.2f}\
            total_reward.append(running_reward)
        if total_step > max_step: # env.spec.reward_threshold:
            # print("Done! Running reward is now {:.2f} and "
            #       "the last episode runs to {} time steps!".format(running_reward, t))
            break
    
    end_time = time.time()
    run_time = round((end_time - start_time)/60)
    print("Run Time: ", end="")
    print(run_time, end="")
    print("min")

    # Plot
    x = np.arange(0, i_episode, args.log_interval, dtype=int)
    y = total_reward
    if len(x) != len(y): y.append(y[-1])
    fig, axs = plt.subplots()
    axs.plot(x, y, label="ep_rew_mean")
    # axs.fill_between(x, mean + 0.5*std, mean - 0.5*std, alpha=0.2)

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()