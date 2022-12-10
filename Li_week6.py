
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from random import choice
from MDP_T1 import MyKarel_1
from MDP_T2 import MyKarel_2

# Choose one set of parameters and run multiple times
def REINFORCE(env, n_e, pf):
    
    # Information from the enviornment
    actions = env.actions # Action space
    states = env.states # State space
    walls = env.walls # WALL setting
    markers = env.markers # MARKER setting
    X = env.x # Coordinate X
    Y = env.y # Coordinate Y
    gamma = env.gamma # Discount factor

    # Input and problem setup | Initialize policy parameters and compute initial policy
    theta = _init_theta(walls, markers, actions, X, Y) # Initialize policy parameter vector theta(s, a)
    feature_vector = _init_feature_vector(walls, markers, actions, X, Y) # TODO Initialize feature vectors with one-hot encodings
    policy = _compute_policy(states, actions, theta)

    # Hyperparameters    
    alpha = 0.5 # Step size
    N = n_e # Number of episodes
    H = 5 / (1 - gamma) # Maximum length of an episode

    # Other parameters    
   
    print_fre = pf # Print frequence
    acc_reward = np.zeros(int(N / print_fre)) # For plot, average total rewards per episode along the time
    episode_num = 0 # Count episode numbers in order to cacluate average total rewards
    ind_r = 0 # Index for reward ploting
    episode_counter = 0 # To count the number of total episodes
    av_ep_length_log = [] # For plot 2

    # REINFORCE
    while episode_num < N:
        episode = []
        episode_num += 1
        episode_steps = 0 # Count episode steps

        s = env.reset()
        t = True # termination
        while t is True:
            a = get_action(policy, s, actions)
            episode_steps += 1
            t, r, s_1 = env.transition(s, a) # Get termination state, reward and next state from env
            episode_steps += 1
            if episode_steps > H:
                break
            acc_reward[ind_r] += r
            episode.append([s, a, r])
            s = s_1
        
        episode = episode.reverse()
        G = 0
        for i in range(len(episode)):
            G = gamma * G + episode[i][2]
            for s in states:
                for a in actions:
                    if s == episode[i][0] and a == episode[i][1]:
                        theta[s,a] = theta[s,a] + alpha * gamma**i * G (1 - policy[s,a])
                    elif s == episode[i][0] and a != episode[i][1]:
                        theta[s,a] = theta[s,a] + alpha * gamma**i * G (-policy[s,a])
                    else:
                        pass
            policy = _compute_policy(states, actions, theta)
        
        # For print and reward scaling 
        if episode_num % print_fre == 0:
            acc_reward[ind_r] /= episode_num
            episode_counter += episode_num
            episode_num = 0
            ind_r += 1
            av_ep_length = episode_num / episode_counter
            av_ep_length_log.append(av_ep_length)
            print('Steps: ', episode_num, '| Total episode number: ', episode_counter)
            # Last 10% steps for test
            if episode_num == 0.9 * N: epsilon = 0

    return policy, acc_reward, av_ep_length_log
    
def _init_theta(walls, markers, actions, X, Y):
    theta = {}
    if len(markers) == 0: # No MARKERs
        for x in range(X):
            for y in range(Y):
                if [x, y] in walls:
                    pass
                else:
                    for i in range(4): # 4 directions
                        for a in actions:
                            theta[(i, x, y), a] = 0
    else: # Simplified version, work for Task T1
        for x in range(X):
            for y in range(Y):
                if [x, y] in walls:
                    pass
                else:
                    for i in range(4):
                        for a in actions:
                            theta[(i, x, y, 0, 0), a] = 0
                            theta[(i, x, y, 0, 1), a] = 0
                            theta[(i, x, y, 1, 0), a] = 0 
                            theta[(i, x, y, 1, 1), a] = 0
    return theta

def _init_feature_vector(walls, markers, actions, X, Y):
    feature_vector = 0
    return feature_vector

def _compute_policy(states, actions, theta):
    policy = {}
    sum = 0
    for s in states:
        for a in actions:
            sum += math.exp(theta[s,a])
        for a in actions:
            policy[s,a] = math.exp(theta[s,a]) / sum
        sum = 0
    return policy

def get_action(policy, s, actions):
    next_action_can = {}
    for a in actions:
        next_action_can[a] = policy[s, a]
    next_action = max(next_action_can.items(), key=lambda x:x[1])
    action = next_action[0]
    return action

# Run the game with optimal policy
def run_optimal(env, actions, policy):
    print()
    print("** Best Sequence of Commands **")
    
    t = True
    s = env.reset()
    env.show(s)
    
    while True:
        a = get_action(policy, s, actions)
        t, r, s_1 = env.transition(s, a)
        s = s_1
        print("Action:", a, "/ Reward:", r)       
        env.show(s)
        if t:    
            break
    return r

if __name__ == '__main__':
    # Choose the environment based on the input parameter
    print()
    if len(sys.argv) < 2: # No environment specified so choose T2 by default
        print ("Task T2")
        env = MyKarel_2()
        env_name = "T2"
    else:
        if sys.argv[1] == "T1": 
            print("Task T1")
            env = MyKarel_1()
            env_name = "T1"
        elif sys.argv[1] == "T2": 
            print("Task T2")
            env = MyKarel_2()
            env_name = "T2"
        else: # Wrong environment specified so choose T2 by default
            print("Task T2")
            env = MyKarel_2()
            env_name = "T2"

    number_of_episodes = 1000000 # Maxmimum steps in total
    print_fre = 10000 # Print frequence

    # Try to get the optimal policy
    Q_values, _, _= REINFORCE(env, number_of_episodes,  print_fre)

    # Multiple running for ploting
    x = np.linspace(0, number_of_episodes, int(number_of_episodes / print_fre))
    acc_reward = []
    acc_length = []
    for i in range(10):
        _, a_r, a_e_l = REINFORCE(env, number_of_episodes, print_fre)
        acc_reward.append(a_r)
        acc_length.append(a_e_l)
    acc_reward = np.array(acc_reward)
    mean = np.mean(acc_reward, axis=0)
    std = np.std(acc_reward, axis=0)
    acc_length = np.array(acc_length)
    mean_2 = np.mean(acc_length, axis=0)
    std_2 = np.std(acc_length, axis=0)
        
    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(x, mean)
    axs[0].fill_between(x, mean + std, mean - std, alpha=0.2)
    axs[1].plot(x, mean_2, label='Average Episode Length')
    axs[1].fill_between(x, mean_2 + std_2, mean_2 - std_2, alpha=0.2)

    plt.legend()
    plt.show()

    # Generate the best sequence
    r = run_optimal(env, env.actions, Q_values)

