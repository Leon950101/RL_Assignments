
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from random import choice
from MDP import MyKarel_1

# Choose one set of parameters and run multiple times
def REINFORCE(env, n_e, pf):
    
    # Information from the enviornment
    actions = env.actions # Action space
    walls = env.walls # WALL setting
    markers = env.markers # MARKER setting
    X = env.x # Coordinate X
    Y = env.y # Coordinate Y
    gamma = env.gamma # Discount factor

    # Input and problem setup | Initialize policy parameters and compute initial policy
    theta = {} # Initialize policy parameter vector theta(s, a)
    policy = {} # Computer policy using soft-max

    # Hyperparameters    
    alpha = 0.5 # Step size
    N = n_e # Number of episodes
    H = 1 / (1 - gamma) # Maximum length of an episode: 5 / (1 - gamma)

    # Other parameters
    episode_counter = 0 # To count the number of total episodes
    print_fre = pf # Print frequence
    acc_reward = np.zeros(int(N / print_fre)) # For plot, average total rewards per episode along the time
    ind_r = 0 # Index for reward ploting

    # REINFORCE
    while episode_counter < N:
        
        episode = [] # Store the generated sequence of an episode
        episode_steps = 0 # Count episode steps

        s = env.reset()
        t = False # Termination status
        while t is not True:
            a = get_action(policy, theta, s, actions) # Get action from the current policy
            t, r, s_1 = env.step(s, a) # Get termination state, reward and next state from env
            episode_steps += 1
            if episode_steps > H:
                break
            acc_reward[ind_r] += r
            episode.append([s, a, r])
            s = s_1
        episode_counter += 1
       
        G = 0
        for i in range(len(episode) - 1, -1, -1): # From T-1 to 0
            G = gamma * G + episode[i][2]
            # Update theta
            s = episode[i][0]
            for a in actions:
                s_a = ''.join(np.append(s,a))
                if s_a not in theta:
                    theta[s_a] = 0
                if a == episode[i][1]:
                    theta[s_a] = theta[s_a] + alpha * gamma**i * G * (1 - policy[s_a])
                elif a != episode[i][1]:
                    theta[s_a] = theta[s_a] + alpha * gamma**i * G * (- policy[s_a])
                else:
                    pass
            # Compute policy
            policy = _compute_policy(policy, actions, s, theta)
        
        # For print and reward scaling 
        if episode_counter % print_fre == 0:
            acc_reward[ind_r] /= print_fre
            print('Total episode number: ', episode_counter, len(policy), acc_reward[ind_r])
            ind_r += 1
            
    return policy, theta, acc_reward

def _compute_policy(policy, actions, s, theta):
    sum = 0
    for a in actions:
        s_a = ''.join(np.append(s,a))
        if s_a not in theta:
            theta[s_a] = 0
        sum += math.exp(theta[s_a])
    for a in actions:
        s_a = ''.join(np.append(s,a))
        policy[s_a] = math.exp(theta[s_a]) / sum
    sum = 0
    return policy

def get_action(policy, theta, s, actions):
    next_action_prob = []
    next_actions = actions
    for a in actions:
        s_a = ''.join(np.append(s,a))
        if s_a not in policy:
            policy = _compute_policy(policy, actions, s, theta)
        next_action_prob.append(policy[s_a])
    # Pick the action based on probability
    action = np.random.choice(a=next_actions, size=1, replace=True, p=next_action_prob)
    return action

# Run the game with optimal policy
def run_optimal(env, actions, policy, theta):
    print()
    print("** Best Sequence of Commands **")
    
    t = True
    s = env.reset()
    env.render(s)
    
    while True:
        a = get_action(policy, theta, s, actions)
        t, r, s_1 = env.step(s, a)
        s = s_1
        print("Action:", a, "/ Reward:", r)       
        env.render(s)
        if t:    
            break
    return r

if __name__ == '__main__':
    # Choose the environment based on the input parameter
    print()
    env = MyKarel_1()
    env_name = "T1"

    number_of_episodes = 10000 # Maxmimum episodes in total
    print_fre = 100 # Print frequence
    acc_reward = []

    # Try to get the optimal policy
    policy, theta, a_r = REINFORCE(env, number_of_episodes,  print_fre)

    # Multiple running for ploting
    x = np.linspace(0, number_of_episodes, int(number_of_episodes / print_fre))
    acc_reward.append(a_r)
    acc_reward = np.array(acc_reward)
    mean = np.mean(acc_reward, axis=0)
    std = np.std(acc_reward, axis=0)
        
    # Plot
    fig, axs = plt.subplots()
    axs.plot(x, mean, label=env_name)
    axs.fill_between(x, mean + 0.5*std, mean - 0.5*std, alpha=0.2)

    plt.legend()
    plt.show()

    # Generate the best sequence
    # r = run_optimal(env, env.actions, policy, theta)

# def _init_theta(walls, markers, actions, X, Y):
#     theta = {}
#     for x in range(X):
#         for y in range(Y):
#             if [x, y] in walls:
#                 pass
#             else:
#                 for i in range(4):
#                     for a in actions:
#                         theta[(i, x, y), a] = 0
#                         theta[(i, x, y), a] = 0 
#     return theta

# def _compute_policy(states, actions, theta):
#     policy = {}
#     sum = 0
#     for s in states:
#         for a in actions:
#             sum += math.exp(theta[s,a])
#         for a in actions:
#             policy[s,a] = math.exp(theta[s,a]) / sum
#         sum = 0
#     return policy