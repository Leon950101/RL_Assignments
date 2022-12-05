
import sys
import numpy as np
import matplotlib.pyplot as plt
from random import choice
from MDP_T1 import MyKarel_1
from MDP_T2 import MyKarel_2

# Choose one set of parameters and run multiple times
def Q_learning(env, imn, pf):
    
    # Information from the enviornment
    actions = env.actions # Action space
    states = env.states # State space
    walls = env.walls # WALL setting
    markers = env.markers # MARKER setting
    X = env.x # Coordinate X
    Y = env.y # Coordinate Y
    gamma = env.gamma # Discount factor
    
    H = 5 / (1 - gamma) # For episode horizon
    episode_steps = 0 # Count episode steps
    alpha = 0.5 # Step size
    epsilon = 0.1 # For e-greedy policy

    # Initialization for the algorithm
    Q_values = _init_Q(walls, markers, actions, X, Y) # Initialize Q(s), 56 * 4 = 224 or 512 * 5 = 2560
    it_num = 0 # To count the iteration number for finding the optimal policy
    it_max_num = imn # Maxmimum steps in total
    print_fre = pf # Print frequence
    acc_reward = np.zeros(int(it_max_num / print_fre)) # For plot, average total rewards per episode along the time
    episode_num = 0 # Count episode numbers in order to cacluate average total rewards
    ind_r = 0 # Index for reward ploting
    episode_counter = 0 # To count the number of total episodes

    # Compute the optimal Q-value function
    t = True # Termination
    s = env.reset()
    while it_num < it_max_num:
        it_num += 1
        
        a = e_greedy_policy(s, epsilon, actions, Q_values) # Get action from current policy
        t, r, s_1 = env.transition(s, a) # Get termination state, reward and next state from env
        episode_steps += 1
        acc_reward[ind_r] += r
        
        if t is True or episode_steps >= H: # Check termination condition
            Q_values[s, a] += alpha * (r - Q_values[s, a])
            s = env.reset()
            episode_num += 1
            episode_steps = 0          
        else:
            a_1= e_greedy_policy(s_1, 0, actions, Q_values) # Set epsilon = 0 to choose maximum action, off-policy
            Q_values[s, a] += alpha * (r + gamma * Q_values[s_1, a_1] - Q_values[s, a])
            s = s_1
        
        if it_num % print_fre == 0: # For print and reward scaling 
            acc_reward[ind_r] /= episode_num
            episode_counter += episode_num
            episode_num = 0
            ind_r += 1
            print('Steps: ', it_num, '| Total episode number: ', episode_counter, 
                  '| Average episode length: ', round(it_num / episode_counter, 2))

    return Q_values, acc_reward
    
def e_greedy_policy(s, epsilon, actions, Q_values): # e-greedy policy
    a = choice(actions)
    if np.random.random() >= epsilon: # Choose the maximum
        q_s_values = {}
        for a in actions:
            q_s_values[a] = Q_values[s, a]
        a_max = max(q_s_values.items(), key=lambda x:x[1])
        a = a_max[0]
    return a

def _init_Q(walls, markers, actions, X, Y): # TODO set initial state value as 1
    Q_values = {}
    if len(markers) == 0: # No MARKERs
        for x in range(X):
            for y in range(Y):
                if [x, y] in walls:
                    pass
                else:
                    for i in range(4): # 4 directions
                        for a in actions:
                            Q_values[(i, x, y), a] = 0
    else: # Simplified version, work for Task T1
        for x in range(X):
            for y in range(Y):
                if [x, y] in walls:
                    pass
                else:
                    for i in range(4):
                        for a in actions:
                            Q_values[(i, x, y, 0, 0), a] = 0
                            Q_values[(i, x, y, 0, 1), a] = 0
                            Q_values[(i, x, y, 1, 0), a] = 0 
                            Q_values[(i, x, y, 1, 1), a] = 0
    return Q_values

# Run the game with optimal policy
def run_optimal(env, actions, Q_values):
    print()
    print("** Best Sequence of Commands **")
    
    t = True
    s = env.reset()
    env.show(s)
    
    while True:
        a = e_greedy_policy(s, 0, actions, Q_values)
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

    it_max_num = 1000000 # Maxmimum steps in total
    print_fre = 10000 # Print frequence

    # Try to get the optimal policy and run the game with it
    Q_values, _ = Q_learning(env, it_max_num,  print_fre)

    # Multiple running for ploting
    test_reward = []
    x = np.linspace(0, it_max_num, int(it_max_num / print_fre))
    acc_reward = []
    for i in range(10):
        _, a_r = Q_learning(env, it_max_num, print_fre)
        acc_reward.append(a_r)
    acc_reward = np.array(acc_reward)
    mean = np.mean(acc_reward, axis=0)
    std = np.std(acc_reward, axis=0)
        
    # Plot
    plt.plot(x, mean, label=env_name)
    plt.fill_between(x, mean + std, mean - std, alpha=0.2)

    plt.legend()
    plt.show()

    # Generate the best sequence
    r = run_optimal(env, env.actions, Q_values)

    # fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # axs[0].plot(x, mean, label=env_name)
    # axs[0].fill_between(x, mean + std, mean - std, alpha=0.2)
    # axs[0].plot(x, test_reward, label=env_name)
    # axs[1].plot(x, test_reward, label=env_name)