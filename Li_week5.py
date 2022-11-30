
import sys
import numpy as np
import matplotlib.pyplot as plt
from random import choice
from MDP_T1 import MyKarel_1
from MDP_T2 import MyKarel_2

def Q_learning(env):
    
    # Information from the enviornment
    actions = env.actions # Action space
    states = env.states # State space
    walls = env.walls # WALL setting
    markers = env.markers # MARKER setting
    X = env.x # Coordinate X
    Y = env.y # Coordinate Y
    gamma = env.gamma 
    
    H = 5 / (1 - gamma) # For episode horizon
    episode_steps = 0 # Count episode steps
    episode_num = 0 # Count episode numbers
    alpha = 0.5 # Step size
    epsilon = 0.1 # For e-greedy policy

    # Initialization for the algorithm
    Q_values = _init_Q(walls, markers, actions, X, Y) # Initialize Q(s), 56 * 4 = 224 or 512 * 5 = 2560
    it_num = 0 # To count the iteration number for finding the optimal policy
    it_max_num = 1000000 # Maxmimum steps in total
    print_fre = 1000
    acc_reward = np.zeros(int(it_max_num / print_fre)) # For plot
    ind_r = 0

    # Compute the optimal Q-value function
    t = True
    s = env.reset()
    while it_num < it_max_num:
        it_num += 1
        
        a = e_greedy_policy(s, epsilon, actions, Q_values)
        t, r, s_1 = env.transition(s, a)
        episode_steps += 1
                
        if t is True or episode_steps >= H:
            Q_values[s, a] += alpha * (r - Q_values[s, a])
            s = env.reset()
            episode_num += 1
            episode_steps = 0
            acc_reward[ind_r] += r
        else:
            a_1= max_action(s_1, actions, Q_values) # TODO
            Q_values[s, a] += alpha * (r + gamma * Q_values[s_1, a_1] - Q_values[s, a])
            s = s_1
        
        if it_num % print_fre == 0: 
            print(it_num)
            acc_reward[ind_r] /= episode_num
            episode_num = 0
            ind_r += 1

    # Plot
    x = np.linspace(0, it_max_num, int(it_max_num / print_fre))
    y = acc_reward

    fig, ax = plt.subplots()

    ax.plot(x, y, linewidth=2.0)
    plt.show()

    return Q_values
    
def max_action(s, actions, Q_values):
    a = choice(actions)
    q_s_values = {}
    for a in actions:
        q_s_values[a] = Q_values[s, a]
    a_max = max(q_s_values.items(), key=lambda x:x[1])
    a = a_max[0]
    return a

def e_greedy_policy(s, epsilon, actions, Q_values):
    a = choice(actions)
    if np.random.random() > epsilon: # Choose the maximum
        q_s_values = {}
        for a in actions:
            q_s_values[a] = Q_values[s, a]
        a_max = max(q_s_values.items(), key=lambda x:x[1])
        a = a_max[0]
    return a

def _init_Q(walls, markers, actions, X, Y):
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
def run_optimal(env, epsilon, actions, Q_values):
    print()
    print("** Best Sequence of Commands **")
    
    t = True
    s = env.reset()
    env.show(s)
    
    while True:
        a = e_greedy_policy(s, epsilon, actions, Q_values)
        t, r, s_1 = env.transition(s, a)
        s = s_1
        print("Action:", a, "/ Reward:", r)       
        env.show(s)
        if t:    
            break

if __name__ == '__main__':
    # Choose the environment based on the input parameter
    print()
    if len(sys.argv) < 2: # No environment specified so choose T2 by default
        print ("Task T2")
        env = MyKarel_2()
    else:
        if sys.argv[1] == "T1": 
            print("Task T1")
            env = MyKarel_1()
        elif sys.argv[1] == "T2": 
            print("Task T2")
            env = MyKarel_2()
        else: # Wrong environment specified so choose T2 by default
            print("Task T2")
            env = MyKarel_2()

    # Calculate the optimal policy and run the game with it
    Q_values = Q_learning(env)
    # run_optimal(env, 0.1, env.actions, Q_values)

