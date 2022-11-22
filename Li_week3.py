
import sys
from MDP_T1 import MyKarel_1
from MDP_T2 import MyKarel_2

def value_iteration(env):
    
    # Initialization
    theta = 0.0000001 # a small threshold determining accuracy of estimation
    delta = 1
    gamma = env.gamma
    actions = env.actions
    walls = env.walls
    markers = env.markers
    X = env.x
    Y = env.y
    states, V_values = init_s_V(walls, markers, X, Y) # 14(16 grids-2 walls) * 4 = 56, ignore the terminal state

    # Computing the optimal value function
    while delta > 0.0000001:
        delta = 0
        for s in states:
            v = V_values[s]
            V_temp = []
            for a in actions:
                t, r, s_1 = env.transition(s, a)
                if t is True:
                    V_temp.append(r)
                else:
                    V_temp.append(r + gamma * V_values[s_1])

            V_values[s] = max(V_temp)
            delta = max(delta, abs(v - V_values[s]))

    # Extracting the optimal policy
    policy = {}
    for s in states:
        V_temp = {}
        for a in actions:
            t, r, s_1 = env.transition(s, a)
            if t is True:
                V_temp[a] = r
            else:
                V_temp[a] = r + gamma * V_values[s_1]
        a_1 = max(V_temp.items(), key=lambda x:x[1]) # ??
        policy[s] = a_1[0]

    return policy

def run_optimal(env, policy):
    print("** Best Actions **")
    termination = True
    s = env.reset()
    env.show(s)
    while True:
        a = policy[s]
        termination, r, s_1 = env.transition(s, a)
        s = s_1
        print("reward:", r)
        env.show(s)
        if termination:    
            break

def init_s_V(walls, markers, X, Y):
    states = []
    V_values = {}
    if len(markers) == 0:
        for x in range(X):
            for y in range(Y):
                if [x, y] in walls:
                    pass
                else:
                    for i in range(4):
                        states.append((i, x, y))
                        V_values[i, x, y] = 0
    else: # TODO
        for x in range(X):
            for y in range(Y):
                if [x, y] in walls:
                    pass
                else:
                    for i in range(4):
                        states.append((i, x, y, 0, 0))
                        V_values[i, x, y, 0, 0] = 0
                        states.append((i, x, y, 0, 1))
                        V_values[i, x, y, 0, 1] = 0
                        states.append((i, x, y, 1, 0))
                        V_values[i, x, y, 1, 0] = 0
                        states.append((i, x, y, 1, 1))
                        V_values[i, x, y, 1, 1] = 0

    return states, V_values

if __name__ == '__main__':
    # Choose the environment
    if len(sys.argv) < 2:
        print (2)
        env = MyKarel_2()
    else:
        if sys.argv[1] == "T1": 
            print(1)
            env = MyKarel_1()
        elif sys.argv[1] == "T2": 
            print(2)
            env = MyKarel_2()
        else: 
            print(2)
            env = MyKarel_2()

    # Calculate the optimal policy
    policy = value_iteration(env)
    actions_sequence = run_optimal(env, policy)
    print(actions_sequence)

