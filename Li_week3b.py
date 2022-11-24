
import sys
from MDP_T1 import MyKarel_1
from MDP_T2 import MyKarel_2

def value_iteration(env):
    
    # Information from the enviornment
    actions = env.actions # Action space
    walls = env.walls # WALL setting
    markers = env.markers # MARKER setting
    X = env.x # Coordinate X
    Y = env.y # Coordinate Y

    # Initialization for the algorithm
    theta = 0.00001 # A small threshold determining accuracy of estimation
    delta = 1 # Error initialization
    states, V_values = init_s_V(walls, markers, X, Y) # Generate state space (14(16 grids-2 walls) * 4) and Initialize V(s)
    gamma = 0.99 # For reward discount, not needed for episodic task
    it_num = 0 # To count the iteration number for finding the optimal policy

    # Compute the optimal value function
    while delta > theta:
        it_num += 1
        delta = 0
        for s in states:
            v = V_values[s]
            V_temp = [] # Store each estimated return based on taking each action
            for a in actions:
                t, r, s_1 = env.transition(s, a)
                if t is True:
                    V_temp.append(r)
                else:
                    V_temp.append(r + gamma * V_values[s_1])
            V_values[s] = max(V_temp) # Assign the maximum return to current state
            delta = max(delta, abs(v - V_values[s]))

    print()
    print("Itration number to find the optimal policy:", it_num)
    
    # Output a deterministic policy
    policy = {}
    for s in states:
        V_temp = {}
        for a in actions:
            t, r, s_1 = env.transition(s, a)
            if t is True:
                V_temp[a] = r
            else:
                V_temp[a] = r + gamma * V_values[s_1]
        a_max = max(V_temp.items(), key=lambda x:x[1])
        policy[s] = a_max[0]

    return policy

def init_s_V(walls, markers, X, Y):
    states = []
    V_values = {}
    if len(markers) == 0: # No MARKERs
        for x in range(X):
            for y in range(Y):
                if [x, y] in walls:
                    pass
                else:
                    for i in range(4): # 4 directions
                        states.append((i, x, y))
                        V_values[i, x, y] = 0
    else: # Simplified version, work for Task T1
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

# Run the game with optimal policy
def run_optimal(env, policy):
    print()
    print("** Best Sequence of Commands **")
    
    termination = True
    s = env.reset()
    env.show(s)
    
    while True:
        a = policy[s]
        termination, r, s_1 = env.transition(s, a)
        s = s_1
        print("Action:", a, "/ Reward:", r)       
        env.show(s)
        if termination:    
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
    policy = value_iteration(env)
    run_optimal(env, policy)

