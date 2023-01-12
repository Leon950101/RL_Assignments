import gym
import numpy as np
from gym import spaces
import random
import json
# from stable_baselines3.common.env_checker import check_env

class Gridworld(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self): # arg1, arg2
        super().__init__()
        
        self.actions = ["m", "l", "r", "pk", "pt", "f"] # Action space: move, turnLeft, turnRight, pickMarker, putMarker, finish
        self.gamma = 0.99 # For reward discount, not needed for episodic task
        # Reward function
        self.rewards = {"minus": -0.1, "normal": -0.01, "positive": 1}
        self.state = []

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        n_actions = len(self.actions)
        self.action_space = spaces.Discrete(n_actions)
        # Example for using image as input (channel-first; channel-last also works):
        # self.observation_space = spaces.Box(low=0, high=255,
                                            # shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8) # TODO
    
    def _init_state_map(self, agent_position, w, m):
         # State: d: 0/1/2/3: west/south/east/north | x/y: 0-3 | state_map: 0/1/2 empty/wall/marker 
        state_map = np.zeros((4, 4), dtype=int)
        for x in range(4):
            for y in range(4):
                if [x, y] in w: state_map[x][y] = 1
                elif [x, y] in m: state_map[x][y] = 2
                else: state_map[x][y] = 0
        flatten = state_map.flatten()
        state = np.concatenate((agent_position, flatten))
        return state

    def reset(self):
        env_id = random.randint(0, 0) # 23999
        with open('data/train/task/'+str(env_id)+'_task.json', 'r') as fcc_file:
            fcc_data = json.load(fcc_file)
        self.walls = fcc_data["walls"] # Set walls
        self.preMarkers = fcc_data["pregrid_markers"]
        self.postMarkers = fcc_data["postgrid_markers"]
        self.markers = fcc_data["pregrid_markers"] # Set markers
        self.x, self.y = fcc_data["gridsz_num_rows"], fcc_data["gridsz_num_cols"]
        dir = {"west":0, "south":1, "east":2, "north":3}
        d_0, x_0, y_0 = dir[fcc_data["pregrid_agent_dir"]], fcc_data["pregrid_agent_row"], fcc_data["pregrid_agent_col"]
        d_f, x_f, y_f = dir[fcc_data["postgrid_agent_dir"]], fcc_data["postgrid_agent_row"], fcc_data["postgrid_agent_col"]
        self.s_0 = self._init_state_map([d_0, x_0, y_0], self.walls, self.preMarkers) # Init state
        self.s_f = self._init_state_map([d_f, x_f, y_f], self.walls, self.postMarkers) # Target state
        observation = self.s_0 # Set init state to the agent
        self.state = self.s_0
        return observation

    def step(self, a):
        s = self.state
        d, x, y = s[0:3]
        s_1 = [d, x, y] # Next state
        t = True # Termination status
        r = 0 # Reward
        if a == "m" : # Action "move"
            if d == 0: # To west
                if (y-1) < 0 or [x, y-1] in self.walls: # Cross the edge or Hit the wall: termination and return minus reward (-0.1)
                    t = True
                    r = self.rewards["minus"]
                else: # Move to the destination grid and return normal reward (0)
                    if [x, y-1] in self.markers:
                        s_1 = (d, x, y-1)
                        t = False
                        r = self.rewards["normal"]
                    else:
                        s_1 = (d, x, y-1)
                        t = False
                        r = self.rewards["normal"]
            elif d == 1: # To south
                if (x+1) > (self.x - 1) or [x+1, y] in self.walls:
                    t = True
                    r = self.rewards["minus"]
                else:
                    if [x+1, y] in self.markers:
                        s_1 = (d, x+1, y)
                        t = False
                        r = self.rewards["normal"]
                    else:
                        s_1 = (d, x+1, y)
                        t = False
                        r = self.rewards["normal"]
            elif d == 2: # To east
                if (y+1) > (self.y - 1) or [x, y+1] in self.walls:
                    t = True
                    r = self.rewards["minus"]
                else:
                    if [x, y+1] in self.markers:
                        s_1 = (d, x, y+1)
                        t = False
                        r = self.rewards["normal"]
                    else:
                        s_1 = (d, x, y+1)
                        t = False
                        r = self.rewards["normal"]
            elif d == 3: # To north
                if (x-1) < 0 or [x-1, y] in self.walls:
                    t = True
                    r = self.rewards["minus"]
                else:
                    if [x-1, y] in self.markers:
                        s_1 = (d, x-1, y)
                        t = False
                        r = self.rewards["normal"]
                    else:
                        s_1 = (d, x-1, y)
                        t = False
                        r = self.rewards["normal"]
        
        if a == "l" : # Action "turnLeft"
            d = (d+1) % 4
            s_1 = (d, x, y)
            t = False
            r = self.rewards["normal"]

        if a == "r" : # Action "turnRight"
            d = (d+3) % 4
            s_1 = (d, x, y)
            t = False
            r = self.rewards["normal"]
        
        if a == "pk" : # Action "pickMarker"
            if [x, y] not in self.markers: # no MARKER this grid-cell: termination and return minus reward (-0.1)
                t = True
                r = self.rewards["minus"]
            else:
                t = False
                s_1 = (d, x, y) # Pick MARKER and ruturn normal reward
                r = self.rewards["normal"]
                self.markers.remove([x, y])
                s[3+x*4+y] = 0

        
        if a == "pt" : # Action "putMarker"
            if [x, y] in self.markers: # has MARKER on grid-cell: termination and return minus reward (-1)
                t = True
                r = self.rewards["minus"]
            else:
                t = False
                s_1 = (d, x, y) # Put MARKER and ruturn normal reward
                r = self.rewards["normal"]
                self.markers.append([x, y])
                s[3+x*4+y] = 0
                
        if a == "f" : # Action "finish"
            t = True
            # If the agent at target grid with the specific MARKER picked/put: return positive reward (1);
            if s_1[0] == self.s_f[0] and s_1[1] == self.s_f[1] and s_1[2] == self.s_f[2]: r = self.rewards["positive"] # self.markers = [] # Set markers
            # Else: return minus reward (-0.1)
            else: r = self.rewards["minus"]
        
        s[0], s[1], s[2] = s_1
        done = t
        observation = s
        reward = r
        info = {}
        self.state = s
        return observation, reward, done, info

    def render(self, mode="human"):
        for x in range(self.x):
            print()
            for y in range(self.y):
                if x == self.state[1] and y == self.state[2]: # Agent current location
                    if self.state[0] == 0:
                        if [x, y] in self.markers: print('W ', end="")
                        else: print('w ', end="")
                    elif self.state[0] == 1:
                        if [x, y] in self.markers: print('S ', end="")
                        else: print('s ', end="")
                    elif self.state[0] == 2:
                        if [x, y] in self.markers: print('E ', end="")
                        else: print('e ', end="")
                    elif self.state[0] == 3:
                        if [x, y] in self.markers: print('N', end="")
                        else: print('n ', end="")
                elif [x, y] in self.walls: print('# ', end="")
                elif [x, y] in self.markers: print('o ', end="")
                else: print("- ", end="")
        print()
        print()

    def close(self):
        pass

    # Run the game
    def run(self):
        termination = True
        while True:
            if termination:
                print("** New Episode **")
                s = self.reset()
                self.render()

            print("Enter your action from m/l/r/pk/pt/f or q to quit: ", end="")
            a = input()
            print()
            if a == "q":
                break
            if a in self.actions:
                s_1, r, termination, _ = self.step(a)
                s = s_1
                print("reward:", r)
                self.render(s) 

if __name__ == '__main__':
   
    env = Gridworld()
    # check_env(env)
    env.run()