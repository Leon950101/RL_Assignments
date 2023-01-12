# cd Documents/New/Courses/RL/Project/code/

import numpy as np
import json
import random

class MyKarel_1:

    def __init__(self):
        self.actions = ["m", "l", "r", "pk", "pt", "f"] # Action space: move, turnLeft, turnRight, pickMarker, putMarker, finish
        self.gamma = 0.99 # For reward discount, not needed for episodic task
        # Reward function
        self.r_minus = -0.1 # Terminations
        self.r_normal = -0.01 # Normal move
        self.r_positive = 1 # Finish the task

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
    
    # Reset environment
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
        return self.s_0 # Set init state to the agent

    # Transition probability function
    def step(self, s, a):
        d, x, y = s[0:3]
        s_1 = [d, x, y] # Next state
        t = True # Termination status
        r = 0 # Reward
        if a == "m" : # Action "move"
            if d == 0: # To west
                if (y-1) < 0 or [x, y-1] in self.walls: # Cross the edge or Hit the wall: termination and return minus reward (-0.1)
                    t = True
                    r = self.r_minus
                else: # Move to the destination grid and return normal reward (0)
                    if [x, y-1] in self.markers:
                        s_1 = (d, x, y-1)
                        t = False
                        r = self.r_normal
                    else:
                        s_1 = (d, x, y-1)
                        t = False
                        r = self.r_normal
            elif d == 1: # To south
                if (x+1) > (self.x - 1) or [x+1, y] in self.walls:
                    t = True
                    r = self.r_minus
                else:
                    if [x+1, y] in self.markers:
                        s_1 = (d, x+1, y)
                        t = False
                        r = self.r_normal
                    else:
                        s_1 = (d, x+1, y)
                        t = False
                        r = self.r_normal
            elif d == 2: # To east
                if (y+1) > (self.y - 1) or [x, y+1] in self.walls:
                    t = True
                    r = self.r_minus
                else:
                    if [x, y+1] in self.markers:
                        s_1 = (d, x, y+1)
                        t = False
                        r = self.r_normal
                    else:
                        s_1 = (d, x, y+1)
                        t = False
                        r = self.r_normal
            elif d == 3: # To north
                if (x-1) < 0 or [x-1, y] in self.walls:
                    t = True
                    r = self.r_minus
                else:
                    if [x-1, y] in self.markers:
                        s_1 = (d, x-1, y)
                        t = False
                        r = self.r_normal
                    else:
                        s_1 = (d, x-1, y)
                        t = False
                        r = self.r_normal
        
        if a == "l" : # Action "turnLeft"
            d = (d+1) % 4
            s_1 = (d, x, y)
            t = False
            r = self.r_normal

        if a == "r" : # Action "turnRight"
            d = (d+3) % 4
            s_1 = (d, x, y)
            t = False
            r = self.r_normal
        
        if a == "pk" : # Action "pickMarker"
            if [x, y] not in self.markers: # no MARKER this grid-cell: termination and return minus reward (-0.1)
                t = True
                r = self.r_minus
            else:
                t = False
                s_1 = (d, x, y) # Pick MARKER and ruturn normal reward
                r = self.r_normal
                self.markers.remove([x, y])
                s[3+x*4+y] = 0

        
        if a == "pt" : # Action "putMarker"
            if [x, y] in self.markers: # has MARKER on grid-cell: termination and return minus reward (-1)
                t = True
                r = self.r_minus
            else:
                t = False
                s_1 = (d, x, y) # Put MARKER and ruturn normal reward
                r = self.r_normal
                self.markers.append([x, y])
                s[3+x*4+y] = 0
                
        if a == "f" : # Action "finish"
            t = True
            # If the agent at target grid with the specific MARKER picked/put: return positive reward (1);
            if s_1[0] == self.s_f[0] and s_1[1] == self.s_f[1] and s_1[2] == self.s_f[2]: r = self.r_positive # self.markers = [] # Set markers
            # Else: return minus reward (-0.1)
            else: r = self.r_minus
        
        s[0], s[1], s[2] = s_1
        return t, r, s

    # Show the grid world with: "-" for empty grid, "o" for MARKER, "#" for WALL, 
    # "w/s/e/n" for the agent's current location with direction "west/south/east/north",
    # "W/S/E/N" for the agent's current location with a MARKER on
    def render(self, s):
        for x in range(self.x):
            print()
            for y in range(self.y):
                if x == s[1] and y == s[2]: # Agent current location
                    if s[0] == 0:
                        if [x, y] in self.markers: print('W ', end="")
                        else: print('w ', end="")
                    elif s[0] == 1:
                        if [x, y] in self.markers: print('S ', end="")
                        else: print('s ', end="")
                    elif s[0] == 2:
                        if [x, y] in self.markers: print('E ', end="")
                        else: print('e ', end="")
                    elif s[0] == 3:
                        if [x, y] in self.markers: print('N', end="")
                        else: print('n ', end="")
                elif [x, y] in self.walls: print('# ', end="")
                elif [x, y] in self.markers: print('o ', end="")
                else: print("- ", end="")
        print()
        print()

    # Run the game
    def run(self):
        termination = True
        while True:
            if termination:
                print("** New Episode **")
                s = self.reset()
                self.render(s)

            print("Enter your action from m/l/r/pk/pt/f or q to quit: ", end="")
            a = input()
            print()
            if a == "q":
                break
            if a in self.actions:
                termination, r, s_1 = self.step(s, a)
                s = s_1
                print("reward:", r)
                self.render(s) 

if __name__ == '__main__':
   
    play = MyKarel_1()
    play.run()

# Env example
# {
#   "gridsz_num_rows": 4,
#   "gridsz_num_cols": 4,
#   "pregrid_agent_row": 2,
#   "pregrid_agent_col": 3,
#   "pregrid_agent_dir": "north",
#   "postgrid_agent_row": 0,
#   "postgrid_agent_col": 1,
#   "postgrid_agent_dir": "west",
#   "walls": [[1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2], [3, 3]],
#   "pregrid_markers": [],
#   "postgrid_markers": [[0, 3]]
# }