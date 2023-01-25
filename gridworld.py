# Env Part
import gym
import numpy as np
from gym import spaces
import random
import json
from stable_baselines3.common.env_checker import check_env

class Gridworld(gym.Env):
    """Custom Environment that follows gym interface."""
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

    metadata = {"render.modes": ["human"]}

    # Define constants for clearer code
    MOVE = 0
    TURNLEFT = 1
    TURNRIGHT = 2
    PICKMARKER = 3
    PUTMARKER = 4
    FINISH = 5

    def __init__(self): # arg1, arg2
        super().__init__()
        
        # Action space: move = 0, turnLeft = 1, turnRight = 2, pickMarker = 3, putMarker = 4, finish = 5
        self.actions = [0, 1, 2, 3, 4, 5]
        self.action_dir = {"move":0, "turnLeft":1, "turnRight":2, "pickMarker":3,
                           "putMarker":4, "finish":5}
        self.gamma = 0.99 # For reward discount
        # Reward function
        self.rewards = {"minus": -0.1, "normal": 0, "medium": 0.1, "positive": 1}
        self.env_index = -1 # Enviornment index
        self.total_step = 0

        # self.walls = []
        # self.markers = []
        # self.preMarkers = []
        # self.postMarkers = []
        # self.x, self.y = 0, 0
        # self.s_0 = []
        # self.s_f = []
        # self.state = []

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        n_actions = len(self.actions)
        self.action_space = spaces.Discrete(n_actions) # For PPO A2C
        # self.action_space = spaces.Box(low=0, high=0.999, shape=(1, ), dtype=np.float32) # For DDPG

        # Example for using image as input (channel-first; channel-last also works):
        # observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
        # maybe float with noise (random.rand()/10.0)make the model more robust
        self.observation_space = spaces.Box(low=0, high=4, shape=(22, ), dtype=np.float32) # 38
    
    def _get_state_map(self, agent_position, s_f, w, m, post_m):
        # part of 32
        # Agent: agent_position 1/2/3/4 - w/s/e/n other 0
        # agent_map = np.zeros((4, 4), dtype=int)
        # for x in range(4):
        #     for y in range(4):
        #         if x == agent_position[1] and y == agent_position[2]:
        #             agent_map[x][y] = agent_position[0] + 1
        # flt_agent = agent_map.flatten()
        # State: d: 0/1/2/3: west/south/east/north | x/y: 0-3
        # state_map = np.zeros((4, 4), dtype=int)
        
        # 38: 44 min / 4 million
        # state_map = np.zeros((4, 4), dtype=int)
        # for x in range(4):
        #     for y in range(4):
        #         if [x, y] in w: state_map[x][y] = 1
        #         elif [x, y] in m: state_map[x][y] = 2
        #         else: state_map[x][y] = 0
        # flt_state = state_map.flatten()

        # s_f_map = np.zeros((4, 4), dtype=int)
        # for x in range(4):
        #     for y in range(4):
        #         if [x, y] in w: s_f_map[x][y] = 1
        #         elif [x, y] in post_m: s_f_map[x][y] = 2
        #         else: s_f_map[x][y] = 0
        # flt_s_f = s_f_map.flatten()
 
        # state = np.concatenate((agent_position, flt_state, s_f, flt_s_f))
        
        # 22: 39 min / 4 million
        s_f_map = np.zeros((4, 4), dtype=int)
        for x in range(4):
            for y in range(4):
                if [x, y] in w: s_f_map[x][y] = 1
                elif [x, y] in m and [x, y] in post_m: s_f_map[x][y] = 2
                elif [x, y] in m and [x, y] not in post_m: s_f_map[x][y] = 3
                elif [x, y] not in m and [x, y] in post_m: s_f_map[x][y] = 4
                else: s_f_map[x][y] = 0
        flt_s_f = s_f_map.flatten()

        state = np.concatenate((agent_position, flt_s_f, s_f))
        return state

    def _get_agent_position(self):
        # part of 32
        # State: d: 0/1/2/3: west/south/east/north | x/y: 0-3
        # for i in range(16):
        #     if state[i] > 0:
        #         x = i // 4
        #         y = i % 4
        #         d = state[i] - 1
        # agent_position = [d, x, y]
        return self.state[0:3]

    def reset(self):

        # Sequencially
        # if self.env_index < 23999:
        #     self.env_index += 1
        # else:
        #     self.env_index = 0

        # Randomly
        self.env_index = random.randint(0, 11999) # 3999, 11999, 23999

        with open('data_medium/train/task/'+str(self.env_index)+'_task.json', 'r') as fcc_file_task:
            fcc_data_task = json.load(fcc_file_task)
        # with open('data/train/seq/'+str(self.env_index)+'_seq.json', 'r') as fcc_file_seq:
        #     fcc_data_seq = json.load(fcc_file_seq)
        # self.op_se = fcc_data_seq["sequence"]

        # Curiculumm Design
        # if self.total_step < 500000:
        #     # Randomly
        #     self.env_index = random.randint(0, 3999) # 3999, 11999, 23999
        #     with open('data_easy/train/task/'+str(self.env_index)+'_task.json', 'r') as fcc_file_task:
        #         fcc_data_task = json.load(fcc_file_task)
        # elif self.total_step < 2000000:
        #     # Randomly
        #     self.env_index = random.randint(0, 11999) # 3999, 11999, 23999
        #     with open('data_medium/train/task/'+str(self.env_index)+'_task.json', 'r') as fcc_file_task:
        #         fcc_data_task = json.load(fcc_file_task)
        # else:
        #     # Randomly
        #     self.env_index = random.randint(0, 23999) # 3999, 11999, 23999
        #     with open('data/train/task/'+str(self.env_index)+'_task.json', 'r') as fcc_file_task:
        #         fcc_data_task = json.load(fcc_file_task)
        
        self.x, self.y = fcc_data_task["gridsz_num_rows"], fcc_data_task["gridsz_num_cols"]
        self.walls = fcc_data_task["walls"] # Set walls
        self.preMarkers = fcc_data_task["pregrid_markers"]
        self.postMarkers = fcc_data_task["postgrid_markers"]
        self.markers = self.preMarkers
        dir = {"west":0, "south":1, "east":2, "north":3}
        d_0, x_0, y_0 = dir[fcc_data_task["pregrid_agent_dir"]], fcc_data_task["pregrid_agent_row"], fcc_data_task["pregrid_agent_col"]
        d_f, x_f, y_f = dir[fcc_data_task["postgrid_agent_dir"]], fcc_data_task["postgrid_agent_row"], fcc_data_task["postgrid_agent_col"]
        self.s_0 = self._get_state_map([d_0, x_0, y_0], [d_f, x_f, y_f], self.walls, self.preMarkers, self.postMarkers) # Init state
        self.s_f = self._get_state_map([d_f, x_f, y_f], [d_f, x_f, y_f], self.walls, self.postMarkers, self.postMarkers) # Target state
        self.state = self.s_0 # Set init state to the agent

        self.if_eva = False # For distinguish Train and Test 
        self.ep_step = 0 # Episode step counter

        return self.state

    # For test
    def reset_val(self, map_index, label):
        
        self.env_index = map_index
        with open('data/'+label+'/task/'+str(map_index)+'_task.json', 'r') as fcc_file_task: # TODO, read once to speed up
            fcc_data_task = json.load(fcc_file_task)
        
        self.x, self.y = fcc_data_task["gridsz_num_rows"], fcc_data_task["gridsz_num_cols"]
        self.walls = fcc_data_task["walls"] # Set walls
        self.preMarkers = fcc_data_task["pregrid_markers"]
        self.postMarkers = fcc_data_task["postgrid_markers"]
        self.markers = self.preMarkers
        dir = {"west":0, "south":1, "east":2, "north":3}
        d_0, x_0, y_0 = dir[fcc_data_task["pregrid_agent_dir"]], fcc_data_task["pregrid_agent_row"], fcc_data_task["pregrid_agent_col"]
        d_f, x_f, y_f = dir[fcc_data_task["postgrid_agent_dir"]], fcc_data_task["postgrid_agent_row"], fcc_data_task["postgrid_agent_col"]
        self.s_0 = self._get_state_map([d_0, x_0, y_0], [d_f, x_f, y_f], self.walls, self.preMarkers, self.postMarkers) # Init state
        self.s_f = self._get_state_map([d_f, x_f, y_f], [d_f, x_f, y_f], self.walls, self.postMarkers, self.postMarkers) # Target state
        self.state = self.s_0 # Set init state to the agent

        self.if_eva = True # For distinguish Train and Test 
        self.ep_step = 0 # Episode step counter

        return self.state
    
    # For I.5
    # def reset_file(self, file):
        
    #     with open(file, 'r') as fcc_file_task: # TODO, read once to speed up
    #         fcc_data_task = json.load(fcc_file_task)
        
    #     self.x, self.y = fcc_data_task["gridsz_num_rows"], fcc_data_task["gridsz_num_cols"]
    #     self.walls = fcc_data_task["walls"] # Set walls
    #     self.preMarkers = fcc_data_task["pregrid_markers"]
    #     self.postMarkers = fcc_data_task["postgrid_markers"]
    #     self.markers = self.preMarkers
    #     dir = {"west":0, "south":1, "east":2, "north":3}
    #     d_0, x_0, y_0 = dir[fcc_data_task["pregrid_agent_dir"]], fcc_data_task["pregrid_agent_row"], fcc_data_task["pregrid_agent_col"]
    #     d_f, x_f, y_f = dir[fcc_data_task["postgrid_agent_dir"]], fcc_data_task["postgrid_agent_row"], fcc_data_task["postgrid_agent_col"]
    #     self.s_0 = self._get_state_map([d_0, x_0, y_0], [d_f, x_f, y_f], self.walls, self.preMarkers, self.postMarkers) # Init state
    #     self.s_f = self._get_state_map([d_f, x_f, y_f], [d_f, x_f, y_f], self.walls, self.postMarkers, self.postMarkers) # Target state
    #     self.state = self.s_0 # Set init state to the agent

    #     self.if_eva = True # For distinguish Train and Test 
    #     self.ep_step = 0 # Episode step counter

    #     return self.state

    def step(self, a):
        a_p = self._get_agent_position()
        d, x, y = a_p

        t = True # Termination status
        r = 0 # Reward
        info = {}

        self.total_step += 1
        self.ep_step +=1
        if self.ep_step > 500: # H = 500 Control the length of episode
            return self.state, r, t, info

        if self.if_eva is True:
            pass
        else: # Imitation Learning
            # a = self.action_dir[self.op_se[self.ep_step]]
            # self.ep_step += 1
            pass
        
        # print(a, end=" ")
        # a = (np.floor(a*6)).astype(int) # For DDPG
        # print(a)
        if a == self.MOVE : # Action "move"
            if d == 0: # To west
                if (y-1) < 0 or [x, y-1] in self.walls: # Cross the edge or Hit the wall: termination and return minus reward (-0.1)
                    t = True
                    r = self.rewards["minus"]
                else: # Move to the destination grid and return normal reward (0)
                    y -= 1 # self.state[2] -= 1
                    t = False
                    r = self.rewards["normal"]
            elif d == 1: # To south
                if (x+1) > (self.x - 1) or [x+1, y] in self.walls:
                    t = True
                    r = self.rewards["minus"]
                else:
                    x += 1 # self.state[1] += 1
                    t = False
                    r = self.rewards["normal"]
            elif d == 2: # To east
                if (y+1) > (self.y - 1) or [x, y+1] in self.walls:
                    t = True
                    r = self.rewards["minus"]
                else:
                    y += 1 # self.state[2] += 1
                    t = False
                    r = self.rewards["normal"]
            elif d == 3: # To north
                if (x-1) < 0 or [x-1, y] in self.walls:
                    t = True
                    r = self.rewards["minus"]
                else:
                    x -= 1 # self.state[1] -= 1
                    t = False
                    r = self.rewards["normal"]
        
        elif a == self.TURNLEFT : # Action "turnLeft"
            d = (d + 1) % 4 # self.state[0] = (self.state[0]+1) % 4
            t = False
            r = self.rewards["normal"]

        elif a == self.TURNRIGHT : # Action "turnRight"
            d = (d + 3) % 4 # self.state[0] = (self.state[0]+3) % 4
            t = False
            r = self.rewards["normal"]
        
        elif a == self.PICKMARKER : # Action "pickMarker"
            if [x, y] not in self.markers: # no MARKER this grid-cell: termination and return minus reward (-0.1)
                t = True
                r = self.rewards["minus"]
            else:
                t = False
                # Pick MARKER and ruturn normal reward
                if [x, y] in self.preMarkers and [x, y] not in self.postMarkers:
                    r = self.rewards["medium"]
                else: 
                    r = self.rewards["minus"]
                self.markers.remove([x, y])
    
        elif a == self.PUTMARKER : # Action "putMarker"
            if [x, y] in self.markers: # has MARKER on grid-cell: termination and return minus reward (-1)
                t = True
                r = self.rewards["minus"]
            else:
                t = False
                # Put MARKER and ruturn normal reward
                if [x, y] in self.postMarkers and [x, y] not in self.preMarkers:
                    r = self.rewards["medium"]
                else:
                    r = self.rewards["minus"]
                self.markers.append([x, y])
                
        elif a == self.FINISH : # Action "finish"
            t = True
            d_f, x_f, y_f = self.s_f[0:3]
            self.state = self._get_state_map([d, x, y], [d_f, x_f, y_f], self.walls, self.markers, self.postMarkers)
            # If the agent at target grid with the specific MARKER picked/put: return positive reward (1);
            if (self.state == self.s_f).all(): 
                r = self.rewards["positive"]
                # print("Got one! {}".format(self.env_index))
            # Else: return minus reward (-0.1)
            else: 
                r = self.rewards["minus"]
        
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(a))
        
        d_f, x_f, y_f = self.s_f[0:3]
        self.state = self._get_state_map([d, x, y], [d_f, x_f, y_f], self.walls, self.markers, self.postMarkers)
        return self.state, r, t, info # obs, reward, done, info

    def render(self, mode="human"):
        if mode != 'human':
            raise NotImplementedError()
        a_p = self._get_agent_position()
        d, a_x, a_y = a_p
        for x in range(self.x):
            print()
            for y in range(self.y):
                if x == a_x and y == a_y: # Agent current location
                    if d == 0:
                        if [x, y] in self.markers: print('W ', end="")
                        else: print('w ', end="")
                    elif d == 1:
                        if [x, y] in self.markers: print('S ', end="")
                        else: print('s ', end="")
                    elif d == 2:
                        if [x, y] in self.markers: print('E ', end="")
                        else: print('e ', end="")
                    elif d == 3:
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

            print("Enter your action: m-0 | l-1 | r-2 | pk-3 | pt-4 | f-5 or q to quit: ", end="")
            a = input()
            print()
            if a == "q":
                break
            a = int(a)
            s_1, r, termination, _ = self.step(a)
            s = s_1
            print("reward:", r)
            self.render() 

if __name__ == '__main__':
   
    env = Gridworld()
    # check_env(env)
    env.run()

