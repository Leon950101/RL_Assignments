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
        
        # Action Space
        self.actions = [0, 1, 2, 3, 4, 5]
        self.action_dir = {"move":0, "turnLeft":1, "turnRight":2, "pickMarker":3, "putMarker":4, "finish":5}
        # Reward Function
        self.rewards = {"minus": -0.1, "normal": 0, "medium": 0.1, "positive": 1}
        # Env Settings
        self.env_index = -1 # Enviornment index
        self.total_step = 0
        self.directions = {"west":0, "south":1, "east":2, "north":3}
        self.gamma = 0.99

        # Define action and observation space. They must be gym.spaces objects
        n_actions = len(self.actions)
        self.action_space = spaces.Discrete(n_actions)
        # self.action_space = spaces.Box(low=0, high=6, shape=(1, ), dtype=np.float32) # For DDPG
        self.observation_space = spaces.Box(low=0, high=4, shape=(38, ), dtype=np.float32) # 38 22 32
    
    def _get_state_map(self, agent_position, s_f, w, m, post_m):
        # 32 Worst
        if self.observation_space.shape[0] == 32:
            state_map = np.zeros((4, 4), dtype=int)
            for x in range(4):
                for y in range(4):
                    if [x, y] in w: state_map[x][y] = 1
                    elif [x, y] in m: state_map[x][y] = 2
                    else: state_map[x][y] = 0
                    if x == agent_position[1] and y == agent_position[2]:
                        if [x, y] not in self.markers:
                            state_map[x][y] = agent_position[0] + 3
                        else:
                            state_map[x][y] = agent_position[0] + 7
            flt_state = state_map.flatten()
            
            target_map = np.zeros((4, 4), dtype=int)
            for x in range(4):
                for y in range(4):
                    if [x, y] in w: target_map[x][y] = 1
                    elif [x, y] in post_m: target_map[x][y] = 2
                    else: target_map[x][y] = 0
                    if x == s_f[1] and y == s_f[2]:
                        if [x, y] not in post_m:
                            target_map[x][y] = s_f[0] + 3
                        else:
                            target_map[x][y] = s_f[0] + 7
            flt_target = target_map.flatten()
            
            state = np.concatenate((flt_state, flt_target))

        # 38 Best
        if self.observation_space.shape[0] == 38:
            state_map = np.zeros((4, 4), dtype=int)
            for x in range(4):
                for y in range(4):
                    if [x, y] in w: state_map[x][y] = 1
                    elif [x, y] in m: state_map[x][y] = 2
                    else: state_map[x][y] = 0
            flt_state = state_map.flatten()

            s_f_map = np.zeros((4, 4), dtype=int)
            for x in range(4):
                for y in range(4):
                    if [x, y] in w: s_f_map[x][y] = 1
                    elif [x, y] in post_m: s_f_map[x][y] = 2
                    else: s_f_map[x][y] = 0
            flt_s_f = s_f_map.flatten()
 
            state = np.concatenate((agent_position, flt_state, s_f, flt_s_f))
        
        # 22 Medium
        if self.observation_space.shape[0] == 22:
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

    def _get_agent_position(self, state):
        # 32
        if len(state) == 32:
            agent_part = state[0:16]
            for i in range(16):
                if agent_part[i] > 2:
                    x = i // 4
                    y = i % 4
                    d = (agent_part[i] + 1) % 4
            agent_position = [d, x, y]       
        # 22 38
        else:
            agent_position = state[0:3]
        return agent_position

    # Read Json File
    def _reset_all(self, fcc_data_task):
        
        self.x, self.y = fcc_data_task["gridsz_num_rows"], fcc_data_task["gridsz_num_cols"]
        self.walls = fcc_data_task["walls"] # Set walls
        self.preMarkers = fcc_data_task["pregrid_markers"]
        self.postMarkers = fcc_data_task["postgrid_markers"]
        self.markers = self.preMarkers
        d_0, x_0, y_0 = self.directions[fcc_data_task["pregrid_agent_dir"]], fcc_data_task["pregrid_agent_row"], fcc_data_task["pregrid_agent_col"]
        d_f, x_f, y_f = self.directions[fcc_data_task["postgrid_agent_dir"]], fcc_data_task["postgrid_agent_row"], fcc_data_task["postgrid_agent_col"]
        self.s_0 = self._get_state_map([d_0, x_0, y_0], [d_f, x_f, y_f], self.walls, self.preMarkers, self.postMarkers) # Init state
        self.s_f = self._get_state_map([d_f, x_f, y_f], [d_f, x_f, y_f], self.walls, self.postMarkers, self.postMarkers) # Target state
        self.state = self.s_0 # Set init state to the agent

        self.ep_step = 0 # Episode step counter

    # Basic for PPO
    def reset(self):
        # Randomly
        self.env_index = random.randint(0, 23999)
        with open('data/train/task/'+str(self.env_index)+'_task.json', 'r') as fcc_file_task:
            fcc_data_task = json.load(fcc_file_task)

        self._reset_all(fcc_data_task)
        return self.state

    # Curiculumm Design
    def reset_cd(self, easy_per, medium_per, total_step):
        if self.total_step < int(easy_per * total_step):
            self.env_index = random.randint(0, 3999)
            with open('data_easy/train/task/'+str(self.env_index)+'_task.json', 'r') as fcc_file_task:
                fcc_data_task = json.load(fcc_file_task)
            with open('data_easy/train/seq/'+str(self.env_index)+'_seq.json', 'r') as fcc_file_seq:
                fcc_data_seq = json.load(fcc_file_seq)
        elif self.total_step < int(medium_per * total_step):
            self.env_index = random.randint(0, 11999)
            with open('data_medium/train/task/'+str(self.env_index)+'_task.json', 'r') as fcc_file_task:
                fcc_data_task = json.load(fcc_file_task)
            with open('data_medium/train/seq/'+str(self.env_index)+'_seq.json', 'r') as fcc_file_seq:
                fcc_data_seq = json.load(fcc_file_seq)
        else:
            self.env_index = random.randint(0, 23999)
            with open('data/train/task/'+str(self.env_index)+'_task.json', 'r') as fcc_file_task:
                fcc_data_task = json.load(fcc_file_task)
            with open('data/train/seq/'+str(self.env_index)+'_seq.json', 'r') as fcc_file_seq:
                fcc_data_seq = json.load(fcc_file_seq)
        
        self.op_se = fcc_data_seq["sequence"]
        self._reset_all(fcc_data_task)
        return self.state, self.op_se
    
    # Imitation Learnning
    def reset_il(self):
        
        # Randomly
        self.env_index = random.randint(0, 23999)

        with open('data/train/task/'+str(self.env_index)+'_task.json', 'r') as fcc_file_task:
            fcc_data_task = json.load(fcc_file_task)
        with open('data/train/seq/'+str(self.env_index)+'_seq.json', 'r') as fcc_file_seq:
            fcc_data_seq = json.load(fcc_file_seq)     
            
        self.op_se = fcc_data_seq["sequence"]
        self._reset_all(fcc_data_task)
        return self.state, self.op_se
    
    # Test
    def reset_val(self, map_index, label):
        
        self.env_index = map_index
        with open('data/'+label+'/task/'+str(map_index)+'_task.json', 'r') as fcc_file_task:
            fcc_data_task = json.load(fcc_file_task)
        
        self._reset_all(fcc_data_task)
        return self.state

    # I.5
    def reset_file(self, file):
        
        with open(file, 'r') as fcc_file_task:
            fcc_data_task = json.load(fcc_file_task)
        
        self._reset_all(fcc_data_task)
        return self.state

    def step(self, a):
        d, x, y = self._get_agent_position(self.state)
        t = True # Termination status
        r = 0 # Reward
        info = {}

        self.total_step += 1
        self.ep_step +=1
        if self.ep_step > 100: # H = 500 Control the length of episode
            return self.state, r, t, info
        
        if a == self.MOVE : # Action "move"
            if d == 0: # West
                x_t = x
                y_t = y - 1 
            elif d == 1: # South
                x_t = x + 1
                y_t = y
            elif d == 2: # East
                x_t = x
                y_t = y + 1 
            elif d == 3: # North
                x_t = x - 1
                y_t = y
            
            # Cross the edge or Hit the wall: termination and return minus reward (-0.1)
            if x_t < 0 or x_t > (self.x - 1) or y_t < 0 or (y_t > self.y - 1) or [x_t, y_t] in self.walls:
                t = True
                r = self.rewards["minus"]
            # Move to the destination grid and return normal reward (0)
            else:
                x, y = x_t, y_t
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
                    r = - self.rewards["medium"]
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
                    r = - self.rewards["medium"]
                self.markers.append([x, y])
                
        elif a == self.FINISH : # Action "finish"
            t = True 
            d_f, x_f, y_f = self._get_agent_position(self.s_f)
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
        
        d_f, x_f, y_f = self._get_agent_position(self.s_f)
        self.state = self._get_state_map([d, x, y], [d_f, x_f, y_f], self.walls, self.markers, self.postMarkers)
        return self.state, r, t, info # obs, reward, done, info

    def render(self, mode="human"):
        dir_s= {'0':'w ', '1':'s ', '2':'e ', '3':'n '}
        dir_c = {'0':'W ', '1':'S ', '2':'E ', '3':'N '}
        if mode != 'human':
            raise NotImplementedError()
        a_d, a_x, a_y = self._get_agent_position(self.state)
        for x in range(self.x):
            print()
            for y in range(self.y):
                if x == a_x and y == a_y: # Agent current location
                    if [x, y] in self.markers: print(dir_c[str(a_d)], end="")
                    else: print(dir_s[str(a_d)], end="")
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

