import json
import random
import numpy as np
 
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

def init_state_map(agent_position, w, m):
    state_map = np.zeros((4, 4), dtype=int)
    for x in range(4):
        for y in range(4):
            if [x, y] in w: state_map[x][y] = 1
            elif [x, y] in m: state_map[x][y] = 2
            else: state_map[x][y] = 0
    flatten = state_map.flatten()
    state = np.concatenate((agent_position, flatten))
    return state

env_id = random.randint(0, 1) # 23999
with open('data/train/task/'+str(env_id)+'_task.json', 'r') as fcc_file:
    fcc_data = json.load(fcc_file)
w = fcc_data["walls"]
m_pregrid = fcc_data["pregrid_markers"]
m_postgrid = fcc_data["postgrid_markers"]
m = fcc_data["pregrid_markers"]
x, y = fcc_data["gridsz_num_rows"], fcc_data["gridsz_num_cols"]
dir = {"west":0, "south":1, "east":2, "north":3}
d_0, x_0, y_0 = dir[fcc_data["pregrid_agent_dir"]], fcc_data["pregrid_agent_row"], fcc_data["pregrid_agent_col"]
d_f, x_f, y_f = dir[fcc_data["postgrid_agent_dir"]], fcc_data["postgrid_agent_row"], fcc_data["postgrid_agent_col"]
s_0 = init_state_map([d_0, x_0, y_0], w, m_pregrid) # Init state
s_f = init_state_map([d_f, x_f, y_f], w, m_postgrid) # Target state 

print(s_0)
print(s_f)