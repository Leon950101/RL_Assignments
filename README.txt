conda create -n rl python=3.8
pip install stable-baselines3

cd Documents/New/Courses/RL/Project/code
conda activate rl

ppo.py for train ppo
ac.py for train ac
reinforce.py for train reinforce

evaluation.py for test
gridworld.py for MDP
plot.py for plotting