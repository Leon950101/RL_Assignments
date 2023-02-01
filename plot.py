import matplotlib.pyplot as plt
from stable_baselines3.common.logger import read_json
import numpy as np

fig, axs = plt.subplots() # 1, 3, figsize=(14, 4)
axs.set_xlabel("Episodes") # Iterations
axs.set_ylabel("Episode Average Reward")
axs.set_title("Imitation Learning")
# axs.set_xlim([0, 750])

# Curicullum Design
plot_log_3 = np.array(read_json("reward/ac_reward_cd_0.03_0.15.json")).flatten()
axs.plot(np.arange(0, len(plot_log_3), 1), plot_log_3, label="0.03_0.15")

plot_log_4 = np.array(read_json("reward/ac_reward_cd_0.03_0.15_8m.json")).flatten()
axs.plot(np.arange(0, len(plot_log_4), 1), plot_log_4, label="8m")

plot_log_7 = np.array(read_json("reward/ac_reward_cd_3.json")).flatten()
axs.plot(np.arange(0, len(plot_log_7), 1), plot_log_7, label="cd_3")

# Imitation Learning
# plot_log_1 = np.array(read_json("reward/ac_reward_0.01.json")).flatten()
# axs.plot(np.arange(0, len(plot_log_1), 1), plot_log_1, label="0.01")

# plot_log_2 = np.array(read_json("reward/ac_reward_0.05.json")).flatten()
# axs.plot(np.arange(0, len(plot_log_2), 1), plot_log_2, label="0.05")

# plot_log_5 = np.array(read_json("reward/ac_reward_baseline.json")).flatten()
# axs.plot(np.arange(0, len(plot_log_5), 1), plot_log_5, label="0.0")

# Reward Design
# plot_log_1 = read_json("log/0001_38/progress.json")
# x_1 = plot_log_1["time/iterations"]
# y_1 = plot_log_1["rollout/ep_rew_mean"]
# axs.plot(x_1, y_1, label="None")
# plot_log_2 = read_json("log/1001_38/progress.json")
# x_2 = plot_log_2["time/iterations"]
# y_2 = plot_log_2["rollout/ep_rew_mean"]
# axs.plot(x_2, y_2, label="Termination Punishment")
# plot_log_3 = read_json("log/0011_38/progress.json")
# x_3 = plot_log_3["time/iterations"]
# y_3 = plot_log_3["rollout/ep_rew_mean"]
# axs.plot(x_3, y_3, label="Marker Reward")
# plot_log_4 = read_json("log/1011_38/progress.json")
# x_4 = plot_log_4["time/iterations"]
# y_4 = plot_log_4["rollout/ep_rew_mean"]
# axs.plot(x_4, y_4, label="MR+TP")

# axs[1].plot(x, y_2)
# axs[1].set_title('loss')
# axs.fill_between(x, mean + 0.5*std, mean - 0.5*std, alpha=0.2)

plt.legend()
plt.show()