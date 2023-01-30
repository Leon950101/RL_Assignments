import matplotlib.pyplot as plt
from stable_baselines3.common.logger import read_json

# Plot
plot_log = read_json("log/progress.json")
x = plot_log["time/iterations"]
y_1 = plot_log["rollout/ep_rew_mean"]
y_2 = plot_log["train/loss"]
fig, axs = plt.subplots(1, 2, figsize=(10, 6))
axs[0].plot(x, y_1)
axs[0].set_title('ep_rew_mean')
axs[1].plot(x, y_2)
axs[1].set_title('loss')
# axs.fill_between(x, mean + 0.5*std, mean - 0.5*std, alpha=0.2)

# plt.legend()
plt.show()