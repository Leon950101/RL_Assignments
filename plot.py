import matplotlib.pyplot as plt
from stable_baselines3.common.logger import read_json

# Plot
plot_log = read_json("log/progress.json")
x = plot_log["time/iterations"]
y = plot_log["rollout/ep_rew_mean"]
fig, axs = plt.subplots()
axs.plot(x, y, label="ep_rew_mean")
# axs.fill_between(x, mean + 0.5*std, mean - 0.5*std, alpha=0.2)

plt.legend()
plt.show()