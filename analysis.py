# Library Import
import matplotlib.pyplot as plt
import numpy as np

# Load the score_log.npz
score_log = np.load('data/score_log.npz', allow_pickle=False)
score_log = score_log['arr_0']

# Plot the graphs
plt.plot(score_log[:,0], label='Alpha_Agent')
plt.plot(score_log[:,1], label='Beta_Agent')
plt.plot(score_log[:,2], label='Delta_Agent')
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.grid(True)
plt.legend(loc='best')
plt.title("Performance Mapping of the Agents")
plt.savefig('data/Performance Plot.png')