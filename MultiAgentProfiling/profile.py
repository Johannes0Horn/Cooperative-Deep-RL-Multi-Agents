# Library Imports
import os
import numpy as np
import matplotlib.pyplot as plt

# Import Data
avg = np.load(os.getcwd()+'/MultiAgentProfiling/data/agent1_avglog.npy')
sum = np.load(os.getcwd()+'/MultiAgentProfiling/data/agent1_scorelog.npy')

#Plot and Save the Graphs
plt.figure(1)
plt.plot(sum, color='red', alpha=0.2, label='Summed Rewards')
plt.plot(avg, color='red', label='Average Rewards')
plt.grid(True)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title("'agent1' Training Profile")
plt.legend(loc='best')
plt.savefig(os.getcwd()+'/MultiAgentProfiling/data/agent1 Training Profile.png')

avg = np.load(os.getcwd()+'/MultiAgentProfiling/data/agent2_avglog.npy')
sum = np.load(os.getcwd()+'/MultiAgentProfiling/data/agent2_scorelog.npy')

#Plot and Save the Graphs
plt.figure(2)
plt.plot(sum, color='green', alpha=0.2, label='Summed Rewards')
plt.plot(avg, color='green', label='Average Rewards')
plt.grid(True)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title("'agent2' Training Profile")
plt.legend(loc='best')
plt.savefig(os.getcwd()+'/MultiAgentProfiling/data/agent2 Training Profile.png')

avg = np.load(os.getcwd()+'/MultiAgentProfiling/data/agent3_avglog.npy')
sum = np.load(os.getcwd()+'/MultiAgentProfiling/data/agent3_scorelog.npy')

#Plot and Save the Graphs
plt.figure(3)
plt.plot(sum, color='blue', alpha=0.2, label='Summed Rewards')
plt.plot(avg, color='blue', label='Average Rewards')
plt.grid(True)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title("'agent3' Training Profile")
plt.legend(loc='best')
plt.savefig(os.getcwd()+'/MultiAgentProfiling/data/agent3 Training Profile.png')