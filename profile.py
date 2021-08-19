# Library Import
import numpy as np
import matplotlib.pyplot as plt
import os

# Import Data
agent1 = np.load(os.getcwd()+'/MultiAgentProfiling/data/agent1_avglog.npy')
agent2 = np.load(os.getcwd()+'/MultiAgentProfiling/data/agent2_avglog.npy')
agent3 = np.load(os.getcwd()+'/MultiAgentProfiling/data/agent3_avglog.npy')
agent0 = np.load(os.getcwd()+'/MultiAgentProfiling/data/agent3_avglog.npy')

#Plot and Save the Graphs
plt.figure(1)
plt.plot(agent0, color='black', label='solo_agent')
plt.plot(agent1, color='red', label='inteam_agent1')
plt.plot(agent2, color='blue', label='inteam_agent2')
plt.plot(agent3, color='green', label='inteam_agent3')
plt.grid(True)
plt.xlabel('Episodes')
plt.ylabel('Avg. Rewards')
plt.title("Collective Training Profile")
plt.legend(loc='best')
plt.savefig(os.getcwd()+'/data/Collective Training Profile.png')

# Import Data
solo_score = np.load(os.getcwd()+'/data/solo_score.npy')
agent_score = np.load(os.getcwd()+'/data/team_score.npy')

#Plot and Save the Graphs
plt.figure(2)
plt.plot(np.abs(solo_score), label='Solo Agent')
plt.plot(np.abs(agent_score), label='Team Agent')
plt.grid(True)
plt.xlabel('Episodes')
plt.ylabel('Total Rewards')
plt.title("Agent Testing Profile")
plt.legend(loc='best')
plt.savefig(os.getcwd()+'/data/Agent Testing Profile.png')
