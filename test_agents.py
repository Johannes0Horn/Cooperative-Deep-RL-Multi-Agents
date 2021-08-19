# Library Imports
import os
import numpy as np
import gym
from MultiAgentProfiling.MultiTD3 import Agent as Team_Agent
from SingleAgentProfiling.TD3 import Agent as Solo_Agent 

# Absolute Path
path = os.getcwd()

# Load the Environment
env = gym.make('Pendulum-v0')

# Init. Agent
team_agent = Team_Agent(env, 'solo')
solo_agent = Solo_Agent(env)

# Load the Trained Actor
for i in range(10):
    done = False
    obs = env.reset()
    while not done:
        action = team_agent.choose_action(obs)
        action = solo_agent.choose_action(obs)
        _obs, reward, done, info = env.step(action)
        obs = _obs              
team_agent.actor.load_weights(path+'/MultiAgentProfiling/data/teamactor.h5')
solo_agent.actor.load_weights(path+'/SingleAgentProfiling/data/actor.h5')

# Init. & Profile the Solo and Team Agent
print (f'Testing Solo Agent...')
solo_scorelog = []
score = 0
for i in range(100):
    done = False
    obs = env.reset()
    while not done:
        #env.render()
        action = solo_agent.choose_action(obs)
        _obs, reward, done, info = env.step(action)
        obs = _obs
    score += reward
    solo_scorelog.append(score)
env.close()
np.save(path+'/data/solo_score', solo_scorelog, allow_pickle=False)

print (f'Testing Team Agent...')
team_scorelog = []
score = 0
for i in range(100):
    done = False
    obs = env.reset()
    while not done:
        #env.render()
        action = solo_agent.choose_action(obs)
        _obs, reward, done, info = env.step(action)
        obs = _obs
    score += reward
    team_scorelog.append(score)
env.close()
np.save(path+'/data/team_score', team_scorelog, allow_pickle=False)

