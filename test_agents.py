# Library Imports
import os
import numpy as np
import gym
from copy import deepcopy
from MultiAgentProfiling.MultiTD3 import Agent as Team_Agent
from SingleAgentProfiling.TD3 import Agent as Solo_Agent 

# Absolute Path
path = os.getcwd()

# Load the Environment
env = gym.make('Pendulum-v0')

# Init. Agent
team_agent1 = Team_Agent(env, 'team_agent1')
team_agent2 = Team_Agent(env, 'team_agent2')
team_agent3 = Team_Agent(env, 'team_agent3')
solo_agent = Solo_Agent(env)

# Load the Trained Actor
for i in range(10):
    done = False
    obs = env.reset()
    while not done:
        action = team_agent1.choose_action(obs)
        action = team_agent2.choose_action(obs)
        action = team_agent3.choose_action(obs)
        action = solo_agent.choose_action(obs)
        _obs, reward, done, info = env.step(action)
        obs = _obs              
team_agent1.actor.load_weights(path+'/MultiAgentProfiling/data/agent1.h5')
team_agent2.actor.load_weights(path+'/MultiAgentProfiling/data/agent2.h5')
team_agent3.actor.load_weights(path+'/MultiAgentProfiling/data/agent3.h5')
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
        # Taking a step using agent1
        env.state = obs
        action = team_agent1.choose_action(obs)
        agent1_obs, agent1_reward, done, info = env.step(action)
        
        # Taking a step using agent2
        env.state = obs
        action = team_agent2.choose_action(obs)
        agent2_obs, agent2_reward, done, info = env.step(action)
        
        # Taking a step using agent3
        env.state = obs
        action = team_agent3.choose_action(obs)
        agent3_obs, agent3_reward, done, info = env.step(action)

        score_card = np.array([agent1_reward, agent2_reward, agent3_reward])
        best_step = np.argmax(score_card)
        
        if best_step == 0:
            score += agent1_reward
            obs = deepcopy(agent1_obs)
        elif best_step == 1:
            score += agent2_reward
            obs = deepcopy(agent2_obs)
        elif best_step == 2:
            score += agent3_reward
            obs = deepcopy(agent3_obs)
    
    team_scorelog.append(score)
    
env.close()
np.save(path+'/data/team_score', team_scorelog, allow_pickle=False)
