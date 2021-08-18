# Library Imports
import os
import numpy as np
import gym
from TD3 import Agent

# Absolute Path
path = os.getcwd()

# Load the Environment
env = gym.make('Pendulum-v0')

# Init. Agent
agent = Agent(env)

# Init. Training
n_games = 1000
score_history = []
avg_history = []
best_score = env.reward_range[0]
avg_score = 0

for i in range(n_games):
    score = 0
    done = False

    # Initial Reset of Environment
    obs = env.reset()

    while not done:
        action = agent.choose_action(obs)
        _obs, reward, done, info = env.step(action)
        agent.store_exp(obs, action, reward, _obs, done)
        obs = _obs
        score += reward
    
    # Optimize the Agent    
    agent.learn(64)
        
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    avg_history.append(avg_score)

    if avg_score > best_score:
        best_score = avg_score
        agent.save_models(path+'/SingleAgentProfiling/data/')
        print(f'Episode:{i} \t ACC. Rewards: {score:3.2f} \t AVG. Rewards: {avg_score:3.2f} \t *** MODEL SAVED! ***')
    else:
        print(f'Episode:{i} \t ACC. Rewards: {score:3.2f} \t AVG. Rewards: {avg_score:3.2f}')
        
    # Save the Training data and Model Loss
    np.save(path+'/SingleAgentProfiling/data/score_history', score_history, allow_pickle=False)
    np.save(path+'/SingleAgentProfiling/data/avg_history', avg_history, allow_pickle=False)