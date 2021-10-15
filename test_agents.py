# Library Imports
import os

# this line allows control-c while executing
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import numpy as np
from copy import deepcopy
from MultiAgentProfiling.MultiTD3 import Agent as Team_Agent
from SingleAgentProfiling.TD3 import Agent as Solo_Agent
from gym.wrappers.time_limit import TimeLimit
from custom_pendulum import CustomPendulum
from gym.envs.classic_control.pendulum import PendulumEnv

# Absolute Path
path = os.getcwd()

# Load the Environment
env = TimeLimit(env=CustomPendulum(PendulumEnv(), _seed=1), max_episode_steps=200)
# env = gym.make('Pendulum-v0')
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
team_agent1.actor.load_weights(path + '/MultiAgentProfiling/data/agent1.h5')
team_agent2.actor.load_weights(path + '/MultiAgentProfiling/data/agent2.h5')
team_agent3.actor.load_weights(path + '/MultiAgentProfiling/data/agent3.h5')
solo_agent.actor.load_weights(path + '/SingleAgentProfiling/data/actor.h5')

# Init. & Profile the Solo and Team Agent
print(f'Testing Solo Agent...')
solo_scorelog = []
solo_epslog = []
score = 0
env = TimeLimit(env=CustomPendulum(PendulumEnv(), _seed=1), max_episode_steps=200)
for i in range(100):
    done = False
    obs = env.reset()
    i = 0
    while not done:
        # env.render()
        action = solo_agent.choose_action(obs)
        _obs, reward, done, info = env.step(action)
        obs = _obs
        score += reward
        i += 1
    solo_scorelog.append(score)
    solo_epslog.append(i)
env.close()
np.save(path + '/data/solo_score', solo_scorelog, allow_pickle=False)
np.save(path + '/data/solo_eps', solo_epslog, allow_pickle=False)

print(f'Testing Team Agent...')
team_scorelog = []
team_epslog = []
score = 0
env = TimeLimit(env=CustomPendulum(PendulumEnv(), _seed=1), max_episode_steps=200)
for i in range(100):
    done = False
    obs = env.reset()
    state = env.get_state()
    i = 0
    while not done:
        # env.render()
        # Taking a step using agent1
        env.set_state(state)
        action = team_agent1.choose_action(obs)
        agent1_obs, agent1_reward, done, info = env.step(action)
        agent1_state = env.get_state()

        # Taking a step using agent2
        env.set_state(state)
        action = team_agent2.choose_action(obs)
        agent2_obs, agent2_reward, done, info = env.step(action)
        agent2_state = env.get_state()

        # Taking a step using agent3
        env.set_state(state)
        action = team_agent3.choose_action(obs)
        agent3_obs, agent3_reward, done, info = env.step(action)
        agent3_state = env.get_state()

        score_card = np.array([agent1_reward, agent2_reward, agent3_reward])
        best_step = np.argmax(score_card)

        if best_step == 0:
            score += agent1_reward
            obs = deepcopy(agent1_obs)
            state = deepcopy(agent1_state)
        elif best_step == 1:
            score += agent2_reward
            obs = deepcopy(agent2_obs)
            state = deepcopy(agent2_state)
        elif best_step == 2:
            score += agent3_reward
            obs = deepcopy(agent3_obs)
            state = deepcopy(agent3_state)

        i += 1

    team_scorelog.append(score)
    team_epslog.append(i)

env.close()
np.save(path + '/data/team_score', team_scorelog, allow_pickle=False)
np.save(path + '/data/team_eps', team_epslog, allow_pickle=False)
