# Library Imports
import os
import numpy as np
import gym
from threading import Thread
from MultiTD3 import Agent

# Absolute Path
path = os.getcwd()

# Load the Environment
env = gym.make('Pendulum-v0')

# Init. Global Replay Buffer
class ReplayBuffer:
    """Defines the Buffer dataset from which the agent learns"""
    def __init__(self, max_size, input_shape, dim_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, dim_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        
    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr +=1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        _states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, _states, dones

# Init. Agents & replay buffer
agent1 = Agent(env, 'agent1')
agent2 = Agent(env, 'agent2')
agent3 = Agent(env, 'agent3')
Buffer = ReplayBuffer(3000000, env.observation_space.shape[0], env.action_space.shape[0])

# DEF. to transfer weights
def transfer_weights(best_agent):
    agent1.actor.set_weights(best_agent.actor.get_weights())
    agent2.actor.set_weights(best_agent.actor.get_weights())
    agent3.actor.set_weights(best_agent.actor.get_weights())

    agent1.target_actor.set_weights(best_agent.target_actor.get_weights())
    agent2.target_actor.set_weights(best_agent.target_actor.get_weights())
    agent3.target_actor.set_weights(best_agent.target_actor.get_weights())
    
    agent1.critic.set_weights(best_agent.critic.get_weights())
    agent2.critic.set_weights(best_agent.critic.get_weights())
    agent3.critic.set_weights(best_agent.critic.get_weights())

    agent1.target_critic.set_weights(best_agent.target_critic.get_weights())
    agent2.target_critic.set_weights(best_agent.target_critic.get_weights())
    agent3.target_critic.set_weights(best_agent.target_critic.get_weights())

# Init. Training
n_games = 1000
agent1_scorelog = []
agent2_scorelog = []
agent3_scorelog = []

agent1_avglog = []
agent2_avglog = []
agent3_avglog = []

best_score = env.reward_range[0]

for i in range(n_games):
	
    print(f'\nTraining for Episode: {i}')

    # Agent#1 Training
    score = 0
    obs = env.reset()
    done = False
    while not done:
        action = agent1.choose_action(obs)
        _obs, reward, done, info = env.step(action)
        Buffer.store_transition(obs, action, reward, _obs, done)
        obs = _obs
        score += reward
          
    agent1.learn(Buffer, 64)
        
    agent1_scorelog.append(score)
    agent1_score = np.mean(agent1_scorelog[-100:])
    agent1_avglog.append(agent1_score)
    
    print(f'Agent#1 -> Episode:{i} \t ACC. Rewards: {score:4.2f} \t AVG. Rewards: {agent1_score:3.2f} \t Buffer Size:{Buffer.mem_cntr}' )
    np.save(path+'/MultiAgentProfiling/data/agent1_scorelog', agent1_scorelog, allow_pickle=False)
    np.save(path+'/MultiAgentProfiling/data/agent1_avglog', agent1_avglog, allow_pickle=False)
    
    # Agent#2 Training
    score = 0
    obs = env.reset()
    done = False
    while not done:
        action = agent2.choose_action(obs)
        _obs, reward, done, info = env.step(action)
        Buffer.store_transition(obs, action, reward, _obs, done)
        obs = _obs
        score += reward
          
    agent2.learn(Buffer, 64)
        
    agent2_scorelog.append(score)
    agent2_score = np.mean(agent2_scorelog[-100:])
    agent2_avglog.append(agent2_score)
    
    print(f'Agent#2 -> Episode:{i} \t ACC. Rewards: {score:4.2f} \t AVG. Rewards: {agent2_score:3.2f}\t Buffer Size:{Buffer.mem_cntr}' )
    np.save(path+'/MultiAgentProfiling/data/agent2_scorelog', agent2_scorelog, allow_pickle=False)
    np.save(path+'/MultiAgentProfiling/data/agent2_avglog', agent2_avglog, allow_pickle=False)
    
    # Agent#3 Training
    score = 0
    obs = env.reset()
    done = False
    while not done:
        action = agent3.choose_action(obs)
        _obs, reward, done, info = env.step(action)
        Buffer.store_transition(obs, action, reward, _obs, done)
        obs = _obs
        score += reward
          
    agent3.learn(Buffer, 64)
        
    agent3_scorelog.append(score)
    agent3_score = np.mean(agent3_scorelog[-100:])
    agent3_avglog.append(agent3_score)
    
    print(f'Agent#3 -> Episode:{i} \t ACC. Rewards: {score:4.2f} \t AVG. Rewards: {agent3_score:3.2f}\t Buffer Size:{Buffer.mem_cntr}' )
    np.save(path+'/MultiAgentProfiling/data/agent3_scorelog', agent3_scorelog, allow_pickle=False)
    np.save(path+'/MultiAgentProfiling/data/agent3_avglog', agent3_avglog, allow_pickle=False)

    # Compute the best performing agent
    score_frame = np.array([agent1_score, agent2_score, agent3_score])
    best_agent = np.argmax(score_frame)
    if best_agent == 0: 
        best_agent = agent1
        avg_score = agent1_score
    elif best_agent == 1:
        best_agent = agent2
        avg_score = agent2_score
    elif best_agent == 2: 
        best_agent = agent3
        avg_score = agent3_score 

    # Init. transfer
    transfer_weights(best_agent)
        
    # Save the best 'actor' model
    if avg_score > best_score:
        best_score = avg_score
        best_agent.actor.save_weights(path+'/MultiAgentProfiling/data/teamactor.h5')
        print(f'Model Saved -> Best Model:{best_agent.name}')
        
