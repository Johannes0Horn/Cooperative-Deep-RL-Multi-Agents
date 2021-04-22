##############################################################################
# Library Imports
##############################################################################
from collections import deque
import random
import numpy as np

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

##############################################################################
# Class for Defining the Deep Q-Network Agent
##############################################################################
class DQN_Agent:
    def __init__(self,env,name):
        
        #Hyperparams.
        self.env= env
        self.gamma= 0.99
        self.len_episode= env._max_episode_steps
        self.epsilon= 1
        self.epsilon_decay= 0.05
        self.epsilon_min= 0.01
        self.learning_rate= 0.001
        self.best_score = -np.Inf
        self.name = name
        
        self.replay_memory= deque(maxlen= 20000)
        self.buffer_size= 32
        
        self.train_network= self.Policy_Network()
        self.target_network= self.Policy_Network()
        self.target_network.set_weights(self.train_network.get_weights())
        
        # Detect GPU for accelerating the computation
        gpus = tf.config.experimental.list_physical_devices('GPU')

        if len(gpus) !=0:
            self.device= '/gpu:0'
        else:
            self.device= '/cpu:0'

    ##############################################################################
    def Policy_Network(self):
        """ Creates a dense network to estimate a policy """
        
        model = models.Sequential()
        state_shape = self.env.observation_space.shape

        model.add(Dense(8, activation='relu', input_shape=state_shape))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    ##############################################################################
    def epsgreedyaction(self,state):
        """ Agent picks an action based on Epsilon-Greedy Policy """
        
        self.epsilon= max(self.epsilon_min, self.epsilon)

        if np.random.rand(1) < self.epsilon:
            action= self.env.action_space.sample()
        else:
            action= np.argmax(self.train_network.predict(state)[0])

        return action

    ##############################################################################
    def train_buffer(self):
        """ Dataset based optimization using Tensorflow """
     
        if len(self.replay_memory) < self.buffer_size:
            return

        samples= random.sample(self.replay_memory,self.buffer_size)

        states= []
        new_states= []
        for sample in samples:
            state, action, reward, new_state, done = sample
            states.append(state)
            new_states.append(new_state)

        states= np.array(states)
        new_states= np.array(new_states)
        
        states= states.reshape(self.buffer_size, 2)
        new_states= new_states.reshape(self.buffer_size, 2)

        targets= self.train_network.predict(states)
        new_state_targets= self.target_network.predict(new_states)

        for i,sample in enumerate(samples):
            state, action, reward, new_state, done = sample
            target= targets[i]
            if done:
                target[action]= reward
            else:
                Q_future= max(new_state_targets[i])
                target[action]= reward + Q_future * self.gamma
        
        tf.debugging.set_log_device_placement(True)
        with tf.device(self.device):
            self.train_network.fit(states, targets, epochs=1, verbose=0)

    ##############################################################################
    def run(self):
        """ Generate Episodic Data """
        
        sum_reward= 0
        done= False
        current_state= self.env.reset().reshape(1,2)
        score_log = []
        
        for i in range(self.len_episode):
            opt_action= self.epsgreedyaction(current_state)
            new_state, reward, done, _= self.env.step(opt_action)
            new_state = new_state.reshape(1, 2)
                
            # Reward Engineering
            if new_state[0][0] >= 0.5:
                reward+= 10

            self.replay_memory.append([current_state, opt_action, reward, new_state, done])
            self.train_buffer()

            sum_reward+= reward
            current_state= new_state

            if done:
                self.env.close()
                break
            
            score_log.append(sum_reward)
            
        if np.mean(score_log[-10:]) > self.best_score:
            self.best_score = np.mean(score_log[-10:])
            self.train_network.save('data/'+self.name+'.h5')
            self.best_score = sum_reward
 
        # Exploration Decay
        self.epsilon -= self.epsilon_decay
            
