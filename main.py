
##############################################################################
# Library Imports
##############################################################################
import gym
import numpy as np
from Agent import DQN_Agent
from concurrent.futures import ProcessPoolExecutor

##############################################################################
# Main!
##############################################################################
if __name__ == "__main__":
    # Train the DQN Agent
    env= gym.make('MountainCar-v0')
    n_episodes = 500
    score_log = []
    
    # Initiate 3Nos. of Agent
    Alpha_Agent = DQN_Agent(env, 'alpha')
    Beta_Agent = DQN_Agent(env, 'beta')
    Delta_Agent = DQN_Agent(env, 'delta')
    
    # Train the Agents for n_episodes
    for i in range(n_episodes):
        
        # Multiprocessing 
        with ProcessPoolExecutor(2) as executor:
            p1 = executor.submit(Alpha_Agent.run())
            p2 = executor.submit(Beta_Agent.run())
            p3 = executor.submit(Delta_Agent.run())
        
        # Log the Scores    
        score_frame = np.array([Alpha_Agent.best_score, Beta_Agent.best_score, Delta_Agent.best_score])
        score_log.append(score_frame)
        np.savez('data/score_log', np.vstack(score_log), allow_pickle=False)
        print (f'Agent Scores at Episode: {i} are {score_frame}')
        
        # Check the best performing Agent
        OP_Agent = np.argmax(score_frame)
        if OP_Agent == 0:
            OP_Agent = Alpha_Agent
        if OP_Agent == 1:
            OP_Agent = Beta_Agent
        if OP_Agent == 2:
            OP_Agent = Delta_Agent    
            
        # Transfer the weights to 'target_network' and 'train_network'
        Alpha_Agent.train_network.set_weights(OP_Agent.train_network.get_weights())
        Beta_Agent.train_network.set_weights(OP_Agent.train_network.get_weights())
        Delta_Agent.train_network.set_weights(OP_Agent.train_network.get_weights())
        
        Alpha_Agent.target_network.set_weights(OP_Agent.train_network.get_weights())
        Beta_Agent.target_network.set_weights(OP_Agent.train_network.get_weights())
        Delta_Agent.target_network.set_weights(OP_Agent.train_network.get_weights())
        
        # Save the OP model
        OP_Agent.train_network.save('Federated_Model.h5')
            
        
    

    
    
    

