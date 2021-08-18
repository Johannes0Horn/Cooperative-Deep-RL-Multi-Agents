# Library Imports
import gym
import numpy as np

# Initialize Env.
env = gym.make('Pendulum-v0')

# Reset Env.
env.reset()

# Render the environment
print (f"Starting Random Play")
done = False
while not done:
    env.render()
    obs, reward, done, info = env.step(env.action_space.sample())
    print(f'obs ={np.around(obs, 3)} \t reward ={reward:.3f} \t done ={done} \t info ={info}')
env.close()

# Print Details
print(f'\nENV. Parameters')
print(f'Obsevation Space: Shape ={env.observation_space.shape} \t High ={env.observation_space.high} \t Low ={env.observation_space.low}')
print(f'Action Space: Shape ={env.action_space.shape}\t High ={env.action_space.high} \t Low ={env.action_space.low}')
print(f'Reward Space: Range ={env.reward_range}')