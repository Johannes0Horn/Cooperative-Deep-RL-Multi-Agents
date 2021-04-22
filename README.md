# A Simpler Federated Deep Reinforcement Learning
This repository implements a simpler version of 'Federated Deep Reinforcement Learning. The Agents try to solve the 'MountainCar-V0' openai-gym environment and are based on DQN. This implementation does not use ROS.
## About the FRL based Agents,
* One of the agents reached reached the goal position at epoch: 14
* Then the best agent's model parameters are shared to the other agents. 
* The other agents quickly learn to solve the environments using those shared parameters.
* The experience buffer is unique to the agents are not shared. 
* Results,
    * Performance Plot,
    <p ><img src="data/Performance Plot.png" width="500" ></p>

## Dependencies
Install dependencies using:
```bash
pip3 install -r requirements.txt 
```

## Contact
* email: navalekanishk@gmail.com