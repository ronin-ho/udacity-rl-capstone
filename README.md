# Capstone Project

### Domain Background

In this project, I am going to make use of reinforcement learning and implement an agent which can control a lander to land on a designated area. The environment used is OpenAI Gym LunarLanderContinuous-v2

### Project Status
Implementing

### Project Instruction
1. Clone the repository and navigate to the downloaded folder
```bash
$git clone https://github.com/ronin-ho/udacity-rl-capstone.git
```

2. Install required packages
```bash
$pip install keras
$pip install tensorflow
$pip install gym
```

3. Execute the main.py with help options
```sh
$usage: main.py [-h] -a [AGENT_ID] [-t [IS_TRAINING]] [-l [LOAD_DIR]]

optional arguments:
  -h, --help          Show this help message and exit
  -a [AGENT_ID]       Select a agent to run, e.g. DDPG, RandomAgent or RBFAgent
  -t [IS_TRAINING]	  1 for training, 0 for testing
  -l [LOAD_DIR]       Path to load model or weight
```
