# Capstone Project

### Domain Background

In this project, I am going to make use of reinforcement learning and implement an agent which can control a lander to land on a designated area. The environment used is OpenAI Gym LunarLanderContinuous-v2

### Project Instruction
1. Clone the repository and navigate to the downloaded folder
```bash
$git clone https://github.com/ronin-ho/udacity-rl-capstone.git
```

2. Install required packages
```bash
$pip3 install tensorflow
$pip3 install keras
$pip3 install gym
$pip3 install box2d box2d-kengz
```

3. Change the directory to src folder
4. Execute the main.py with help options
```sh
$usage: python3 main.py [-h] -a [AGENT_ID] [-t [IS_TRAINING]] [-l [LOAD_DIR]]

optional arguments:
  -h, --help          Show this help message and exit
  -a [AGENT_ID]       Select a agent to run, e.g. DDPG, RandomAgent or RBFAgent
  -t [IS_TRAINING]	  1 for training, 0 for testing
  -l [LOAD_DIR]       Path to load model
```

To launch a DDPG Agent, you can enter
```sh
$python3 main.py -a DDPG &
```
The reward episode csv file will be generated at ./results/<timestamp>-DDPG/reward.csv

To load a DDPG Agent model for testing, you can enter
```sh
$python3 main.py -a DDPG -t 0 -l ../model &
```
The agent will exeucte 5 episodes and generate a reward episode csv file at ./results/<timestamp>-DDPG/reward.csv

By default, the program does not generate a video file. You have to uncomment below line in main.py before starting the training or testing
```python
#env = wrappers.Monitor(env, directory=outdir + 'video/', force=True)
```