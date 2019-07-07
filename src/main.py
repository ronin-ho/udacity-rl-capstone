import argparse
import sys
import os
from time import gmtime, strftime

import gym
from gym import wrappers
import csv
import logging
from copy import deepcopy

import numpy as np

from agents.RandomAgent import RandomAgent
from agents.DDPG import DDPG
from agents.RBFAgent import RBFAgent

def run(agent_id, is_training, load_dir):    
    env = gym.make('LunarLanderContinuous-v2')

    time_format = strftime("%Y%m%d%H%M%S", gmtime())
    outdir = './results/' + time_format+ '-' + agent_id +'/'
    
    if not os.path.exists(outdir + 'video/'):
        os.makedirs(outdir + 'video/')
    
    #Enable below comment to record video
    #env = wrappers.Monitor(env, directory=outdir + 'video/', force=True)
    env.seed(123)
    np.random.seed(123)

    #set up logger    
    logging.basicConfig(filename=outdir + 'application.log',level=logging.INFO)
    logger = logging.getLogger('gym')    
    
    agent = None
    if(agent_id == 'RandomAgent'):
        agent = RandomAgent(env)
    elif(agent_id == 'DDPG'):
        agent = DDPG(env)
        agent.build_models()
    elif(agent_id == 'RBFAgent'):
        agent = RBFAgent(env)
    else:
        logger.error("Invalid Agent chosen!")
        return

    if load_dir is not None:
        logger.info("Load model at " + load_dir)
        agent.load_weight(load_dir + '/')
    
    file_output = outdir + 'reward.csv'
    labels = ['episode','reward']
    with open(file_output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)

        episode = np.int16(1)
        step = np.int16(0)
        observation = None
        episode_reward = None
        max_steps = 1000000
        max_episodes = 1100 + episode
        
        try:
            while step < max_steps and episode < max_episodes:
                
                if observation is None:  # start of a new episode
                    logger.info("Start episode " + str(episode))
                    episode_reward = np.float32(0)

                    # Obtain the initial observation by resetting the environment.
                    agent.reset_episode()
                    observation = deepcopy(env.reset())

                action = agent.act(observation)
                    
                reward = np.float32(0)
                done = False
                
                next_state, r, done, info = env.step(action)
                next_state = deepcopy(next_state)
                reward = r
                
                if is_training:
                    agent.step(observation, action, reward, next_state, done)
                    
                episode_reward += reward
                step += 1
                
                observation = next_state
                
                if episode % 20 ==0:
                    logger.info("State-Action: {}, {}, {}, {}, reward: {}".format(observation[0], observation[1], action[0],action[1], reward))
                    
                if done:
                    # Act on the final state
                    # Step on final state but without adding to memory as next state is the reset state
                    action = agent.act(next_state)
                    agent.step_without_memory()

                    to_write = [episode] + [episode_reward]
                    writer.writerow(to_write)
                    if episode % 20 ==0:
                        csvfile.flush()
                    if episode % 20 == 0:
                        agent.save_weight(outdir)
                        
                    episode += 1
                    observation = None
                    episode_reward = None    
                    
        except KeyboardInterrupt:
            csvfile.flush()
            agent.save_weight(outdir)
            
        # Close the env
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-a', dest='agent_id', nargs='?', help='Select a agent to run',required=True)
    parser.add_argument('-t', dest='is_training', nargs='?', help='1 for training, 0 for testing', type=int, default=1)
    parser.add_argument('-l', dest='load_dir', nargs='?', help='Path to load model or weight')
    args = parser.parse_args()
    run(args.agent_id, args.is_training, args.load_dir)
