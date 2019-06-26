import argparse
import sys
import os
# import os.path
from time import gmtime, strftime

import gym
from gym import wrappers#, logger
import csv
import logging

from agents.RandomAgent import RandomAgent
from agents.DDPG import DDPG
from agents.DDPG2 import DDPG2

def run(agent_id, episode_count, start_episode, load_dir):    

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    #logger.setLevel(logger.INFO)

    env = gym.make('LunarLanderContinuous-v2')

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    time_format = strftime("%Y%m%d%H%M%S", gmtime())
    outdir = './results/' + time_format+ '-' + agent_id + '-' + str(start_episode) + '-' + str(start_episode + episode_count -1) +'/'
    #env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    #agent = RandomAgent(env.action_space)
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    #set up logger    
#    logger = logging.getLogger('gym')
#    fh = logging.FileHandler(outdir + 'application.log')
#    fh.setLevel(logging.INFO)
#    logger.addHandler(fh)
    logging.basicConfig(filename=outdir + 'application.log',level=logging.INFO)
    logger = logging.getLogger('gym')    
    
    agent = None
    if(agent_id == 'RandomAgent'):
        agent = RandomAgent(env.action_space)
    elif(agent_id == 'DDPG'):
        agent = DDPG(env)
    elif(agent_id == 'DDPG2'):
        agent = DDPG2(env)
    else:
        logger.error("Invalid Agent chosen!")
        return

    #episode_count = 2000
    score = 0
    done = False

    if load_dir is not None:
        logger.info("Load model at " + load_dir)
        agent.load_weight(load_dir + '/')
    
    file_output = outdir + 'reward.csv'
    labels = ['episode','reward']
    with open(file_output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)
    #
        for i in range(start_episode, start_episode + episode_count):
            ob = env.reset()
            agent.reset_episode(ob)
            score = 0
            logger.info("Start episode " + str(i))
            while True:
                action = agent.act(ob)
                
#                 reward = 0
#                 for _ in range(3):
#                     ob, tmp_reward, done, _ = env.step(action)
#                     reward += tmp_reward
#                     if done:
#                         break
                ob, reward, done, _ = env.step(action)
                    
                agent.step(action, reward, ob, done)
                score += reward
                if done:
                    to_write = [i] + [score]
                    writer.writerow(to_write)
                    if i % 50 ==0:
                        csvfile.flush()
                    if i % 100 == 0:
                        agent.save_weight(outdir)
                        
                    break
                # Note there's no env.render() here. But the environment still can open window and
                # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
                # Video is not recorded every episode, see capped_cubic_video_schedule for details.

        # Close the env and write monitor result info to disk
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-a', dest='agent_id', nargs='?', help='Select a agent to run',required=True)
    parser.add_argument('-e', dest='episode_count', nargs='?', help='episode count', type=int, default=1000)
    parser.add_argument('-s', dest='start_episode', nargs='?', help='episode start', type=int, default=1)
    parser.add_argument('-l', dest='load_dir', nargs='?', help='load model directory')
    args = parser.parse_args()
    run(args.agent_id, args.episode_count, args.start_episode, args.load_dir)
