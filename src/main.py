import argparse
import sys

import gym
from gym import wrappers, logger
import csv

from agents.RandomAgent import RandomAgent
from agents.DDPG import DDPG

def run(agent_id):    

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.setLevel(logger.INFO)

    env = gym.make('LunarLanderContinuous-v2')

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = './tmp/random-agent-results'
    #env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    #agent = RandomAgent(env.action_space)
    agent = None
    if(agent_id == 'RandomAgent'):
        agent = RandomAgent(env.action_space)
    elif(agent_id == 'DDPG'):
        agent = DDPG(env)
    else:
        logger.error("Invalid Agent chosen!")
        return

    episode_count = 1000
    score = 0
    done = False

    #
    file_output = 'data.txt'
    labels = ['episode','reward']
    with open(file_output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)
    #
        for i in range(episode_count):
            ob = env.reset()
            agent.reset_episode(ob)
            score = 0
            logger.info("Start episode " + str(i+1))
            while True:
                action = agent.act(ob)
                
                reward = 0
                for _ in range(3):
                    ob, tmp_reward, done, _ = env.step(action)
                    reward += tmp_reward
                    if done:
                        break
                    
                agent.step(action, reward, ob, done)
                score += reward
                if done:
                    to_write = [i+1] + [score]
                    writer.writerow(to_write)
                    if i % 50 ==0:
                        csvfile.flush()
                    break
                # Note there's no env.render() here. But the environment still can open window and
                # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
                # Video is not recorded every episode, see capped_cubic_video_schedule for details.

        # Close the env and write monitor result info to disk
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-a', dest='agent_id', nargs='?', help='Select a agent to run',required=True)
    args = parser.parse_args()
    run(args.agent_id)
