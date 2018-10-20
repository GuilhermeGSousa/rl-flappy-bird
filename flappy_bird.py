import logging
import os, sys
import time
import matplotlib.pyplot as plt
from collections import deque

import gym
from gym.wrappers import Monitor
from agent import DQN, preprocess
import numpy as np

import gym_ple

if __name__ == '__main__':

    N_EP = 10000
    N_SAVE = 500
    env = gym.make('FlappyBird-v0')
    agent = DQN(env)
    scores = deque(maxlen=100)
    for i in range(N_EP):
        score = 0
        ob = env.reset()

        # Stack observations
        pre_ob = preprocess(ob)
        pre_ob = pre_ob.reshape(1, 100, 100)
        ob_stack = np.stack((pre_ob,)*4, -1)
        pre_ob = ob_stack

        while True:
            action = agent.act(pre_ob, step = i)

            ob, reward, done, _ = env.step(action)
            if reward <= -1:
                reward = -1

            next_pre_ob = preprocess(ob)

            # Stack observations
            next_pre_ob = next_pre_ob.reshape(1, 100, 100)
            ob_stack = np.insert(ob_stack, -1, next_pre_ob, axis=3)
            ob_stack = np.delete(ob_stack, 0, axis = 3)
            next_pre_ob = ob_stack

            agent.remember(pre_ob, action, reward, next_pre_ob, done)
            agent.replay()
            pre_ob = next_pre_ob
            score = score + reward

            if done:
                break

        scores.append(score)
        print("Episode {} score: {}".format(i + 1, score))
        mean_score = np.mean(scores)
        
        if (i + 1) % 5 == 0:
            print("Episode {}, score: {}, exploration at {}%, mean of last 100 episodes was {}".format(i + 1, score,agent.epsilon * 100, mean_score))

        if ((i + 1) % N_SAVE == 0 and i > 0):
            agent.model.save_weights("models/model_{}.h5".format(i + 1))
            agent.target.save_weights("models/target_{}.h5".format(i + 1))
