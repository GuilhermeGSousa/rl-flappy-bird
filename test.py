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

    N_EP = 10
    N_REPS = 10
    env = gym.make('FlappyBird-v0')
    env.seed(2)
    agent = DQN(env)


    agent.model.load_weights("models/model_{}.h5".format(8000))

    for j in range(N_REPS):
        ob = env.reset()
        pre_ob = preprocess(ob)
        pre_ob = pre_ob.reshape(1, 100, 100)
        ob_stack = np.stack((pre_ob,) * 4, -1)
        while True:
            action = agent.act(ob_stack)
            ob, reward, done, _ = env.step(action)

            next_pre_ob = preprocess(ob)
            next_pre_ob = next_pre_ob.reshape(1, 100, 100)
            ob_stack = np.insert(ob_stack, 0, next_pre_ob, axis=3)
            ob_stack = np.delete(ob_stack, -1, axis = 3)

            if done:
                break
            env.render()
            time.sleep(0.05)
