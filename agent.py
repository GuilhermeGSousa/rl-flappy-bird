import time
import random
import gym
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import itertools
import random

import cv2

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D,MaxPooling2D, Flatten, Reshape
from keras.optimizers import Adam

def preprocess(state):
    x_t = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    #x_t = cv2.adaptiveThreshold(x_t, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 30)
    #x_t = cv2.normalize(x_t, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    x_t = cv2.threshold(x_t, 170, 255, cv2.THRESH_BINARY)[1]
    x_t = cv2.resize(x_t, (100, 100))
    x_t = 1/255 * x_t
    return x_t

class DQN():
    def __init__(self, env,  gamma=0.99, warmup = True,
        memory_size = 5000, epsilon=1.0, epsilon_min=0.0, epsilon_decay=0.0005,
        alpha=0.00025, alpha_decay=0.0, batch_size=32, tau = 0.0001):

        # warmup: whether to pre-fill memory with random action_space

        self.env = env
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.warmup = warmup
        self.gamma = gamma

        #Exploration
        self.epsilon = epsilon
        self.epsilon_max = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.explore_step = None

        #Training
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size

        self.tau = tau
        self.input_shape = (100, 100, 4)

        #Models
        self.model = self.create_model()
        self.target = self.create_model()
        self.update_target()

    def reset_target(self):
        self.target.set_weights(self.model.get_weights())

    def update_target(self):
        model_weights = self.model.get_weights()
        target_weights = self.target.get_weights()
        updated_weights = []
        for i in range(len(model_weights)):
            updated_weights.append(self.tau * model_weights[i] + (1 - self.tau) * target_weights[i])

        self.target.set_weights(updated_weights)

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(8, 8), strides=(3, 3),
                 activation='relu',
                 input_shape = self.input_shape))
        model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2),
                 activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, step = None):
        if step is not None:
            if len(self.memory) < self.memory_size:
                if self.warmup:
                    return self.env.action_space.sample()
                else:
                    return np.argmax(self.model.predict(state))
            else:
                if self.explore_step is None:
                    self.explore_step = step

                self.epsilon = np.maximum(self.epsilon_min, -self.epsilon_decay * (step - self.explore_step) + self.epsilon_max)
                return self.env.action_space.sample() if (np.random.random() <= self.epsilon) else np.argmax(self.model.predict(state))
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):

        if len(self.memory) < self.batch_size:
            return
        #Get one random sample and train the model with it
        x_batch, y_batch = [], []

        sample = random.sample(self.memory, self.batch_size)


        for state, action, reward, next_state, done in sample:
            y_target = self.model.predict(state)

            #Using target network here for next Q
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.target.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        self.update_target()
