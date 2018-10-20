# rl-flappy-bird
****************
Keras implementation of a DQN agent for solving OpenAI's Flappy Bird environment

# Installing things and such
----------------------------
To install the environment, PLE and PyGame:

### PyGame

On OSX:
```bash
brew install sdl sdl_ttf sdl_image sdl_mixer portmidi  # brew or use equivalent means
conda install -c tlatorre pygame=1.9.2 # using Anaconda
```
On Ubuntu 14.04:
```bash
apt-get install -y python-pygame
```
More configurations and installation details on: http://www.pygame.org/wiki/GettingStarted#Pygame%20Installation


### PLE
```bash
git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
cd PyGame-Learning-Environment/
pip install -e .
```
and finally in order to use PLE as an OpenAI gym environment

### Gym PLE
```bash
pip install gym_ple
```

### Keras
Given that the agent is implemented using Keras, having it installed may proove useful, to do so:
```bash
sudo pip install keras
```
As a backend for Keras I highly recommend gpu-enabled tensorflow


# Using the agent
----------------
To train the agent simply run
```bash
python flappy_bird.py
```
