import gym
import play
from convDQN import ConvDQNAgent
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
sys.path.append('../snake_gym')

env = gym.make('snake-v0')
agent = ConvDQNAgent(env.observation_space.shape, env.action_space.n, 3)
agent.load("./SNEK-dqn.h5")

while True:
    play.watch_agent(agent)
