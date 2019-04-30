import gym
import play
# from convDQN import ConvDQNAgent
from convTorchDQN import ConvTorchDQNAgent
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
sys.path.append('../snake_gym')

env = gym.make('snake-v0')
agent = ConvTorchDQNAgent(env.observation_space.shape, env.action_space.n, 4, )
agent.load("./models/", 'INSERT MODELS DIR HERE')

while True:
    play.watch_agent(agent)
