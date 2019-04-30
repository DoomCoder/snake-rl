import gym
import play
from convAC import ConvACAgent
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
sys.path.append('../snake_gym')

env = gym.make('snake-v0')
agent = ConvACAgent(env.observation_space.shape, env.action_space.n, 4, )
agent.load("./models/", 20000, nameQ='SNEK-pg-2-Q-99000-episodes.h5', nameP='SNEK-pg-2-P-99000-episodes.h5')

while True:
    play.watch_agent(agent)
