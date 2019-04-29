import gym
import play
# from convDQN import ConvDQNAgent
from convDDPG import ConvDDPGAgent
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
sys.path.append('../snake_gym')

env = gym.make('snake-v0')
# agent = ConvDQNAgent(env.observation_space.shape, env.action_space.n, 3)
agent = ConvDDPGAgent(env.observation_space.shape, env.action_space.n, 4,)
# agent.load("./models/SNEK-dqnt-195000-episodes.h5")
agent.load("./models/", 20000, nameQ='SNEK-pg-2-Q-9000-episodes.h5', nameP='SNEK-pg-2-P-9000-episodes.h5')
# agent.load("./models/...")

while True:
    play.watch_agent(agent)
