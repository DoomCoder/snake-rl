import play
from convDQN import ConvDQNAgent
import sys

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4' 

sys.path.append('../snake_gym')

agent = ConvDQNAgent((1,10,10), 4)
#agent.load("./SNEK-dqn.h5")

play.watch_agent(agent)
