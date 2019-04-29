import os
import gym
from convDDPG import ConvDDPGAgent

MODELS_DIR = './models'

NUM_LAST_FRAMES = 4
BATCH_SIZE = 32
N_EPISODES = 10**5
EXPLORATION_PHASE_SIZE = 0.8
REPORT_FREQ = 100
TARGET_UPDATE_FREQ = 1000
SAVE_FREQ = 1000

if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    env = gym.make('snake-v0')
    agent = ConvDDPGAgent(env.observation_space.shape, env.action_space.n, NUM_LAST_FRAMES,)
    # agent.load("./models/", 20000, nameQ='SNEK-pg-2-Q-4000-episodes.h5', nameP='SNEK-pg-2-P-4000-episodes.h5')
    agent.train(env, BATCH_SIZE, N_EPISODES, EXPLORATION_PHASE_SIZE, REPORT_FREQ, SAVE_FREQ, MODELS_DIR)