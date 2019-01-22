import os
import gym
from convDQN import ConvDQNAgent

MODELS_DIR = './models'

NUM_LAST_FRAMES = 3
BATCH_SIZE = 64
N_EPISODES = 10**5 + 10**4
EXPLORATION_PHASE_SIZE = 0.95
REPORT_FREQ = 100
SAVE_FREQ = 5000

if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    env = gym.make('snake-v0')
    agent = ConvDQNAgent(env.observation_space.shape, env.action_space.n, NUM_LAST_FRAMES)
    agent.train(env, BATCH_SIZE, N_EPISODES, EXPLORATION_PHASE_SIZE, REPORT_FREQ, SAVE_FREQ, MODELS_DIR)
