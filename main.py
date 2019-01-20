import gym
from convDQN import ConvDQNAgent

NUM_LAST_FRAMES = 3
BATCH_SIZE = 64
N_EPISODES = 10**5
EXPLORATION_PHASE_SIZE = 0.5
SAVE_FREQ = 1000

if __name__ == "__main__":
    env = gym.make('snake-v0')
    agent = ConvDQNAgent(env.observation_space.shape, env.action_space.n, NUM_LAST_FRAMES)
    agent.train(env, BATCH_SIZE, N_EPISODES, EXPLORATION_PHASE_SIZE, SAVE_FREQ)
