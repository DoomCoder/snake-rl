import gym
from reporter import Reporter
from simpleDQN import SimpleDQNAgent
from convDQN import ConvDQNAgent
import numpy as np
from collections import deque

NUM_LAST_FRAMES = 3
N_EPISODES = 10**3
# the percentage of the training process at which the exploration rate should reach its minimum
EXPLORATION_PHASE_SIZE = 0.5


MAX_EPISODES = 10 ** 7
STATS_N_EPISODES = 100  # stats calculated on this many last episodes
STATS_FREQ = 50  # print stats every STATS_FREQ number of episodes
def get_last_frames(history):
    states = np.array(history)
    states = states[-NUM_LAST_FRAMES:, 0, :, :]
    return states


def fill_empty_frames(states):
    if states.shape[0] < NUM_LAST_FRAMES:
        duplicated_states = np.tile(states[0], (NUM_LAST_FRAMES - states.shape[0], 1, 1))
        states = np.append(duplicated_states, states, axis=0)

    return states


if __name__ == "__main__":
    env = gym.make('snake-v0')
    action_size = env.action_space.n
    # agent = SimpleDQNAgent(env.observation_space.shape, action_size)
    agent = ConvDQNAgent(env.observation_space.shape, action_size)
    agent.load("./SNEK-dqn600k.h5")
    done = False
    batch_size = 64
    reporter = Reporter(STATS_N_EPISODES, STATS_FREQ, MAX_EPISODES)

    # agent.epsilon_decay = ((agent.epsilon - agent.epsilon_min) / (N_EPISODES * EXPLORATION_PHASE_SIZE))

    samples_with_fruits = 0
    for e in range(MAX_EPISODES):
    # calc constant epsilon dacay
        state = env.reset()
        done = False
        # env.renderer.close_window()
        # exit()
        steps = 0
        reward_sum = 0
        state_history = deque(maxlen=4)
        while not done:
            state_history.append(state)
            # env.render()
            states = get_last_frames(state_history)
            states = fill_empty_frames(states)
            action = agent.act(states)
            next_state, reward, done, _ = env.step(action)
            if reward > 0:
                samples_with_fruits += 1

            if done:
                reward = -1

            reward_sum += reward
            next_states = np.append(states[1:], next_state, axis=0)

            agent.remember(states, action, reward, next_states, done)
            state = next_state
            steps += 1
            if done:
                reporter.remember(steps, len(env.game.snake.body), reward_sum)
                if reporter.wants_to_report():
                    print(reporter.get_report_str())
                break

            if len(agent.memory) > batch_size and (i % batch_size == 0):
                agent.replay(batch_size)

        # update epsilon decay every episode (should be in agent's train method
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon -= agent.epsilon_decay

        if e % 1000 == 0:
            agent.save("./SNEK-dqn600k.h5")
