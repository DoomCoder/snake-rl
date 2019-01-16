import gym

from simpleDQN import SimpleDQNAgent
from convDQN import ConvDQNAgent
import numpy as np
from collections import deque

NUM_LAST_FRAMES = 5
N_EPISODES = 10**7


def get_last_frames(history):
    states = np.array(history)
    states = states[:, 0, :, :]
    return states


def fill_empty_frames(states):
    if states.shape[0] < NUM_LAST_FRAMES - 1:
        empty_states = np.zeros_like(state[0, :, :])
        empty_states = np.tile(empty_states, (NUM_LAST_FRAMES - 1 - states.shape[0], 1, 1))
        states = np.append(empty_states, states, axis=0)

    return states


if __name__ == "__main__":
    env = gym.make('snake-v0')
    action_size = env.action_space.n
    # agent = SimpleDQNAgent(env.observation_space.shape, action_size)
    agent = ConvDQNAgent(env.observation_space.shape, action_size)
    # agent.load("./SNEK-dqn.h5")
    done = False
    batch_size = 128
    i = 0

    for e in range(N_EPISODES):
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
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            if done:
                reward = -100
            elif reward:
                reward = 100
            else:
                reward = -1

            reward_sum += reward

            last_states = get_last_frames(state_history)
            last_states = fill_empty_frames(last_states)
            states = np.append(last_states, state, axis=0)
            next_states = np.append(states[1:], next_state, axis=0)

            agent.remember(states, action, reward, next_states, done)
            state = next_state
            steps += 1
            i += 1
            if done:
                print("episode: {}/{}, steps: {}, score: {}, e: {:.2}"
                      .format(e, N_EPISODES, steps, reward_sum, agent.epsilon))
                break
            if len(agent.memory) > batch_size and (i % batch_size == 0):
                agent.replay(batch_size)
        if e % 100 == 0:
            agent.save("./SNEK-dqn2137.h5")
