import gym

from simpleDQN import SimpleDQNAgent
from convDQN import ConvDQNAgent


N_EPISODES = 10**7

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
        while not done:
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
            agent.remember(state, action, reward, next_state, done)
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
            agent.save("./SNEK-dqn.h5")
