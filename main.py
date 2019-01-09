import gym

from simpleDQN import SimpleDQNAgent

N_EPISODES_SIMPLE = 1000

if __name__ == "__main__":
    env = gym.make('snake-v0')
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
    action_size = env.action_space.n
    agent = SimpleDQNAgent(state_size, action_size)
    # agent.load("./SNEK-dqn.h5")
    done = False
    batch_size = 32

    for e in range(N_EPISODES_SIMPLE):
        state = env.reset()
        done = False
        # env.renderer.close_window()
        # print(state)
        # exit()
        # state = np.reshape(state, [1, state_size])
        time = 0
        reward_sum = 0
        while not done:
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            reward_sum += reward
            # next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            time += 1
            if done:
                print("episode: {}/{}, time: {}, score: {}, e: {:.2}"
                      .format(e, N_EPISODES_SIMPLE, time, reward_sum, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 100 == 0:
            agent.save("./SNEK-dqn.h5")
