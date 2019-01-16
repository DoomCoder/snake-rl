import gym
from reporter import Reporter
from simpleDQN import SimpleDQNAgent
from convDQN import ConvDQNAgent


MAX_EPISODES = 10 ** 7
STATS_N_EPISODES = 100  # stats calculated on this many last episodes
STATS_FREQ = 50  # print stats every STATS_FREQ number of episodes

if __name__ == "__main__":
    env = gym.make('snake-v0')
    action_size = env.action_space.n
    agent = SimpleDQNAgent(env.observation_space.shape, action_size)
    # agent = ConvDQNAgent(env.observation_space.shape, action_size)
    # agent.load("./SNEK-dqn.h5")
    done = False
    batch_size = 64
    reporter = Reporter(STATS_N_EPISODES, STATS_FREQ, MAX_EPISODES)

    for e in range(MAX_EPISODES):
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
            if done:
                reporter.remember(steps, len(env.game.snake.body), reward_sum)
                if reporter.wants_to_report():
                    print(reporter.get_report_str())
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if e % 100 == 0:
            agent.save("./SNEK-dqn.h5")
