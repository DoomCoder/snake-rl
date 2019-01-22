import gym
from dqn import DQNAgent
import time
import sys

from reporter import Reporter

sys.path.append('../snake_gym')
from gym_snake.envs.snake import Renderer


class AgentInput():
    def __init__(self, agent: DQNAgent):
        self.agent=agent

    def get_input(self):
        self.agent.act()


def watch_agent(agent: DQNAgent):
    env = gym.make('snake-v0')
    env.__init__(human_mode=True)
    observation = env.reset()
    renderer=Renderer(env.game)
    try:
        done = False
        steps = 0
        state = agent.get_last_observations(observation)
        while not done:
            time.sleep(0.1)
            renderer.render_frame()
            action = agent.act(state)
            next_observation, _, done, _ = env.step(action)
            state = agent.get_last_observations(next_observation)
            steps += 1
    finally:
        renderer.close_window()
    print(f"Snake lenght: {len(env.game.snake.body)}")
    print(f"Simulation ended after {steps} steps.")


def collect_stats(agent: DQNAgent, n_games=1000):
    MAX_STEPS = 500
    lenghts = []
    looped = 0
    for i in range(1, n_games+1):
        env = gym.make('snake-v0')
        # env.__init__(human_mode=False)
        observation = env.reset()
        done = False
        steps = 0
        agent.epsilon = 0.0
        state = agent.get_last_observations(observation)
        while not done and steps < MAX_STEPS:
            action = agent.act(state)
            next_observation, _, done, _ = env.step(action)
            state = agent.get_last_observations(next_observation)
            steps += 1

        if steps == MAX_STEPS:
            looped += 1
        else:
            lenghts.append(len(env.game.snake.body))

        if i % (n_games//10) == 0:
            print(f"Avg len: {sum(lenghts) / len(lenghts):.2f}, looped {looped}/{i}")
