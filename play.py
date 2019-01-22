import gym
from dqn import DQNAgent
import time
import sys

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
        agent.epsilon = 0
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
    print(f"Snake length: {len(env.game.snake.body)}")
    print(f"Simulation ended after {steps} steps.")
    