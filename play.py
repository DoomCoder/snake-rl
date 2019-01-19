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
    state = env.reset()
    renderer=Renderer(env.game)

    done = False
    steps = 0

    while not done:
        time.sleep(0.1)
        renderer.render_frame()
        action = agent.act(state)
        next_state, _, done, _ = env.step(action)
        state = next_state
        steps += 1

    renderer.close_window()
    print(f"Snake lenght: {len(env.game.snake.body)}")
    print(f"Simulation ended after {steps} steps.")
    