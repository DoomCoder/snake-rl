import gym
import play
from convTorchDQN import ConvTorchDQNAgent
from convDQN import ConvDQNAgent
from convAC import ConvACAgent
import argparse
import sys
import os

MODELS = {
    'tDQN': {'agent': ConvTorchDQNAgent, 'model': None, 'description': "Deep Q-learning agent in pytorch"},
    'tAC': {'agent': ConvACAgent, 'model': None, 'description': "Actor-Critic agent in pytorch"},
    'kDQN': {'agent': ConvDQNAgent, 'model': None, 'description': "Deep Q-learning agent in keras"}
}


def main(args):
    print(MODELS[args.agent]['description'])
    sleep(3)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
    sys.path.append('../snake_gym')

    env = gym.make('snake-v0')
    agent_class = MODELS[args.agent]['agent']
    if args.model is not None:
        models_path = os.path.expanduser(args.model)
    else:
        models_path = MODELS[args.agent]['model']

    if models_path is None:
        print("Please provide model's path")

    agent = agent_class(env.observation_space.shape, env.action_space.n, 4, )
    agent.load(models_path)

    while True:
        play.watch_agent(agent)

    play.collect_stats(agent)


in __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch agents play Snake")
    parser.add_argument('-a','--agent', choices = MODELS.keys(), default='tAC', help='Pick agent to watch')
    parser.add_argument('--model', type='str', help="Path to agent's models", default=None)
    args  =parser.parse_args()
    main(args)