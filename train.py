import os
import argparse
import sys

from utils import get_experiment_name
from utils import dotdict
from cfg import get_cfg
from model import Model

if __name__ == '__main__':

    # create experiment folder
    experiments_folder = os.path.join(os.getcwd(), 'results')
    experiment_name, experiment_folder = get_experiment_name(experiments_folder)

    # specify environment
    env_name = 'four_rooms_maze'
    env_name = 'CartPole-v0'
    env_name = 'BreakoutNoFrameskip-v4'
    env_name = 'Pong-v0'

    # specify agent
    agent_name = 'dqn'
    agent_name = 'vpg'
    agent_name = 'ppo'

    # load default config parameters
    cfg_env, cfg_agent = get_cfg(experiment_folder, env_name, agent_name)
    cfg = cfg_env
    cfg.update(cfg_agent)
    cfg = dotdict(cfg)

    # modify default config parameters
    cfg.n_episodes = 10
    cfg.batch_size = 1
    cfg.log_step = 1
    cfg.verbose = True

    # load and train model
    model = Model(cfg)
    model.train_model()





