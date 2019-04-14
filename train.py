import os
import sys

from utils import get_experiment_name
from utils import dotdict
from cfg import get_cfg
from model import Model

if __name__ == '__main__':

    # create experiment folder
    experiments_folder = os.path.join(os.getcwd(), 'results')
    experiment_name, experiment_folder = get_experiment_name(experiments_folder)

    # specify environment and agent
    env_name = 'CartPole-v0'
    agent_name = 'dqn'

    # load default config parameters
    cfg_env, cfg_agent = get_cfg(experiment_folder, env_name, agent_name)
    cfg = cfg_env
    cfg.update(cfg_agent)
    cfg = dotdict(cfg)

    # modify default config parameters
    cfg.n_batches = 100
    cfg.n_episodes = 100
    cfg.log_step = 10

    # load and train model
    model = Model(cfg)
    model.train_model()





