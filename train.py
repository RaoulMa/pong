import os
import ray
from ray import tune
import argparse
import sys

from utils import dotdict
from utils import get_experiment_name
from utils import tf_events_to_csv
from utils import subdir_paths
from utils import plot_hp_sensitivities
from utils import data_to_json

from cfg import get_cfg
from model import Model

class RayModel(tune.Trainable):
    """ ray model for hyperparameter search"""

    def _setup(self, config):
        """ initialise model """
        cfg = dotdict(config['cfg'])

        # each model is saved in a subdirectory of experiment_folder
        cfg.experiment_folder = os.getcwd()
        self.model = Model(cfg)

        self.step_number = 0

    def _train(self):
        """ one train step """
        self.step_number = self.model.step_number

        while (self.model.step_number <= self.model.n_steps
               and self.model.step_number <= (self.step_number + self.model.log_step)):
            if 'dqn' in self.model.model_name:
                self.model.train_one_episode_with_dqn()
            elif 'rs' in self.model.model_name:
                self.model.train_one_episode_with_rs()
            elif 'ppo' in self.model.model_name:
                self.model.train_one_batch_with_ppo()
            elif 'vpg' in self.model.model_name:
                self.model.train_one_batch_with_vpg()

        return {'timesteps_this_iter': 1, 'mean_loss': - self.model.returns[-1] if len(self.model.returns)>0 else 0.}

    def _stop(self):
        self.model.save_model()
        self.model.close()

    def _save(self, checkpoint_dir):
        pass

    def _restore(self, checkpoint_filepath, summary_directory=None):
        print('restore')
        pass

if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser(description='Hyperparameter Search')
    parser.add_argument('--cpu', default=1, type=int, help='number of CPUs that should be used')
    parser.add_argument('--gpu', default=1, type=int, help='number of GPUs that should be used')
    args = parser.parse_args()

    # create experiment folder
    experiments_folder = os.path.join(os.getcwd(), 'results')
    experiment_name, experiment_folder = get_experiment_name(experiments_folder)

    # specify environment
    env_name = 'BreakoutNoFrameskip-v4'
    env_name = 'four_rooms_maze'
    env_name = 'CartPole-v0'
    env_name = 'Pong-v0'

    # specify agent
    agent_names  = [#'dqn',
                    #'vpg',
                    'ppo',
                    ]

    cfgs, cfg_spec = [], {}

    for agent_name in agent_names:
        # get optimal/default hyperparameters
        cfg_env, cfg_agent = get_cfg(experiment_folder, env_name, agent_name)
        cfg = cfg_env
        cfg.update(cfg_agent)

        # make hyperparameter changes from default ones
        cfg['n_steps'] = 1000000      # total number of training steps
        cfg['batch_size'] = 4         # batch size in terms of episodes
        cfg['log_step'] = 1000        # in terms of step numbers

        # choose several seeds
        for i in range(1):
            cfg['seed'] = i
            cfgs.append(cfg.copy())

        cfg_spec[agent_name] = cfg_agent

    cfg_spec['env'] = cfg_env
    fpath = os.path.join(experiment_folder, 'cfg_spec.json')
    data_to_json(cfg_spec, fpath)

    cfgs = tune.grid_search(cfgs)
    tune.register_trainable('Model', RayModel)
    train_spec = {
        'run': 'Model',
        'trial_resources': {'cpu': args.cpu, 'gpu': args.gpu},
        'stop': {'timesteps_total': cfg['n_steps']//cfg['log_step']},
        'local_dir': experiments_folder,
        'num_samples': 1,
        'config': {
            'cfg': dict(cfgs)
        },
        'checkpoint_at_end': False
    }
    fpath = os.path.join(experiment_folder, 'train_spec.json')
    data_to_json(train_spec, fpath)

    # run experiments
    ray.init(temp_dir='~/tmp/ray/')
    tune.run_experiments({experiment_name: train_spec})

    # tf events to csv
    for dpath in subdir_paths(experiment_folder):
        tf_events_to_csv(dpath)

    # create plot
    plot_hp_sensitivities(experiment_folder, 'ext_return', 'model_name', {}, True, 1, False)




