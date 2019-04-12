import tensorflow as tf
import numpy as np
import os
import collections
import gym
import random
import copy
import pandas as pd
import sys

from envs import make_env
from utils import dotdict
from utils import one_hot
from utils import json_to_data
from utils import data_to_json
from utils import plot_df

# use tf
if False:
    from nn_tf import VPGAgent
    from nn_tf import PPOAgent
    from nn_tf import DQNAgent
    from nn_tf import StateValueFunction
# use tf eager
elif False:
    from nn_tfe import VPGAgent
    from nn_tfe import PPOAgent
    from nn_tfe import DQNAgent
    from nn_tfe import StateValueFunction
# use pytorch
elif True:
    from nn_pt import VPGAgent
    from nn_pt import PPOAgent
    from nn_pt import DQNAgent
    from nn_pt import StateValueFunction

class Model():

    @classmethod
    def load(cls, experiment_folder):
        """ load a model and then start constructor """
        fpath = os.path.join(experiment_folder, 'cfg.json')
        cfg = dotdict(json_to_data(fpath))
        # update path in case the files were moved
        cfg.experiment_folder = experiment_folder
        return cls(cfg, load=True)

    def __init__(self, cfg, load=False):
        """ constructor """
        self.cfg = cfg

        # generic hyperparameters
        self.model_name = cfg.model_name
        self.env_name = cfg.env_name
        self.log_step = cfg.log_step
        self.verbose = cfg.verbose
        self.global_step = cfg.global_step
        self.observation_encoding = cfg.observation_encoding

        # non-hyperparameters
        self.episode_number, self.step_number = 0, 0
        self.agent_loss, self.predictor_loss = [], []
        self.returns_test, self.returns = [], []
        self.agent_buffer = collections.deque(maxlen=cfg.agent_buffer_size)

        # set seeds
        tf.set_random_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        # maze dependent hyperparameters
        if 'maze' in self.env_name:
            self.env = make_env(self.env_name, cfg.env_max_steps, cfg.env_reward)
            self.n_actions = self.env.n_actions
            self.d_observation = self.env.d_observation
            self.n_observations = self.env.n_observations
            self.layout_shape = self.env.layout.shape
            self.states_visited = set() # distinct states visited by the agent during training

            if self.observation_encoding == 'normalise':
                self.obs_mean = [(self.layout_shape[0] - 1.)/2., (self.layout_shape[1] - 1.)/2.]
                self.obs_std = self.obs_mean

        # cartpole dependent hyperparameters
        if 'CartPole' in self.env_name:
            self.env = gym.make(self.env_name)
            self.env.seed(cfg.seed)
            self.n_actions = self.env.action_space.n
            self.d_observation = self.env.observation_space.shape[0]

        # input dimension for the agent
        self.d_input_agent = self.d_observation
        if self.observation_encoding == 'one_hot':
            self.d_input_agent = self.n_observations

        # baseline dependent hyperparameters
        self.baseline = None
        if 'vpg' in self.model_name or 'ppo' in self.model_name:
            if cfg.baseline == 'advantage':
                self.baseline_value, self.baseline_loss = [], []
                self.baseline = StateValueFunction(self.d_input_agent,
                                                   cfg.baseline_d_hidden_layers,
                                                   1,
                                                   cfg.baseline_learning_rate,
                                                   cfg.reward_discount_factor,
                                                   cfg.gae_lamda,
                                                   [],
                                                   cfg.activation,
                                                   cfg.experiment_folder, 'baseline')

        # policy-gradient dependent hyperparameters
        if 'vpg' in self.model_name:
            self.n_batches = cfg.n_batches
            self.batch_size = cfg.batch_size
            self.batch_number = 0
            self.crewards = []
            self.agent = VPGAgent(self.d_input_agent,
                                  cfg.agent_d_hidden_layers,
                                  self.n_actions,
                                  cfg.agent_learning_rate,
                                  cfg.reward_discount_factor,
                                  self.baseline,
                                  cfg.activation,
                                  cfg.experiment_folder, 'vpg')

        # ppo dependent hyperparameters
        if 'ppo' in self.model_name:
            self.n_batches = cfg.n_batches
            self.batch_size = cfg.batch_size
            self.batch_number = 0
            self.crewards, self.entropy_loss, self.baseline_value, self.baseline_loss = [], [], [], []
            self.policy_loss, self.ratio = [], []
            self.agent = PPOAgent(self.d_input_agent,
                                  cfg.agent_d_hidden_layers,
                                  self.n_actions,
                                  cfg.agent_learning_rate,
                                  cfg.reward_discount_factor,
                                  cfg.clip_range,
                                  self.baseline,
                                  cfg.activation,
                                  cfg.experiment_folder, 'ppo')

        # dqn dependent hyperparameters
        if 'dqn' in self.model_name:
            self.epsilon = cfg.epsilon
            self.epsilon_discount_factor = cfg.epsilon_discount_factor
            self.n_episodes = cfg.n_episodes
            self.update_target_network_freq = cfg.update_target_network_freq
            self.agent = DQNAgent(self.d_input_agent,
                                cfg.agent_d_hidden_layers,
                                self.n_actions,
                                cfg.agent_learning_rate,
                                cfg.reward_discount_factor,
                                cfg.activation,
                                cfg.experiment_folder, 'dqn')
            self.agent_target = copy.deepcopy(self.agent)

        # random-search dependent hyperparameters
        if 'rs' in self.model_name:
            self.n_episodes = cfg.n_episodes
            self.agent = None

        # initialise summary writer and save cfg parameters
        if not load:
            self.summary_writer = tf.contrib.summary.create_file_writer(cfg.experiment_folder, flush_millis=10000)

            # short description
            description = self.env_name
            if 'maze' in self.model_name:
                description += '_maxsteps_' + str(cfg.env_max_steps) + '_reward_' + str(cfg.env_reward)
            with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('data/' + description, 0, step=0)

            # save cfg parameters
            fpath = os.path.join(cfg.experiment_folder, 'cfg.json')
            data_to_json(cfg, fpath)
        else:
            self.load_model()

    def train_model(self):
        """ train the model for n epsiodes/batches """
        if 'vpg' in self.model_name:
            for _ in range(self.n_batches):
                self.train_one_batch_with_vpg()
        if 'dqn' in self.model_name:
            for _ in range(self.n_episodes):
                self.train_one_episode_with_dqn()
        if 'rs' in self.model_name:
            for _ in range(self.n_episodes):
                self.train_one_episode_with_rs()
        if 'ppo' in self.model_name:
            for _ in range(self.n_batches):
                self.train_one_batch_with_ppo()
        self.save_model()
        self.close()

    def encode_obs(self, obs):
        """ choose encoding of the observations """
        if 'maze' in self.env_name:
            if self.observation_encoding == 'one_hot':
                n_classes = self.layout_shape[0] * self.layout_shape[1]
                value = obs[0]*7 + obs[1]
                obs = one_hot(value, n_classes)
            elif self.observation_encoding == 'normalise':
                obs = (obs - self.obs_mean) / self.obs_std
        obs = obs.astype(np.float32)
        return obs

    def action(self, obs, epsilon=0.):
        """ agent's action """
        if ('rs' in self.model_name) or (np.random.rand(1) < epsilon):
            action = np.random.randint(0, self.n_actions)
            policy = 'random'
        else:
            obs = obs.astype(np.float32)
            obs = np.reshape(obs, [1, -1])
            action, policy = self.agent.action(obs)
        return action

    def add_to_buffer(self, obs, action, next_obs, reward, done):
        if self.agent!=None:
            self.agent_buffer.append([obs, action, next_obs, reward, done])

    def dqn_update(self):
        """ one deep q-learning update step """
        size = self.cfg.agent_buffer_batch_size // 2
        history_batch = random.sample(list(self.agent_buffer), min(size, len(self.agent_buffer))) \
                        + list(self.agent_buffer)[-size:]
        obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = zip(*history_batch)
        loss, _ = self.agent.update(obs_batch, action_batch, next_obs_batch, reward_batch, done_batch)
        self.agent_loss.append(loss)

    def vpg_update(self, obs_batch, action_batch, reward_batch, done_batch):
        """ one vanilla policy gradient update step """
        loss, grads, stats = self.agent.update(obs_batch, action_batch, reward_batch, done_batch)
        self.agent_loss.append(loss)
        self.crewards.append(np.mean(stats['crewards']))
        if self.baseline!=None:
            self.baseline_value.append(stats['baseline_value'])
            self.baseline_loss.append(stats['baseline_loss'])

    def ppo_update(self, obs_batch, action_batch, reward_batch, done_batch):
        """ one proximal policy optimisation update step """
        loss, grads, stats = self.agent.update(obs_batch, action_batch, reward_batch, done_batch)
        self.agent_loss.append(loss)
        self.crewards.append(np.mean(stats['crewards']))
        self.entropy_loss.append(stats['entropy_loss'])
        self.policy_loss.append(stats['policy_loss'])
        self.ratio.append(stats['ratio'])
        if self.baseline!=None:
            self.baseline_value.append(stats['baseline_value'])
            self.baseline_loss.append(stats['baseline_loss'])

    def train_one_episode_with_rs(self):
        """ random search (training) """
        self.episode_number += 1
        self.global_step += 1
        self.returns.append(0.)
        obs, done = self.env.reset(), False

        while not done:
            self.step_number += 1
            action = self.action(obs)
            next_obs, reward, done, _ = self.env.step(action)
            self.returns[-1] += reward
            obs = next_obs

        if 'maze' in self.env_name:
            self.states_visited.update(self.env.visited)

        if self.episode_number % self.log_step == 0:
            self.print_logs()
            self.write_summary()

    def train_one_episode_with_dqn(self):
        """ deep q-learning training """
        self.episode_number += 1
        self.global_step += 1
        reward_batch = []

        obs, done = self.env.reset(), False
        obs = self.encode_obs(obs)
        self.returns.append(0.)

        while not done:
            self.step_number += 1

            # update q-value target network
            if self.step_number % self.update_target_network_freq == 0:
                self.agent_target.set_trainable_weights(self.agent.get_trainable_weights())

            # epsilon-greedy action
            action = self.action(obs, self.epsilon)

            # make one step in the environment
            next_obs, reward, done, _ = self.env.step(action)
            next_obs = self.encode_obs(next_obs)

            # extrinsic reward
            reward += self.cfg.time_reward
            reward_batch.append(reward)
            self.returns[-1] += reward

            # add transition to buffer
            self.add_to_buffer(obs, action, next_obs, reward, done)

            # update agent
            self.dqn_update()

            # decrease epsilon for exploration
            self.epsilon *= self.epsilon_discount_factor

            obs = next_obs

        if 'maze' in self.env_name:
            self.states_visited.update(self.env.visited)

        if self.episode_number % self.log_step == 0:
            self.print_logs()
            self.write_summary()

    def train_one_batch_with_vpg(self):
        """ vanilla policy gradient training """
        self.batch_number += 1
        obs_batch, action_batch, done_batch = [], [], []
        reward_batch = []

        for _ in range(self.batch_size):
            self.episode_number += 1
            self.global_step += 1
            self.returns.append(0.)

            obs_batch.append([])
            action_batch.append([])
            reward_batch.append([])
            done_batch.append([])

            obs, done = self.env.reset(), False
            obs = self.encode_obs(obs)

            while not done:
                self.step_number += 1

                # store observation
                obs_batch[-1].append(obs)

                # take and store epsilon-greedy action
                action = self.action(obs)
                action_one_hot = one_hot(action, self.n_actions)
                action_batch[-1].append(action_one_hot)

                # make one step in the environment
                next_obs, reward, done, _ = self.env.step(action)
                next_obs = self.encode_obs(next_obs)
                done_batch[-1].append(done)

                # extrinsic reward
                reward += self.cfg.time_reward
                reward_batch[-1].append(reward)
                self.returns[-1] += reward

                # add transition to buffer
                self.add_to_buffer(obs, action, next_obs, reward, done)

                obs = next_obs

            if 'maze' in self.env_name:
                self.states_visited.update(self.env.visited)

            if self.episode_number % self.log_step == 0:
                self.print_logs()
                self.write_summary()

        # update agent
        self.vpg_update(obs_batch, action_batch, reward_batch, done_batch)

    def train_one_batch_with_ppo(self):
        """ proximal policy optimisation training """
        self.batch_number += 1
        obs_batch, action_batch, done_batch = [], [], []
        reward_batch = []

        for _ in range(self.batch_size):
            self.episode_number += 1
            self.global_step += 1
            self.returns.append(0.)

            obs_batch.append([])
            action_batch.append([])
            reward_batch.append([])
            done_batch.append([])

            obs, done = self.env.reset(), False
            obs = self.encode_obs(obs)

            while not done:
                self.step_number += 1

                # store observation
                obs_batch[-1].append(obs)

                # take and store epsilon-greedy action
                action = self.action(obs)
                action_one_hot = one_hot(action, self.n_actions)
                action_batch[-1].append(action_one_hot)

                # make one step in the environment
                next_obs, reward, done, _ = self.env.step(action)
                next_obs = self.encode_obs(next_obs)
                done_batch[-1].append(done)

                # extrinsic reward
                reward += self.cfg.time_reward
                reward_batch[-1].append(reward)
                self.returns[-1] += reward

                # add transition to buffer
                self.add_to_buffer(obs, action, next_obs, reward, done)

                obs = next_obs

            if 'maze' in self.env_name:
                self.states_visited.update(self.env.visited)

            if self.episode_number % self.log_step == 0:
                self.print_logs()
                self.write_summary()

        # update agent
        self.ppo_update(obs_batch, action_batch, reward_batch, done_batch)

    def simulate(self):
        """ run one episode with current policy """
        ext_return, n_steps = 0., 0.
        obs, done = self.env.reset(), False
        obs_list = [obs]
        obs = self.encode_obs(obs)
        while not done:
            action = self.action(obs)
            obs, ext_reward, done, _ = self.env.step(action)
            obs_list.append(obs)
            obs = self.encode_obs(obs)
            ext_return += ext_reward
            n_steps += 1
        return ext_return, n_steps, obs_list

    def save_model(self):
        if self.agent!=None: self.agent.save()
        if self.baseline!=None: self.baseline.save()

    def load_model(self):
        self.epsilon = 0.
        if self.agent!=None: self.agent.load()
        if self.baseline!=None: self.baseline.load()

    def print_logs(self):
        """ print log information """
        stats = self.simulate()
        self.returns_test.append(stats[0])

        m = 20
        print_str = 'ep {} st {} {} e.ret.tr {:.2f} e.ret.ts {:.2f} '.format(
            self.episode_number,
            self.step_number,
            self.model_name,
            np.mean(self.returns[-m:]),
            np.mean(self.returns_test[-1]))

        if self.agent!=None:
            print_str += 'ag.ls {:.2f} '.format(np.mean(self.agent_loss[-m:]))
        if self.baseline!=None:
            print_str += 'bs.ls {:.2f} '.format(np.mean(self.baseline_loss[-m:]))
        if 'dqn' in self.model_name:
            print_str += 'eps {:.2f} '.format(self.epsilon)
        if 'maze' in self.env_name:
            print_str += 'vis {:.2f} '.format(len(self.states_visited))

        print(print_str)

    def save_logs(self):
        """ write log files """
        fpath = os.path.join(self.cfg.experiment_folder, 'ext_returns_test')
        np.save(fpath, np.array(self.returns_test))

        fpath = os.path.join(self.cfg.experiment_folder, 'ext_returns')
        np.save(fpath, np.array(self.returns))

        if True:
            # optional plots
            y_values = np.array(self.returns_test)
            x_values = np.arange(0, y_values.shape[0] * self.log_step, self.log_step)

            df = pd.DataFrame()
            df['episode'] = x_values
            df['ext_returns_test'] = y_values

            fpath = os.path.join(self.cfg.experiment_folder, 'ext_returns_test.png')
            plot_df(df, 'episode', 'ext_returns_test', None, '', fpath)

    def write_summary(self):
        """ write tensorflow summaries """
        m = 20
        summary = {'data/ext_return': np.mean(self.returns[-m:])}

        if self.agent!=None:
            summary['data/agent_loss'] = np.mean(self.agent_loss[-m:])

        if 'vpg' in self.model_name or 'ppo' in self.model_name:
            summary['data/crewards'] = np.mean(self.crewards[-m:])

        if 'ppo' in self.model_name:
            summary['data/entropy_loss'] = np.mean(self.entropy_loss[-m:])
            summary['data/policy_loss'] = np.mean(self.policy_loss[-m:])
            summary['data/ratio'] = np.mean(self.ratio[-m:])

        if self.baseline!=None:
            summary['data/baseline_loss'] = np.mean(self.baseline_loss[-m:])
            summary['data/baseline_value'] = np.mean(self.baseline_value[-m:])

        if 'maze' in self.env_name:
            summary['data/n_states_visited'] = len(self.states_visited)

        with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            for key in summary.keys():
                tf.contrib.summary.scalar(key, summary[key], step=self.global_step)

    def close(self):
        self.summary_writer.close()
