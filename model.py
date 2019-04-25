import tensorflow as tf
import numpy as np
import os
import collections
import gym
import random
import pandas as pd
import time
import datetime
import sys

from envs import AtariGame
from utils import dotdict
from utils import one_hot
from utils import json_to_data
from utils import data_to_json
from utils import plot_df

# choose between tf, tf eager and pt
USE_TF = True
USE_TFE = False
USE_PT = False

if USE_TF:
    from nn_tf import VPGAgent
    from nn_tf import PPOAgent
    from nn_tf import DQNAgent
elif USE_TFE:
    from nn_tfe import VPGAgent
    from nn_tfe import PPOAgent
    from nn_tfe import DQNAgent
elif USE_PT:
    from nn_pt import VPGAgent
    from nn_pt import PPOAgent
    from nn_pt import DQNAgent

class Model(object):
    """ Interactions of RL agents in various environments

    This class implements the interactions of the PPO, VPG, DQN with
    the environments Atari Pong, Breakout and OpenAI CartPole.

    The code is largely agnostic to the used deep learning framework
    tensorflow, tensorflow in eager mode and pytorch

    Training progress can be visualised via tensorboard and models
    can be saved/loaded to/from the filesystem.

    Default hyperparameters for each agent and environment are stored in the
    configuration file cfg.py.
    """

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
        self.n_steps = cfg.n_steps

        # non-hyperparameters
        self.episode_number, self.step_number = 0, 0
        self.agent_loss = []
        self.returns_test, self.returns, self.episode_lengths = [], [], []
        self.baseline = None
        self.t_start = -1
        self.speed = []

        # set seeds
        tf.set_random_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        # breakout dependent hyperparameters
        if 'Breakout' in self.env_name or 'Pong' in self.env_name:
            self.env = AtariGame(self.env_name, cfg.seed)
            self.n_actions = self.env.n_actions
            self.d_observation = self.env.d_observation
            self.nn_type = 'convolution'

        # cartpole dependent hyperparameters
        elif 'CartPole' in self.env_name:
            self.env = gym.make(self.env_name)
            self.env.seed(cfg.seed)
            self.n_actions = self.env.action_space.n
            self.d_observation = self.env.observation_space.shape[0]
            self.nn_type = 'dense'

        # input dimension for the agent
        self.d_input_agent = self.d_observation

        if 'vpg' in self.model_name or 'ppo' in self.model_name:
            # baseline dependent hyperparameters
            baseline_cfg = dotdict({'baseline': cfg.baseline})
            if cfg.baseline != 'None':
                self.baseline_value, self.baseline_loss = [], []
                baseline_cfg = dotdict({'baseline': cfg.baseline,
                                        'learning_rate': cfg.baseline_learning_rate,
                                        'gae_lamda': cfg.gae_lamda,
                                        'd_hidden_layers': cfg.baseline_d_hidden_layers})

        # policy-gradient dependent hyperparameters
        if 'vpg' in self.model_name:
            self.batch_size = cfg.batch_size
            self.batch_number = 0
            self.crewards, self.baseline_value, self.baseline_loss = [], [], []
            self.agent = VPGAgent(self.d_input_agent,
                                  cfg.agent_d_hidden_layers,
                                  self.n_actions,
                                  cfg.agent_learning_rate,
                                  cfg.reward_discount_factor,
                                  self.nn_type,
                                  cfg.activation,
                                  baseline_cfg,
                                  cfg.experiment_folder, 'vpg')

            self.baseline = self.agent.baseline

        # ppo dependent hyperparameters
        elif 'ppo' in self.model_name:
            self.batch_size = cfg.batch_size
            self.batch_number = 0
            self.crewards, self.entropy_loss, self.baseline_value, self.baseline_loss = [], [], [], []
            self.policy_loss, self.ratio, self.clipped_ratio, self.n_clips = [], [], [], []

            self.agent = PPOAgent(self.d_input_agent,
                                  cfg.agent_d_hidden_layers,
                                  self.n_actions,
                                  cfg.agent_learning_rate,
                                  cfg.reward_discount_factor,
                                  cfg.clip_range,
                                  self.nn_type,
                                  cfg.activation,
                                  baseline_cfg,
                                  cfg.experiment_folder, 'ppo')

            self.baseline = self.agent.baseline

        # dqn dependent hyperparameters
        elif 'dqn' in self.model_name:
            self.agent_buffer = collections.deque(maxlen=cfg.agent_buffer_size) # replay buffer
            self.epsilon = cfg.epsilon_start
            self.agent = DQNAgent(self.d_input_agent,
                                cfg.agent_d_hidden_layers,
                                self.n_actions,
                                cfg.agent_learning_rate,
                                cfg.reward_discount_factor,
                                self.nn_type,
                                cfg.activation,
                                cfg.experiment_folder, 'dqn')

        # random-search dependent hyperparameters
        elif 'rs' in self.model_name:
            self.agent = None

        # initialise summary writer and save cfg parameters
        if not load:
            # short description
            description = self.env_name

            if USE_TF:
                # use an own suffix to distinguish from ray tf events
                self.writer = tf.summary.FileWriter(self.cfg.experiment_folder, filename_suffix = 'tf.events.model')

            if USE_TFE:
                self.summary_writer = tf.contrib.summary.create_file_writer(cfg.experiment_folder, flush_millis=10000)
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
             while self.step_number < self.n_steps:
                self.train_one_batch_with_vpg()
        if 'dqn' in self.model_name:
            while self.step_number < self.n_steps:
                self.train_one_episode_with_dqn()
        if 'rs' in self.model_name:
            while self.step_number < self.n_steps:
                self.train_one_episode_with_rs()
        if 'ppo' in self.model_name:
            while self.step_number < self.n_steps:
                self.train_one_batch_with_ppo()
        self.save_model()
        self.close()

    def encode_obs(self, obs):
        """ choose encoding of the observations """
        obs = np.expand_dims(obs, axis=0)
        obs = obs.astype(np.float32)
        return obs

    def action(self, obs, epsilon=0.):
        """ agent's action """
        if ('rs' in self.model_name) or (np.random.rand(1) < epsilon):
            action = np.random.randint(0, self.n_actions)
        else:
            obs = obs.astype(np.float32)
            action, _ = self.agent.action(obs)
        return action

    def add_to_buffer(self, obs, action, next_obs, reward, done):
        """ fill replay buffer

        One observation has 84*84*4 pixels of float32 equals 84*84*4*4 Bytes
        which is 0.11MB. We store 2 observations each time i.e. 0.22MB.

        Replay Buffer of size 10000 requires 2.26GB.

        """
        if self.agent is not None:
                self.agent_buffer.append([obs, action, next_obs, reward, done])

    def dqn_update(self):
        """ one deep q-learning update step """
        size = self.cfg.agent_buffer_batch_size
        history_batch = random.sample(list(self.agent_buffer), min(size, len(self.agent_buffer)))
        obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = zip(*history_batch)
        loss, _ = self.agent.update(obs_batch, action_batch, next_obs_batch, reward_batch, done_batch)
        self.agent_loss.append(loss)

    def vpg_update(self, obs_batch, action_batch, reward_batch, done_batch):
        """ one vanilla policy gradient update step """
        loss, grads, stats = self.agent.update(obs_batch, action_batch, reward_batch, done_batch)
        self.agent_loss.append(loss)
        self.crewards.append(np.mean(stats['crewards']))
        if self.baseline is not None:
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
        self.clipped_ratio.append(stats['clipped_ratio'])
        self.n_clips.append(stats['n_clips'])
        if self.baseline is not None:
            self.baseline_value.append(stats['baseline_value'])
            self.baseline_loss.append(stats['baseline_loss'])

    def train_one_episode_with_rs(self):
        """ random search (training) """
        self.episode_number += 1
        self.global_step += 1
        self.returns.append(0.)
        obs, done = self.env.reset(), False

        episode_length = 0
        while not done:
            episode_length += 1
            self.step_number += 1
            action = self.action(obs)
            next_obs, reward, done, _ = self.env.step(action)
            self.returns[-1] += reward
            obs = next_obs

        if self.step_number % self.log_step == 0 and self.step_number <= self.n_steps:
            self.print_logs()
            self.write_summary()

        self.episode_lengths.append(episode_length)

    def train_one_episode_with_dqn(self):
        """ deep q-learning training """
        self.episode_number += 1
        self.global_step += 1

        obs, done = self.env.reset(), False
        obs = self.encode_obs(obs)

        episode_length, sum_of_rewards = 0, 0
        while not done:
            episode_length += 1
            self.step_number += 1

            # update q-value target network
            if self.step_number % self.cfg.update_target_network_freq == 0:
                self.agent.update_target_network()

            if self.step_number > self.cfg.agent_buffer_start_size:
                # epsilon-greedy action
                action = self.action(obs, self.epsilon)
            else:
                # random action to fill the replay buffer
                action = self.action(obs, 1.)

            # make one step in the environment
            next_obs, reward, done, _ = self.env.step(action)
            next_obs = self.encode_obs(next_obs)

            # extrinsic reward
            sum_of_rewards += reward

            # add transition to buffer
            self.add_to_buffer(obs, action, next_obs, reward, done)

            if self.verbose:
                print(self.episode_number, self.step_number, action, reward, done, self.env.env.unwrapped.ale.lives())
                self.env.render()

            if self.step_number > self.cfg.agent_buffer_start_size:
                # update agent
                if self.step_number % self.cfg.update_freq == 0 :
                    self.dqn_update()

                # decrease epsilon for exploration
                if self.epsilon > self.cfg.epsilon_final:
                    self.epsilon -= (self.cfg.epsilon_start - self.cfg.epsilon_final)/self.cfg.epsilon_step_range

            obs = next_obs

            if self.step_number % self.log_step == 0 and self.step_number <= self.n_steps:
                self.print_logs()
                self.write_summary()

        self.returns.append(sum_of_rewards)
        self.episode_lengths.append(episode_length)

    def train_one_batch_with_vpg(self):
        """ vanilla policy gradient training """
        self.batch_number += 1
        obs_batch, action_batch, done_batch, reward_batch = [], [], [], []

        for _ in range(self.batch_size):
            self.episode_number += 1
            self.global_step += 1

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
                next_obs, reward, done, info = self.env.step(action)
                next_obs = self.encode_obs(next_obs)
                done_batch[-1].append(done)

                # extrinsic reward
                reward_batch[-1].append(reward)

                obs = next_obs

                if self.step_number % self.log_step == 0 and self.step_number <= self.n_steps:
                    self.print_logs()
                    self.write_summary()

            self.returns.append(sum(reward_batch[-1]))
            self.episode_lengths.append(len(obs_batch[-1]))

        # update agent
        self.vpg_update(obs_batch, action_batch, reward_batch, done_batch)

    def train_one_batch_with_ppo(self):
        """ proximal policy optimisation training """
        self.batch_number += 1
        obs_batch, action_batch, done_batch, reward_batch = [], [], [], []

        for _ in range(self.batch_size):
            self.episode_number += 1
            self.global_step += 1

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

                if self.verbose:
                    self.env.render()
                    print(self.episode_number, self.step_number, action, reward, done)
                    #time.sleep(0.01)

                # extrinsic reward
                reward_batch[-1].append(reward)

                obs = next_obs

                if self.step_number % self.log_step == 0 and self.step_number <= self.n_steps:
                    self.print_logs()
                    self.write_summary()

            self.returns.append(sum(reward_batch[-1]))
            self.episode_lengths.append(len(obs_batch[-1]))

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
        if self.agent is not None: self.agent.save()
        if self.baseline is not None: self.baseline.save()

    def load_model(self):
        self.epsilon = 0.
        if self.agent is not None: self.agent.load()
        if self.baseline is not None: self.baseline.load()

    def print_logs(self):
        """ print log information """

        # compute speed
        if self.t_start>0:
            self.speed.append(self.log_step / (time.time() - self.t_start))
        self.t_start = time.time()

        m = 5
        print_str = 'ep {} st {} {} e.ret.tr {:.2f} ep.ln {:.0f} sp {:.1f} '.format(
            self.episode_number,
            self.step_number,
            self.model_name,
            np.mean(self.returns[-m:] if len(self.returns)>0 else 0.),
            np.mean(self.episode_lengths[-m:] if len(self.episode_lengths)>0 else 0.),
            self.speed[-1] if len(self.speed)>0 else 0.
        )

        if self.agent is not None:
            print_str += 'ag.ls {:.2f} '.format(np.mean(self.agent_loss[-m:]) if len(self.agent_loss)>0 else 0.)
        if self.baseline is not None:
            print_str += 'bs.ls {:.2f} '.format(np.mean(self.baseline_loss[-m:]) if len(self.baseline_loss)>0 else 0.)
        if 'dqn' in self.model_name:
            print_str += 'eps {:.2f} '.format(self.epsilon)

        print(print_str)

    def write_summary(self):
        """ write tensorflow summaries """
        m = 5
        summary = {'data/ext_return': np.mean(self.returns[-m:]) if len(self.returns)>0 else 0.,
                   'data/episode_length': np.mean(self.episode_lengths[-m:]) if len(self.episode_lengths)>0 else 0.,
                   'data/speed': self.speed[-1] if len(self.speed)>0 else 0.}

        if self.agent is not None:
            summary['data/agent_loss'] = np.mean(self.agent_loss[-m:]) if len(self.agent_loss)>0 else 0.

        if 'dqn' in self.model_name:
            summary['data/epsilon'] = self.epsilon

        if 'vpg' in self.model_name or 'ppo' in self.model_name:
            summary['data/crewards'] = np.mean(self.crewards[-m:]) if len(self.crewards)>0 else 0.

        if 'ppo' in self.model_name:
            summary['data/entropy_loss'] = np.mean(self.entropy_loss[-m:]) if len(self.baseline_loss)>0 else 0.
            summary['data/policy_loss'] = np.mean(self.policy_loss[-m:]) if len(self.policy_loss)>0 else 0.
            summary['data/ratio'] = np.mean(self.ratio[-m:]) if len(self.ratio)>0 else 0.
            summary['data/clipped_ratio'] = np.mean(self.clipped_ratio[-m:]) if len(self.clipped_ratio)>0 else 0.
            summary['data/n_clips'] = np.mean(self.n_clips[-m:]) if len(self.n_clips)>0 else 0.

        if self.baseline is not None:
            summary['data/baseline_loss'] = np.mean(self.baseline_loss[-m:]) if len(self.baseline_loss)>0 else 0.
            summary['data/baseline_value'] = np.mean(self.baseline_value[-m:]) if len(self.baseline_value)>0 else 0.

        for key in summary.keys():
            if USE_TF:
                summary_value = tf.Summary.Value(tag=key, simple_value=summary[key])
                self.writer.add_summary(tf.Summary(value=[summary_value]), global_step=self.step_number)

            elif USE_TFE:
                with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar(key, summary[key], step=self.step_number)
        if USE_TF:
            self.writer.flush()

    def close(self):
        """ close summary filewriters """
        if USE_TF:
            self.writer.close()
        elif USE_TFE:
            self.summary_writer.close()

        if self.agent is not None:
            self.agent.close()
        if self.baseline is not None:
            self.baseline.close()


