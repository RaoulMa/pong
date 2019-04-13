import tensorflow as tf
import numpy as np
import os
import sys

from utils import advantage_values
from utils import cumulative_rewards

class FNN():
    """ feed-forward neural network """
    def __init__(self, d_input, d_hidden_layers, d_output, learning_rate,
                 prev_layers=[], activation='tanh', experiment_folder=os.getcwd(), name='fnn'):
        super().__init__()
        self.d_input = d_input
        self.d_hidden_layers = d_hidden_layers
        self.d_output = d_output
        self.learning_rate = learning_rate
        self.nn_name = name
        self.global_step = 0

        self.experiment_folder = experiment_folder
        self.checkpoint_prefix = os.path.join(self.experiment_folder, name + '_ckpt')

        bias_initializer = tf.initializers.random_normal(stddev=0.1)
        kernel_initializer = tf.initializers.orthogonal()

        if activation == 'relu':
            self.activation = tf.nn.relu
        else:
            self.activation = tf.nn.tanh

        self.d_layers = d_hidden_layers + [d_output]
        self.n_layers = len(self.d_layers)
        self.layers_ = prev_layers

        self.observations = tf.placeholder(shape=(None, d_input), dtype=tf.float32)
        outputs = self.observations
        for i in range(self.n_layers):
            if i < (self.n_layers - 1):
                outputs = tf.layers.dense(outputs,
                                          units=self.d_layers[i],
                                          activation=self.activation,
                                          kernel_initializer=kernel_initializer)
            else:
                outputs = tf.layers.dense(outputs,
                                          units=self.d_layers[i],
                                          activation=None,
                                          kernel_initializer=kernel_initializer)
            self.layers_.append(outputs)

    def save(self):
        pass

    def load(self):
        pass

class VPGAgent(FNN):
    """ Agent trained via REINFORCE (vanilla policy gradients)
        Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning
        Williams, 1992

        Sample trajectories and update the policy accordingly. On-policy algorithm.
    """
    def __init__(self, d_input, d_hidden_layers, d_output, learning_rate, reward_discount_factor,
                 baseline=None, activation='tanh', experiment_folder=os.getcwd(), name='vpg'):
        super().__init__(d_input, d_hidden_layers, d_output, learning_rate, [], activation, experiment_folder,
                         name)
        self.reward_discount_factor = reward_discount_factor
        self.baseline = baseline

        outputs = self.layers_[-1]

        # policy
        self.policy = tf.nn.softmax(outputs)
        self.log_policy = tf.nn.log_softmax(outputs)

        # log probabilities for taken actions
        self.actions = tf.placeholder(shape=(None, self.d_output), dtype=tf.float32)
        self.log_proba = tf.reduce_sum(self.actions * self.log_policy, axis=1)

        # loss & train operation
        self.crewards = tf.placeholder(shape=(None,), dtype=tf.float32)
        self.loss = - tf.reduce_mean(self.log_proba * self.crewards, axis=0)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # initialise graph
        self.init_session()

    def init_session(self):
        config_tf = tf.ConfigProto(device_count={'GPU': 0})
        self.session = tf.Session(config=config_tf)
        self.session.run(tf.global_variables_initializer())

    def action(self, obs):
        policy = self.session.run(self.policy, {self.observations: obs.reshape(1,-1)})[0]
        action = np.argmax(np.random.multinomial(1, policy))
        return action, policy

    def close_session(self):
        self.session.close()

    def update(self, obs_batch, action_batch, reward_batch, done_batch):
        # cumulative rewards
        creward_batch = cumulative_rewards(reward_batch, self.reward_discount_factor)

        # flatten out episodes in batch
        obs = np.vstack(obs_batch).astype(np.float32)
        actions = np.vstack(action_batch).astype(np.float32)
        crewards = np.hstack(creward_batch).astype(np.float32)
        stats = {'obs': obs, 'crewards': crewards}

        # use advantage baseline
        if self.baseline!=None:
            state_value_batch = []
            for i in range(len(obs_batch)):
                state_values = np.reshape(self.baseline.forward(np.vstack(obs_batch[i]).astype(np.float32)),[-1])
                state_value_batch.append(state_values)
            advantage_batch = advantage_values(obs_batch, reward_batch,
                                               done_batch, state_value_batch,
                                               self.baseline.reward_discount_factor,
                                               self.baseline.gae_lamda)
            advantages = np.hstack(advantage_batch).astype(np.float32)
            stats['baseline_value'] = np.mean(advantages)
            crewards = advantages

        feed = {self.observations: obs, self.actions: actions, self.crewards: crewards}
        loss, _ = self.session.run([self.loss, self.train_op], feed)
        self.global_step += 1

        # monte carlo update of baseline
        if self.baseline!=None:
            obs = stats['obs']
            crewards = stats['crewards']
            baseline_loss, _ = self.baseline.mc_update(obs, crewards)
            stats['baseline_loss'] = baseline_loss

        return loss, _ , stats

class PPOAgent(FNN):
    """ PPO Agent
        Proximal Policy Optimization Algorithms, Schulman et al. 2017
        arXiv:1707.06347v2  [cs.LG]  28 Aug 2017

        In PPO we keep the old and current policies. We sample trajectories
        from the old policy and update the current policy wrt to the clipped
        surrogate objective. If the ratio for one (s_t, a_t) pair is outside
        the allowed region, the objective gets clipped, which means that the
        corresponding gradient is zero.
    """
    def __init__(self, d_input, d_hidden_layers, d_output, learning_rate,
                 reward_discount_factor=0.99, clip_range=0.2, baseline=None,
                 activation='tanh', experiment_folder=os.getcwd(), name='ppo'):
        super().__init__(d_input, d_hidden_layers, d_output, learning_rate, [],
                         activation, experiment_folder, name)
        self.clip_range = clip_range
        self.reward_discount_factor = reward_discount_factor
        self.baseline = baseline

        # old policy weights
        self.old_trainable_weights = self.get_trainable_weights()

class DQNAgent(FNN):
    """ Agent trained via deep q-learning
        Playing Atari with Deep Reinforcement Learning, Mnih et al.
        arXiv:1312.5602v1  [cs.LG]  19 Dec 2013
    """
    def __init__(self, d_input, d_hidden_layers, d_output, learning_rate, gamma, activation='tanh',
                 experiment_folder=os.getcwd(), name='dqn'):
        super().__init__(d_input, d_hidden_layers, d_output, learning_rate, [], activation, experiment_folder, name)
        self.gamma = gamma

class StateValueFunction(FNN):
    """ State-Value Function
        trained via Monte-Carlo or Temporal-Difference learning (td0)
    """
    def __init__(self, d_input, d_hidden_layers, d_output, learning_rate,
                 reward_discount_factor, gae_lamda,
                 prev_layers=[], activation='tanh', experiment_folder=os.getcwd(),
                 name='state_value_function'):
        super().__init__(d_input, d_hidden_layers, d_output, learning_rate, prev_layers,
                         activation, experiment_folder, name)
        self.reward_discount_factor = reward_discount_factor
        self.gae_lamda = gae_lamda

        self.outputs = self.layers_[-1]
        self.crewards = tf.placeholder(shape=(None, 1), dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(labels=self.crewards, predictions=self.outputs)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        self.init_session()

    def forward(self, obs):
        feed = {self.observations: obs}
        outputs = self.session.run(self.outputs, feed)
        return outputs

    def init_session(self):
        config_tf = tf.ConfigProto(device_count={'GPU': 0})
        self.session = tf.Session(config=config_tf)
        self.session.run(tf.global_variables_initializer())

    def mc_update(self, obs, crewards):
        feed = {self.observations: obs, self.crewards: np.reshape(crewards,[-1,1])}
        loss, _ = self.session.run([self.loss, self.train_op], feed)
        self.global_step += 1
        return loss, _



