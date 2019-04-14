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

        with tf.variable_scope(self.nn_name):

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

            self.outputs = outputs

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

        self.d_output = d_output
        self.learning_rate = learning_rate
        self.clip_range = clip_range
        self.reward_discount_factor = reward_discount_factor
        self.baseline = baseline

        # network that keeps the old policy weights
        self.old_fnn = FNN(d_input, d_hidden_layers, d_output, learning_rate,
                           [], activation, experiment_folder, 'old_fnn')

        # current policy
        self.logits = self.layers_[-1]
        self.policy = tf.nn.softmax(self.logits)
        self.log_policy = tf.nn.log_softmax(self.logits)

        # old policy which is not updated by gradient descent
        self.old_logits = tf.stop_gradient(self.old_fnn.layers_[-1])
        self.old_policy = tf.nn.softmax(self.old_logits)
        self.log_old_policy = tf.nn.log_softmax(self.old_logits)

        # placeholder for taken actions
        self.actions = tf.placeholder(shape=(None, self.d_output), dtype=tf.float32)

        # current policy: log probabilities for taken actions
        # current log policy: log p(a_t|s_t, theta) for given (s_t, a_t) pairs
        self.log_proba = tf.reduce_sum(self.actions * self.log_policy, axis=1)

        # old policy: log probabilities for taken actions
        # old log policy: log p(a_t|s_t, theta_old) for given (s_t, a_t) pairs
        self.log_old_policy_for_actions = tf.reduce_sum(self.log_old_policy * self.actions, axis=1)

        # old policy placholder: log probabilities for taken actions
        self.log_old_policy_for_actions_ph = tf.placeholder(shape=(None,), dtype=tf.float32)

        # current log policy: log p(a_t|s_t, theta) for given (s_t, a_t) pairs
        self.log_policy_for_actions = tf.reduce_sum(self.log_policy * self.actions, axis=1)

        # ratio: r_t(theta) = p(a_t|s_t, theta) / p(a_t|s_t, theta_old)
        self.ratio = tf.exp(self.log_policy_for_actions - self.log_old_policy_for_actions_ph)

        # clipped policy objective that should be maximised
        self.crewards = tf.placeholder(shape=(None,), dtype=tf.float32)
        self.clipped_ratio = tf.clip_by_value(self.ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        self.policy_loss = - tf.reduce_mean(tf.minimum(self.ratio * self.crewards,
                                                       self.clipped_ratio * self.crewards), axis=0)

        # entropy loss (exploration bonus)
        self.entropy_loss = tf.reduce_mean(tf.reduce_sum(self.policy * tf.log(self.policy), axis=1), axis=0)

        # complete loss
        self.loss = self.policy_loss  # + 0.01 * entropy_loss
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # initialise graph
        self.init_session()

    def init_session(self):
        config_tf = tf.ConfigProto(device_count={'GPU': 0})
        self.session = tf.Session(config=config_tf)
        self.session.run(tf.global_variables_initializer())
        # set trainable variables equal in both networks
        self.copy_trainable_variables(self.nn_name, self.old_fnn.nn_name)

    def close_session(self):
        self.session.close()

    def get_trainable_variables(self, nn_name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=nn_name)

    def copy_trainable_variables(self, src_name, dest_name):
        """ copy weights from source to destination network """
        src_vars = self.get_trainable_variables(src_name)
        dest_vars = self.get_trainable_variables(dest_name)
        op_list = []
        for src_var, dest_var in zip(src_vars, dest_vars):
            op_list.append(dest_var.assign(src_var.value()))
        self.session.run(op_list)

    def action(self, obs):
        """ return action and policy """
        # note we take actions according to the old policy
        feed = {self.old_fnn.observations: obs.reshape(1,-1)}
        policy = self.session.run(self.old_policy, feed)[0]
        action = np.argmax(np.random.multinomial(1, policy))
        return action, policy

    def update(self, obs_batch, action_batch, reward_batch, done_batch):
        """ one gradient step update """
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

        feed = {self.observations: obs, self.old_fnn.observations: obs,
                self.actions: actions, self.crewards: crewards}

        # calculate log old policy for taken actions
        log_old_policy_for_actions = self.session.run(self.log_old_policy_for_actions, feed)
        feed[self.log_old_policy_for_actions_ph] = log_old_policy_for_actions

        # update old policy network to current policy network
        self.copy_trainable_variables(self.nn_name, self.old_fnn.nn_name)

        # train current policy network
        loss, ratio, clipped_ratio, _, policy_loss, entropy_loss = self.session.run([self.loss, self.ratio,
                                                                                     self.clipped_ratio,
                                                                                     self.train_op,
                                                                                     self.policy_loss,
                                                                                     self.entropy_loss], feed)
        self.global_step += 1
        stats['ratio'] = np.mean(ratio)
        stats['clipped_ratio'] = np.mean(clipped_ratio)
        stats['n_clips'] = sum(diff > 10e-6 for diff in ratio - clipped_ratio)
        stats['policy_loss'] = policy_loss
        stats['entropy_loss'] = entropy_loss

        # monte carlo update of baseline
        if self.baseline!=None:
            obs = stats['obs']
            crewards = stats['crewards']
            baseline_loss, _ = self.baseline.mc_update(obs, crewards)
            stats['baseline_loss'] = baseline_loss

        return loss, _ , stats

class DQNAgent(FNN):
    """ Agent trained via deep q-learning
        Playing Atari with Deep Reinforcement Learning, Mnih et al.
        arXiv:1312.5602v1  [cs.LG]  19 Dec 2013
    """
    def __init__(self, d_input, d_hidden_layers, d_output, learning_rate, gamma, activation='tanh',
                 experiment_folder=os.getcwd(), name='dqn'):
        super().__init__(d_input, d_hidden_layers, d_output, learning_rate, [], activation, experiment_folder, name)
        self.gamma = gamma
        self.d_input = d_input
        self.learning_rate = learning_rate

        # traget q-value network
        self.target_dqn = FNN(d_input, d_hidden_layers, d_output, learning_rate,
                              [], activation, experiment_folder, 'target_dqn')

        # target q-value
        self.q_target = self.target_dqn.layers_[-1]

        # current q-value
        self.q = self.layers_[-1]

        # q-target value ph
        self.q_target = tf.placeholder(shape=(None, d_output), dtype=tf.float32)

        # loss
        self.loss = tf.losses.mean_squared_error(labels=self.q_target, predictions=self.q)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # initialise graph
        self.init_session()

    def init_session(self):
        config_tf = tf.ConfigProto(device_count={'GPU': 0})
        self.session = tf.Session(config=config_tf)
        self.session.run(tf.global_variables_initializer())
        # set trainable variables equal in both networks
        self.copy_trainable_variables(self.nn_name, self.target_dqn.nn_name)

    def close_session(self):
        self.session.close()

    def get_trainable_variables(self, nn_name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=nn_name)

    def copy_trainable_variables(self, src_name, dest_name):
        """ copy weights from source to destination network """
        src_vars = self.get_trainable_variables(src_name)
        dest_vars = self.get_trainable_variables(dest_name)
        op_list = []
        for src_var, dest_var in zip(src_vars, dest_vars):
            op_list.append(dest_var.assign(src_var.value()))
        self.session.run(op_list)

    def update_target_network(self):
        # set trainable variables equal in both networks
        self.copy_trainable_variables(self.nn_name, self.target_dqn.nn_name)

    def action(self, obs):
        """ return action and policy """
        # note we take actions according to the old policy
        feed = {self.observations: obs.reshape(1,-1)}
        # deterministic action
        # logits = self.session.run(self.q, feed)[0]
        # action = np.argmax(logits)
        # probabilistic action
        # add 1e-6 offset to prevent numeric underflow
        policy = self.session.run(tf.nn.softmax(self.q), feed)[0] + 1e-6
        policy /= np.sum(policy)
        action = np.argmax(np.random.multinomial(1, policy))
        return action, policy

    def update(self, obs, actions, next_obs, rewards, done):
        """ one gradient step update """
        # current observation, reward, done flag: s_t, r_t, d_t
        # next observation: s_(t+1)

        # target q-value of next observation s_(t+1)
        feed = {self.target_dqn.observations: next_obs}
        q_target_next_state = self.session.run(self.target_dqn.outputs, feed)

        # maximal target q-value of next observation s_(t+1) for 'best' action
        q_target_next_state_max = np.max(q_target_next_state, axis=1)

        # q-value of current observation s_t
        feed = {self.observations: obs}
        q = self.session.run(self.q, feed)

        # for given pair (s_t, a_t) construct target
        q_target = q
        for x, y in enumerate(actions):
            q_target[x, y] = rewards[x] + self.gamma * (1. - done[x]) * q_target_next_state_max[x]

        # train current policy network
        feed = {self.observations: obs, self.q_target: q_target}
        loss, _ = self.session.run([self.loss, self.train_op], feed)

        self.global_step += 1
        return loss, _

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



