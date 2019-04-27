import tensorflow as tf
import numpy as np
import os
import copy
import sys

from utils import advantage_values
from utils import cumulative_rewards

class FNN():
    """ feed-forward neural network

    The neural networks for each of the agents and baselines are defined.
    One can choose between purely dense neural networks and
    a convolutional neural network.
    """
    def __init__(self, d_input, d_hidden_layers, d_output, learning_rate,
                 prev_layers=[], nn_type='dense', activation='tanh',
                 experiment_folder=os.getcwd(), name='fnn'):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.learning_rate = learning_rate
        self.nn_type = nn_type
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

            # Neural network with dense layers
            # e.g. for CartPole, 2D Mazes
            if nn_type == 'dense':
                self.d_hidden_layers = d_hidden_layers
                self.d_layers = self.d_hidden_layers + [self.d_output]
                self.n_layers = len(self.d_layers)
                self.layers_ = prev_layers

                if len(prev_layers) == 0:
                    # start neural network from scratch
                    shape = (None, *self.d_input) if type(self.d_input).__name__ == 'tuple' else (None, self.d_input)
                    self.observations = tf.placeholder(shape=shape, dtype=tf.float32)
                    outputs = self.observations
                    self.layers_.append(outputs)
                else:
                    # stack layers on top of a previous neural network
                    self.observations = prev_layers[0]
                    outputs = prev_layers[-1]

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

            # Neural network for Pong, Breakout.
            # The architecture follows the one used in
            # Mnih et. al. "Human-level control through deep reinforcement learning "
            elif nn_type == 'convolution':
                self.layers_ = prev_layers

                # input layer (84,84,1)
                shape = (None, *self.d_input) if type(self.d_input).__name__ == 'tuple' else (None, self.d_input)
                self.observations = tf.placeholder(shape=shape, dtype=tf.float32)
                outputs = self.observations
                self.layers_.append(outputs)

                outputs = tf.layers.conv2d(outputs,
                                           name="conv1",
                                           filters=32,
                                           kernel_size=8,
                                           kernel_initializer=kernel_initializer,
                                           strides=4,
                                           padding="valid",
                                           activation=self.activation)
                self.layers_.append(outputs)

                outputs = tf.layers.conv2d(outputs,
                                           name="conv2",
                                           filters=64,
                                           kernel_size=4,
                                           kernel_initializer=kernel_initializer,
                                           strides=2,
                                           padding="valid",
                                           activation=self.activation)
                self.layers_.append(outputs)

                outputs = tf.layers.conv2d(outputs,
                                           name="conv3",
                                           filters=64,
                                           kernel_size=3,
                                           kernel_initializer=kernel_initializer,
                                           strides=1,
                                           padding="valid",
                                           activation=self.activation)
                self.layers_.append(outputs)

                nh = np.prod([v.value for v in outputs.get_shape()[1:]])
                outputs = tf.reshape(outputs, [-1, nh])

                outputs = tf.layers.dense(outputs, 512,
                                          activation=self.activation,
                                          kernel_initializer=kernel_initializer,
                                          name="dense1")
                self.layers_.append(outputs)

                outputs = tf.layers.dense(outputs,
                                          units=self.d_output,
                                          activation=None,
                                          kernel_initializer=kernel_initializer,
                                          name="dense2")
                self.layers_.append(outputs)
                self.outputs = outputs

    def save(self):
        self.saver.save(self.session, self.checkpoint_prefix, global_step=self.global_step)
        return self.saver

    def load(self):
        fnames = os.listdir(self.experiment_folder)
        ckpt_names = [fname.split('.', 1)[0] for fname in fnames if self.nn_name in fname and '.data' in fname]
        global_steps = [int(name.split('-',-1)[1]) for name in ckpt_names]
        latest_ckpt_name = ckpt_names[np.argmax(global_steps)]
        latest_ckpt = os.path.join(self.experiment_folder, latest_ckpt_name)
        self.saver.restore(self.session, latest_ckpt)

class VPGAgent(FNN):
    """ Agent trained via REINFORCE (vanilla policy gradients)
        Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning
        Williams, 1992

        Sample trajectories and update the policy accordingly. On-policy algorithm.
    """
    def __init__(self, d_input, d_hidden_layers, d_output, learning_rate,
                 reward_discount_factor, nn_type, activation, baseline_cfg,
                 experiment_folder=os.getcwd(), name='vpg'):
        super().__init__(d_input, d_hidden_layers, d_output, learning_rate, [],
                         nn_type, activation, experiment_folder, name)
        self.reward_discount_factor = reward_discount_factor
        self.baseline = None

        self.baseline = None
        if 'advantage' in baseline_cfg.baseline:
            if 'shared' in baseline_cfg.baseline:
                # baseline sharing layers with policy except for the last layer
                self.baseline = StateValueFunction(self.layers_[-2].get_shape()[1],
                                                   baseline_cfg.d_hidden_layers,
                                                   1,
                                                   baseline_cfg.learning_rate,
                                                   reward_discount_factor,
                                                   baseline_cfg.gae_lamda,
                                                   self.layers_[:-1],
                                                   'dense',
                                                   activation,
                                                   experiment_folder, 'baseline')
            else:
                # baseline not sharing any layers with policy
                self.baseline = StateValueFunction(d_input,
                                                   baseline_cfg.d_hidden_layers,
                                                   1,
                                                   baseline_cfg.learning_rate,
                                                   reward_discount_factor,
                                                   baseline_cfg.gae_lamda,
                                                   [],
                                                   'dense',
                                                   activation,
                                                   experiment_folder, 'baseline')

        # policy
        outputs = self.layers_[-1]
        self.policy = tf.nn.softmax(outputs)
        self.log_policy = tf.nn.log_softmax(outputs)

        # log probabilities for taken actions
        self.actions = tf.placeholder(shape=(None, self.d_output), dtype=tf.float32)
        self.log_proba = tf.reduce_sum(self.actions * self.log_policy, axis=1)

        # loss & train operation
        self.crewards = tf.placeholder(shape=(None,), dtype=tf.float32)
        self.loss = - tf.reduce_mean(self.log_proba * self.crewards, axis=0)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # saver
        self.saver = tf.train.Saver()

        # initialise graph
        self.init_session()

    def init_session(self):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        # set baseline session
        if self.baseline is not None:
            self.baseline.init_session(self.saver, self.session)

    def action(self, obs):
        feed = {self.observations: obs}
        policy = self.session.run(self.policy, feed)[0] + 1e-6
        policy /= np.sum(policy)
        action = np.argmax(np.random.multinomial(1, policy))
        return action, policy

    def close(self):
        self.session.close()

    def update(self, obs_batch, action_batch, reward_batch, done_batch):
        # cumulative rewards
        creward_batch = cumulative_rewards(reward_batch, self.reward_discount_factor)

        # flatten out episodes in batch
        obs = np.squeeze(np.vstack(obs_batch).astype(np.float32))
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
        corresponding gradient is zero. On-policy algorithm.
    """
    def __init__(self, d_input, d_hidden_layers, d_output, learning_rate,
                 reward_discount_factor, clip_range, nn_type, activation, baseline_cfg,
                 experiment_folder=os.getcwd(), name='ppo'):
        super().__init__(d_input, d_hidden_layers, d_output, learning_rate, [], nn_type,
                         activation, experiment_folder, name)

        self.d_output = d_output
        self.learning_rate = learning_rate
        self.clip_range = clip_range
        self.reward_discount_factor = reward_discount_factor

        # dictionary to store one batch of sampled trajectories
        # as well as corresponding policies, advantages etc...
        self.sample = {}

        self.baseline = None
        if 'advantage' in baseline_cfg.baseline:
            if 'shared' in baseline_cfg.baseline:
                # baseline sharing layers with policy except for the last layer
                self.baseline = StateValueFunction(self.layers_[-2].get_shape()[1],
                                                   baseline_cfg.d_hidden_layers,
                                                   1,
                                                   baseline_cfg.learning_rate,
                                                   reward_discount_factor,
                                                   baseline_cfg.gae_lamda,
                                                   self.layers_[:-1],
                                                   'dense',
                                                   activation,
                                                   experiment_folder, 'baseline')
            else:
                # baseline not sharing any layers with policy
                self.baseline = StateValueFunction(d_input,
                                                   baseline_cfg.d_hidden_layers,
                                                   1,
                                                   baseline_cfg.learning_rate,
                                                   reward_discount_factor,
                                                   baseline_cfg.gae_lamda,
                                                   [],
                                                   'dense',
                                                   activation,
                                                   experiment_folder, 'baseline')
        # logits, policy, log_policy
        self.logits = self.layers_[-1]
        self.policy = tf.nn.softmax(self.logits)
        self.log_policy = tf.nn.log_softmax(self.logits)

        # sampled actions
        self.sampled_actions = tf.placeholder(shape=(None, self.d_output), dtype=tf.float32)

        # log probabilities of sampled actions
        # log p(a_t|s_t, theta) for given (s_t, a_t) pairs
        self.log_policy_of_actions = tf.reduce_sum(self.log_policy * self.sampled_actions, axis=1)

        # sampled log probabilities of actions
        self.sampled_log_policy_of_actions = tf.placeholder(shape=(None,), dtype=tf.float32)

        # sampled advantages
        self.sampled_advantages = tf.placeholder(shape=(None,), dtype=tf.float32)

        # ratio
        # r_t(theta) = p(a_t|s_t, theta) / p(a_t|s_t, theta_old)
        self.ratio = tf.exp(self.log_policy_of_actions - self.sampled_log_policy_of_actions)

        # clipped policy objective that should be maximised
        self.clipped_ratio = tf.clip_by_value(self.ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        self.policy_loss = - tf.reduce_mean(tf.minimum(self.ratio * self.sampled_advantages,
                                                       self.clipped_ratio * self.sampled_advantages), axis=0)

        # entropy loss (exploration bonus)
        self.entropy_loss = tf.reduce_mean(tf.reduce_sum(self.policy * self.log_policy, axis=1), axis=0)

        # complete loss
        self.loss = self.policy_loss  # + 0.01 * entropy_loss
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                               var_list = self.get_trainable_variables(self.nn_name))

        # saver
        self.saver = tf.train.Saver()

        # initialise graph
        self.init_session()

    def init_session(self):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        # set baseline session
        if self.baseline is not None:
            self.baseline.init_session(self.saver, self.session)

    def close(self):
        self.session.close()

    def get_trainable_variables(self, nn_name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=nn_name)

    def action(self, obs):
        """ return action and policy """
        feed = {self.observations: obs}
        policy = self.session.run(self.policy, feed)[0] + 1e-6
        policy /= np.sum(policy)
        action = np.argmax(np.random.multinomial(1, policy))
        return action, policy

    def update(self, obs_batch, action_batch, reward_batch, done_batch):
        """ one gradient step update """

        # cumulative rewards
        creward_batch = cumulative_rewards(reward_batch, self.reward_discount_factor)

        # flatten out episodes in batch
        obs = np.squeeze(np.vstack(obs_batch).astype(np.float32))
        actions = np.vstack(action_batch).astype(np.float32)
        crewards = np.hstack(creward_batch).astype(np.float32)

        # use advantage baseline
        advantages = crewards
        if self.baseline is not None:
            state_value_batch = []
            for i in range(len(obs_batch)):
                state_values = np.reshape(self.baseline.forward(np.vstack(obs_batch[i]).astype(np.float32)),[-1])
                state_value_batch.append(state_values)
            advantage_batch = advantage_values(obs_batch, reward_batch,
                                               done_batch, state_value_batch,
                                               self.baseline.reward_discount_factor,
                                               self.baseline.gae_lamda)
            advantages = np.hstack(advantage_batch).astype(np.float32)

        feed = {self.observations: obs,
                self.sampled_actions: actions,
                self.sampled_advantages: advantages}
        policy, log_policy, log_policy_of_actions = self.session.run([self.policy,
                                                                      self.log_policy,
                                                                      self.log_policy_of_actions], feed)
        # deep copy of sampled trajectories from old policy
        old_sample = copy.deepcopy(self.sample)

        # store current sample
        self.sample = {}
        self.sample['obs'] = obs
        self.sample['actions'] = actions
        self.sample['advantages'] = advantages
        self.sample['log_policy_of_actions'] = log_policy_of_actions

        # at start the old policy is equals the current policy
        if len(old_sample) == 0:
            old_sample = copy.deepcopy(self.sample)

        # train policy with old sample of trajectories
        feed = {self.observations: old_sample['obs'],
                self.sampled_actions: old_sample['actions'],
                self.sampled_advantages: old_sample['advantages'],
                self.sampled_log_policy_of_actions: old_sample['log_policy_of_actions']}
        loss, ratio, clipped_ratio, _, policy_loss, entropy_loss = self.session.run([self.loss,
                                                                                     self.ratio,
                                                                                     self.clipped_ratio,
                                                                                     self.train_op,
                                                                                     self.policy_loss,
                                                                                     self.entropy_loss], feed)
        self.global_step += 1

        # statistics
        stats = {}
        stats['crewards'] = crewards
        stats['obs'] = obs
        stats['baseline_value'] = np.mean(advantages)
        stats['ratio'] = np.mean(ratio)
        stats['clipped_ratio'] = np.mean(clipped_ratio)
        stats['n_clips'] = sum(diff > 10e-6 for diff in ratio - clipped_ratio)
        stats['policy_loss'] = policy_loss
        stats['entropy_loss'] = entropy_loss

        # monte carlo update of baseline
        if self.baseline is not None:
            baseline_loss, _ = self.baseline.mc_update(obs, crewards)
            stats['baseline_loss'] = baseline_loss
        return loss, _ , stats

class DQNAgent(FNN):
    """ Agent trained via deep q-learning
        Playing Atari with Deep Reinforcement Learning, Mnih et al.
        arXiv:1312.5602v1  [cs.LG]  19 Dec 2013
        http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf

        y_i = r_i + 𝛾 * max(Q(next_state, action; 𝜃_target))
        Loss: (y_i - Q(state, action; 𝜃))^2
        Every C step, 𝜃_target <- 𝜃
    """
    def __init__(self, d_input, d_hidden_layers, d_output, learning_rate, gamma,
                 nn_type, activation, experiment_folder=os.getcwd(), name='dqn'):
        super().__init__(d_input, d_hidden_layers, d_output, learning_rate, [], nn_type,
                         activation, experiment_folder, name)
        self.gamma = gamma
        self.d_input = d_input
        self.learning_rate = learning_rate

        # traget q-value network
        self.target_dqn = FNN(d_input, d_hidden_layers, d_output, learning_rate,
                              [], nn_type, activation, experiment_folder, 'target_dqn')

        # current q-value
        self.q = self.layers_[-1]

        # q-target value ph
        self.q_target = tf.placeholder(shape=(None, d_output), dtype=tf.float32)

        # loss
        self.loss = tf.losses.mean_squared_error(labels=self.q_target, predictions=self.q)

        # only train policy network
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # saver
        self.saver = tf.train.Saver()

        # initialise graph
        self.init_session()

    def init_session(self):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        # set trainable variables equal in both networks
        self.copy_trainable_variables(self.nn_name, self.target_dqn.nn_name)

    def close(self):
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
        feed = {self.observations: obs}

        # deterministic action
        q_value = self.session.run(self.q, feed)[0] + 1e-6
        action = np.argmax(q_value)

        return action, None

    def update(self, obs, actions, next_obs, rewards, done):
        """ one gradient step update """
        # current observation, reward, done flag: s_t, r_t, d_t
        # next observation: s_(t+1)

        # stack observations and squeeze
        obs = np.squeeze(np.vstack(obs).astype(np.float32))
        next_obs = np.squeeze(np.vstack(next_obs).astype(np.float32))
        done = np.array(done).astype(np.float32)
        actions = np.array(actions)
        rewards = np.array(rewards).astype(np.float32)

        # 1)  compute target q-value of next observation s_(t+1)
        # 2)  for current observations construct target q values
        # 2a) copy q-values for current observations
        # 2b) use the maximal target q-value of
        #     next observation s_(t+1) as 'best' action
        # 2c) update the q-values for the taken actions
        feed = {self.observations: obs, self.target_dqn.observations: next_obs}
        q_target, q_target_next_state = self.session.run([self.q, self.target_dqn.outputs], feed)
        q_target[np.arange(len(actions)), actions] = rewards + self.gamma * (1. - done) * np.max(q_target_next_state, axis=1)

        # train current policy network
        feed = {self.observations: obs, self.q_target: q_target}
        loss, _ = self.session.run([self.loss, self.train_op], feed)

        self.global_step += 1
        return loss, _

class StateValueFunction(FNN):
    """ State-Value Function trained via Monte-Carlo """
    def __init__(self, d_input, d_hidden_layers, d_output, learning_rate,
                 reward_discount_factor, gae_lamda,
                 prev_layers=[], nn_type='dense', activation='relu',
                 experiment_folder=os.getcwd(), name='state_value_function'):
        super().__init__(d_input, d_hidden_layers, d_output, learning_rate, prev_layers, nn_type,
                         activation, experiment_folder, name)
        self.reward_discount_factor = reward_discount_factor
        self.gae_lamda = gae_lamda
        self.nn_name = name

        self.outputs = self.layers_[-1]

        # loss
        self.crewards = tf.placeholder(shape=(None, 1), dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(labels=self.crewards, predictions=self.outputs)

        # only train baseline layer
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                               var_list=self.get_trainable_variables(self.nn_name))

    def forward(self, obs):
        feed = {self.observations: obs}
        outputs = self.session.run(self.outputs, feed)
        return outputs

    def init_session(self, saver=None, session=None):
        if session == None:
            self.saver = tf.train.Saver()
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
        else:
            self.session = session
            self.saver = saver

    def close(self):
        self.session.close()

    def get_trainable_variables(self, nn_name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=nn_name)

    def mc_update(self, obs, crewards):
        feed = {self.observations: obs, self.crewards: np.reshape(crewards,[-1,1])}
        loss, _ = self.session.run([self.loss, self.train_op], feed)
        self.global_step += 1
        return loss, _



