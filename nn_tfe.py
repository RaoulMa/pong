import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
import sys

from utils import advantage_values
from utils import cumulative_rewards

# using tf in eager mode
tfe.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)

class FNN(tf.keras.Model):
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
        for i in range(len(prev_layers), self.n_layers):
            if i < (self.n_layers - 1):
                self.layers_.append(tf.layers.Dense(self.d_layers[i],
                                    activation=self.activation,
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer))
            else:
                self.layers_.append(tf.layers.Dense(self.d_layers[i],
                                    activation=None,
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # initialise weights
        logits = tf.constant(np.zeros(self.d_input), dtype=tf.float32)[tf.newaxis]
        for i in range(self.n_layers):
            logits = self.layers_[i](logits)

    def forward(self, input_, depth=None):
        logits = input_
        if depth == None:
            depth = self.n_layers
        # forward pass through depth layers
        for i in range(depth):
            logits = self.layers_[i](logits)
        return logits

    def all_variables(self):
        return (self.variables + self.optimizer.variables())

    def save(self):
        # save all relevant variables
        all_variables = self.all_variables()
        saver = tfe.Saver(all_variables).save(self.checkpoint_prefix, global_step=self.global_step)
        return saver

    def load(self):
        # get latest checkpoint
        fnames = os.listdir(self.experiment_folder)
        ckpt_names = [fname.split('.', 1)[0] for fname in fnames if self.nn_name in fname and '.data' in fname]
        global_steps = [int(name.split('-',-1)[1]) for name in ckpt_names]
        latest_ckpt_name = ckpt_names[np.argmax(global_steps)]
        latest_ckpt = os.path.join(self.experiment_folder, latest_ckpt_name)
        tfe.Saver(self.all_variables()).restore(latest_ckpt)

class VPGAgent(FNN):
    """ Agent trained via REINFORCE (vanilla policy gradients)
        Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning
        Williams, 1992

        Sample trajectories and update the policy accordingly. On-policy algorithm.
    """
    def __init__(self, d_input, d_hidden_layers, d_output, learning_rate, reward_discount_factor,
                 baseline=None, activation='tanh',
                 experiment_folder=os.getcwd(), name='vpg'):
        super().__init__(d_input, d_hidden_layers, d_output, learning_rate, [], activation, experiment_folder, name)
        self.reward_discount_factor = reward_discount_factor
        self.baseline = baseline

    def action(self, obs):
        logits = self.forward(obs)
        # add 1e-6 offset, otherwise np.random.multinomial might lead to an error
        policy = tf.nn.softmax(logits).numpy()[0] + 1e-6
        policy /= np.sum(policy)
        action = np.argmax(np.random.multinomial(1, policy))
        return action, policy

    def loss(self, obs_batch, action_batch, reward_batch, done_batch):
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
                state_values = np.reshape(self.baseline.forward(np.vstack(obs_batch[i]).astype(np.float32)).numpy(),[-1])
                state_value_batch.append(state_values)
            advantage_batch = advantage_values(obs_batch, reward_batch,
                                               done_batch, state_value_batch,
                                               self.baseline.reward_discount_factor,
                                               self.baseline.gae_lamda)
            advantages = np.hstack(advantage_batch).astype(np.float32)
            stats['baseline_value'] = np.mean(advantages)
            crewards = advantages

        # standardise crewards/advantages: introduces bias but reduces variance
        #crewards = (crewards - np.mean(crewards)) / np.std(crewards)

        # vpg loss
        logits = self.forward(obs)
        log_policy = tf.nn.log_softmax(logits)
        log_policy_for_actions = tf.reduce_sum(log_policy * actions, axis=1)
        loss = - tf.reduce_mean(log_policy_for_actions * crewards, axis=0)
        return loss, stats

    def grads(self, obs_batch, action_batch, reward_batch, done_batch):
        with tfe.GradientTape() as tape:
            loss, stats = self.loss(obs_batch, action_batch, reward_batch, done_batch)
        return loss, tape.gradient(loss, self.trainable_variables), stats

    def update(self, obs_batch, action_batch, reward_batch, done_batch):
        loss, grads, stats = self.grads(obs_batch, action_batch, reward_batch, done_batch)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.global_step += 1

        # monte carlo update of baseline
        if self.baseline!=None:
            obs = stats['obs']
            crewards = stats['crewards']
            baseline_loss, baseline_grads = self.baseline.mc_update(obs, crewards)
            stats['baseline_loss'] = baseline_loss
        return loss, grads, stats

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

    def load(self):
        """ overwrite load function """
        super(PPOAgent,self).load()
        self.old_trainable_weights = self.get_trainable_weights()

    def action(self, obs):
        """ return action and policy """
        logits = self.forward(obs)
        policy = tf.nn.softmax(logits).numpy()[0] + 1e-6  # add 1e-6 offset
        policy /= np.sum(policy)
        action = np.argmax(np.random.multinomial(1, policy))
        return action, policy

    def entropy(self, logits):
        """ compute entropy of policy with fixed state input """
        policy = tf.nn.softmax(logits)
        entropy = - tf.reduce_sum(policy * tf.log(policy), axis=1)
        return entropy

    def forward_old_policy(self, obs):
        """ forward with old policy weights """
        current_trainable_weights = self.get_trainable_weights()
        self.set_trainable_weights(self.old_trainable_weights) # set old weights
        forward = super(PPOAgent,self).forward(obs)
        self.set_trainable_weights(current_trainable_weights) # reset to current weights
        return forward

    def forward(self, obs):
        """ overwrite forward function """
        return self.forward_old_policy(obs)

    def forward_current_policy(self, obs):
        """ forward with current policy weights """
        return super(PPOAgent,self).forward(obs)

    def loss(self, obs_batch, action_batch, reward_batch, done_batch):
        """ ppo loss function"""
        # cumulative rewards
        creward_batch = cumulative_rewards(reward_batch, self.reward_discount_factor)

        # flatten out episodes in batch
        obs = np.vstack(obs_batch).astype(np.float32)
        actions = np.vstack(action_batch).astype(np.float32)
        crewards = np.hstack(creward_batch).astype(np.float32)
        stats = {'obs': obs, 'actions': actions , 'crewards': crewards}

        # use advantage baseline
        if self.baseline!=None:
            # state-value function values
            state_value_batch = []
            for i in range(len(obs_batch)):
                state_values = np.reshape(self.baseline.forward(np.vstack(obs_batch[i]).astype(np.float32)).numpy(), [-1])
                state_value_batch.append(state_values)
            # advantage function
            advantage_batch = advantage_values(obs_batch, reward_batch, done_batch, state_value_batch,
                                               self.baseline.reward_discount_factor, self.baseline.gae_lamda)
            advantages = np.hstack(advantage_batch).astype(np.float32)
            stats['baseline_value'] = np.mean(advantages)
            crewards = advantages

        # standardise advantages and crewards: introduces bias but reduces variance
        #crewards = (crewards - np.mean(crewards)) / np.std(crewards)

        # old log policy: log p(a_t|s_t, theta_old) for given (s_t, a_t) pairs
        logits = self.forward_old_policy(obs)
        log_old_policy = tf.nn.log_softmax(logits)
        log_old_policy_for_actions = tf.reduce_sum(log_old_policy * actions, axis=1).numpy() # not a tensor

        # current log policy: log p(a_t|s_t, theta) for given (s_t, a_t) pairs
        logits = self.forward_current_policy(obs)
        log_policy = tf.nn.log_softmax(logits)
        log_policy_for_actions = tf.reduce_sum(log_policy * actions, axis=1)

        # ratio: r_t(theta) = p(a_t|s_t, theta) / p(a_t|s_t, theta_old)
        ratio = tf.exp(log_policy_for_actions - log_old_policy_for_actions)
        stats['ratio'] = np.mean(ratio.numpy())

        # clipped policy objective that should be maximised
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        policy_loss = - tf.reduce_mean(tf.minimum(ratio * crewards, clipped_ratio * crewards), axis=0)
        stats['policy_loss'] = policy_loss.numpy()
        stats['clipped_ratio'] = np.mean(clipped_ratio.numpy())
        stats['n_clips'] = sum(diff > 10e-6 for diff in ratio.numpy() - clipped_ratio.numpy())

        # entropy loss (exploration bonus)
        entropy_loss = - tf.reduce_mean(self.entropy(logits), axis=0)
        stats['entropy_loss'] = entropy_loss.numpy()

        # complete loss
        loss = policy_loss # + 0.01 * entropy_loss

        return loss, stats

    def grads(self, obs_batch, action_batch, reward_batch, done_batch):
        with tfe.GradientTape() as tape:
            loss, stats = self.loss(obs_batch, action_batch, reward_batch, done_batch)
        return loss, tape.gradient(loss, self.trainable_variables), stats

    def get_trainable_weights(self):
        """ return weights as numpy arrays """
        ws = []
        for i,layer in enumerate(self.layers_):
            ws.append([w.numpy() for w in layer.trainable_weights])
        return ws

    def set_trainable_weights(self, trainable_weights):
        """ assign numpy arrays to weights """
        for i,layer in enumerate(self.layers_):
            ws = [w for w in trainable_weights[i]]
            layer.set_weights(ws)

    def update(self, obs_batch, action_batch, reward_batch, done_batch):
        loss, grads, stats = self.grads(obs_batch, action_batch, reward_batch, done_batch)

        # reset old policy weights to the policy before th gradient update
        self.old_trainable_weights = self.get_trainable_weights()

        # apply gradients to obtain new policy
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.global_step += 1

        # monte carlo update of baseline
        if self.baseline!=None:
            obs = stats['obs']
            crewards = stats['crewards']
            baseline_loss, baseline_grads = self.baseline.mc_update(obs, crewards)
            stats['baseline_loss'] = baseline_loss
        return loss, grads, stats

class DQNAgent(FNN):
    """ Agent trained via deep q-learning
        Playing Atari with Deep Reinforcement Learning, Mnih et al.
        arXiv:1312.5602v1  [cs.LG]  19 Dec 2013
    """
    def __init__(self, d_input, d_hidden_layers, d_output, learning_rate, gamma, activation='tanh',
                 experiment_folder=os.getcwd(), name='dqn'):
        super().__init__(d_input, d_hidden_layers, d_output, learning_rate, [], activation, experiment_folder, name)
        self.gamma = gamma

    def action(self, obs):
        logits = self.forward(obs)
        # take deterministic action according to largest Q-value
        policy = tf.nn.softmax(logits).numpy()[0] + 1e-6  # add 1e-6 offset to prevent numeric underflow
        policy /= np.sum(policy)
        action = np.argmax(policy)
        # take probabilistic action
        action = np.argmax(np.random.multinomial(1, policy))
        return action, policy

    def grads(self, obs, action, next_obs, reward, done):
        # compute target q-values
        with tfe.GradientTape() as tape:
            q = self.forward(obs)
            next_q = self.forward(next_obs).numpy()
            max_next_q = np.max(next_q, axis=1)
            q_target = q.numpy()
            for x, y in enumerate(action):
                q_target[x, y] = reward[x] + self.gamma * (1. - done[x]) * max_next_q[x]
            loss = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        return loss, tape.gradient(loss, self.trainable_variables)

    def update(self, obs, actions, next_obs, rewards, done):
        loss, grads = self.grads(obs, actions, next_obs, rewards, done)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.global_step += 1
        return loss, grads

    def flat_variables(self):
        return tf.concat([tf.reshape(v, [-1]) for v in self.trainable_variables], axis=0)

    def get_trainable_weights(self):
        """ return weights as numpy arrays """
        ws = []
        for i,layer in enumerate(self.layers_):
            ws.append([w.numpy() for w in layer.trainable_weights])
        return ws

    def set_trainable_weights(self, trainable_weights):
        """ assign numpy arrays to weights """
        for i,layer in enumerate(self.layers_):
            ws = [w for w in trainable_weights[i]]
            layer.set_weights(ws)

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

    def td0_grads(self, obs, next_obs, rewards, dones):
        # td(0) bootstrapping
        with tfe.GradientTape() as tape:
            state_values = self.forward(obs)
            next_state_values = self.forward(next_obs).numpy()
            targets = np.reshape(rewards, [-1,1]) + (1 - np.reshape(dones, [-1,1])) * next_state_values
            loss = tf.losses.mean_squared_error(labels=targets, predictions=state_values)
        return loss, tape.gradient(loss, self.trainable_variables)

    def td0_update(self, obs, next_obs, rewards, dones):
        loss, grads = self.td0_grads(obs, next_obs, rewards, dones)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.global_step += 1
        return loss, grads

    def mc_grads(self, obs, crewards):
        with tfe.GradientTape() as tape:
            loss = self.mc_loss(obs, crewards)
        return loss, tape.gradient(loss, self.trainable_variables)

    def mc_update(self, obs, crewards):
        loss, grads = self.mc_grads(obs, crewards)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.global_step +=1
        return loss, grads

    def mc_loss(self, obs, crewards):
        # monte carlo loss
        state_values = self.forward(obs)
        targets = np.reshape(crewards, [-1, 1])
        loss = tf.losses.mean_squared_error(labels=targets, predictions=state_values)
        return loss


