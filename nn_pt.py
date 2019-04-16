import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import sys

from utils import advantage_values
from utils import cumulative_rewards

class FNN(nn.Module):
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
        self.checkpoint_prefix = os.path.join(experiment_folder, name + '_ckpt')

        self.d_layers = [d_input] + d_hidden_layers + [d_output]
        self.n_layers = len(self.d_layers)

        # todo: implement prev_layers
        self.layers_ = nn.ModuleList()
        for i in range(len(prev_layers), self.n_layers-1):
            self.layers_.append(nn.Linear(self.d_layers[i], self.d_layers[i + 1], bias=True))
            if i < (self.n_layers - 2):
                if activation == 'relu':
                    self.layers_.append(nn.ReLU())
                else:
                    self.layers_.append(nn.Tanh())

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        x = torch.tensor(x).float()
        model = torch.nn.Sequential(*self.layers_)
        return model(x)

    def all_variables(self):
        pass

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
                 baseline=None, activation='tanh',
                 experiment_folder=os.getcwd(), name='vpg'):
        super().__init__(d_input, d_hidden_layers, d_output, learning_rate, [], activation, experiment_folder, name)
        self.reward_discount_factor = reward_discount_factor
        self.baseline = baseline

    def action(self, obs):
        logits = self.forward(obs)
        # add 1e-6 offset, otherwise np.random.multinomial might lead to an error
        policy = F.softmax(logits, dim=-1).data.numpy()[0]+ 1e-6
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
                state_values = np.reshape(self.baseline.forward(np.vstack(obs_batch[i]).astype(np.float32)).data.numpy(),[-1])
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
        log_policy = F.log_softmax(logits, dim=-1)
        log_policy_for_actions = torch.sum(log_policy * torch.from_numpy(actions), dim=1)
        loss = - torch.mean(log_policy_for_actions * torch.from_numpy(crewards), dim=0)
        return loss, stats

    def update(self, obs_batch, action_batch, reward_batch, done_batch):
        loss, stats = self.loss(obs_batch, action_batch, reward_batch, done_batch)
        self.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.global_step += 1

        # monte carlo update of baseline
        if self.baseline!=None:
            obs = stats['obs']
            crewards = stats['crewards']
            baseline_loss, _ = self.baseline.mc_update(obs, crewards)
            stats['baseline_loss'] = baseline_loss
        return loss.data.numpy(), None, stats

    def close(self):
        pass

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

    def close(self):
        pass


    # todo: implementation in pytorch

class DQNAgent(FNN):
    """ Agent trained via deep q-learning
        Playing Atari with Deep Reinforcement Learning, Mnih et al.
        arXiv:1312.5602v1  [cs.LG]  19 Dec 2013
    """
    def __init__(self, d_input, d_hidden_layers, d_output, learning_rate, gamma, activation='tanh',
                 experiment_folder=os.getcwd(), name='dqn'):
        super().__init__(d_input, d_hidden_layers, d_output, learning_rate, [], activation, experiment_folder, name)
        self.gamma = gamma
        self._loss = nn.MSELoss()

    def action(self, obs):
        logits = self.forward(obs)
        # add 1e-6 offset, otherwise np.random.multinomial might lead to an error
        policy = F.softmax(logits, dim=-1).data.numpy()[0] + 1e-6
        policy /= np.sum(policy)
        action = np.argmax(np.random.multinomial(1, policy))
        return action, policy

    def loss(self, obs, action, next_obs, reward, done):
        # current q-value
        q = self.forward(obs)
        # next q-value
        next_q = self.forward(next_obs).data.numpy()
        # max next q-value
        max_next_q = np.max(next_q, axis=1)
        # target q-value
        q_target = q.clone().data.numpy()
        for x, y in enumerate(action):
            q_target[x, y] = reward[x] + self.gamma * (1. - done[x]) * max_next_q[x]
        q_target = torch.tensor(q_target).float()
        loss = self._loss(input=q, target=q_target).unsqueeze(0)
        return loss

    def update(self, obs, actions, next_obs, rewards, done):
        loss = self.loss(obs, actions, next_obs, rewards, done)
        self.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.global_step += 1
        return loss.data.numpy(), None

    def get_trainable_weights(self):
        """ return weights as numpy arrays """
        ws = []
        for i, layer in enumerate(self.layers_):
            ws.append([w.clone().data.numpy() for w in list(layer.parameters())])
        return ws

    def set_trainable_weights(self, trainable_weights):
        """ assign numpy arrays to weights """
        for i, layer in enumerate(self.layers_):
            ws = [w for w in trainable_weights[i]]
            if len(ws) > 0:
                layer.weight = nn.Parameter(torch.from_numpy(ws[0]))
                layer.bias = nn.Parameter(torch.from_numpy(ws[1]))

    def close(self):
        pass

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
        self.loss_ = nn.MSELoss()

    def mc_update(self, obs, crewards):
        loss = self.mc_loss(obs, crewards)
        self.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.global_step +=1
        return loss.data.numpy(), None

    def mc_loss(self, obs, crewards):
        state_values = self.forward(obs)
        targets = torch.from_numpy(np.reshape(crewards, [-1, 1])).float()
        loss = self.loss_(input=state_values, target=targets).unsqueeze(0)
        return loss

    def close(self):
        pass

