import numpy as np
import pandas as pd
import os
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import seaborn as sns
import sys

class dotdict(dict):
    """ dictionary class """
    __getattr__ = dict.__getitem__
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError

def get_experiment_name(experiments_folder):
    """ create new experiment directory """
    if not os.path.exists(experiments_folder):
        os.makedirs(experiments_folder)
    dir_names = subdir_names(experiments_folder)
    c = 0
    for i, dir_name in enumerate(dir_names):
        if dir_name.isdigit() and int(dir_name) > c:
            c = int(dir_name)
    experiment_name = str(c + 1)
    experiment_folder = os.path.join(experiments_folder, experiment_name)
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    return experiment_name, experiment_folder

def subdir_paths(dpath):
    return [os.path.join(dpath, o) for o in os.listdir(dpath) if os.path.isdir(os.path.join(dpath, o))]

def subdir_names(dpath):
    return [o for o in os.listdir(dpath) if os.path.isdir(os.path.join(dpath, o))]

def data_to_txt(data, fpath):
    with open(fpath, 'w') as f:
        f.write(data)

def txt_to_data(fpath):
    with open(fpath) as f:
        data = f.readlines()
    return data

def data_to_json(data, fpath):
    with open(fpath, 'w') as f:
        f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))

def json_to_data(fpath):
    with open(fpath, 'r') as f:
        data = json.load(f)
    return data

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_df(df, x, y, hue, title, fpath, footnote = None, show=False):
    """ lineplot and save figure """
    plt.clf()
    if footnote!=None:
        plt.figure(figsize=(15, 15))
    else:
        plt.figure(figsize=(15, 9))
    if not show: plt.ioff()
    sns.set(style='darkgrid')
    ax = sns.lineplot(x=x, y=y, hue=hue, data=df)
    ax.set(title=title)
    if footnote!=None:
        plt.annotate(footnote, (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top')
    plt.tight_layout()
    plt.savefig(fpath, dpi=300)
    if show:
        plt.show()
    plt.close

def find_optimal_hp(experiment_folder, variable_name, hp_fixed={}):
    """ get best hyperparameters that maximize the variable """
    dpaths = subdir_paths(experiment_folder)

    # read in events into dictionary for chosen hyperparameter and variable respecting the constraints
    variable = defaultdict(list)
    for dpath in dpaths:
        # read config file
        with open(os.path.join(dpath, 'params.json'), 'r') as f:
            data = json.load(f)
        cfg = dotdict(data['cfg'])

        # impose constraints on hyperparameters
        skip=False
        for key, value in hp_fixed.items():
            if cfg[key] not in value:
                skip=True
                break

        # read in data from files
        if not skip:
            if False:
                # use tf event files
                event_name = 'data/' + variable_name
                events, steps = read_tf_events(dpath, event_name)
                data = np.reshape(events[event_name], [-1])
            elif True:
                # use csv files
                event_name = 'data_' + variable_name
                fpath = os.path.join(dpath, event_name + '.csv')
                df = pd.read_csv(fpath)
                steps = df.iloc[:,0].values
                data = df.iloc[:,1].values
            elif False:
                # use npy files
                data = np.load(os.path.join(dpath, variable_name + '.npy'))

            variable[os.path.basename(dpath)] = np.mean(data)

    dname = max(variable, key=variable.get)
    dpath = os.path.join(experiment_folder, dname)
    with open(os.path.join(dpath, 'params.json'), 'r') as f:
        data = json.load(f)
    cfg = dotdict(data['cfg'])
    print(cfg)

def plot_hp_sensitivities(experiment_folder, variable_name, hp_name, hp_fixed={}, details=False, box_pts=1, show=False):
    """ plot hyper-parameter sensitivities """
    dpaths = subdir_paths(experiment_folder)

    cfg_global = {}
    # read in events into dictionary for chosen hyperparameter and variable respecting the constraints
    variable = defaultdict(list)
    for dpath in dpaths:
        # read config file
        with open(os.path.join(dpath, 'cfg.json'), 'r') as f:
            data = json.load(f)
        cfg = dotdict(data)

        # impose constraints on hyperparameters
        skip=False
        for key, value in hp_fixed.items():
            if cfg[key] not in value:
                skip=True
                break

        # store all different hp's
        for key, value in cfg.items():
            if key not in cfg_global.keys():
                cfg_global[key] = []
            if value not in cfg_global[key]:
                cfg_global[key].append(value)

        # read in data from files
        if not skip:
            if False:
                # use tf event files
                event_name = 'data/' + variable_name
                events, steps = read_tf_events(dpath, event_name)
                data = np.reshape(events[event_name], [-1])
            elif True:
                # use csv files
                event_name = 'data_' + variable_name
                fpath = os.path.join(dpath, event_name + '.csv')
                df = pd.read_csv(fpath)
                steps = df.iloc[:,0].values
                data = df.iloc[:,1].values
            elif False:
                # use npy files
                data = np.load(os.path.join(dpath, variable_name + '.npy'))

            # smooth data
            data = smooth(data, box_pts)

            # append data to dictionary
            hp_value = str(cfg[hp_name])

            info = {}
            if hp_name == 'model_name' and details:
                if 'pg' in hp_value:
                    info['agent_d_hidden_layers'] = cfg.agent_d_hidden_layers
                if 'predictor' in hp_value:
                    info['predictor_d_hidden_layers'] = cfg.predictor_d_hidden_layers
            hp_value += str(info)

            if hp_value not in variable:
                variable[hp_value] = []
            variable[hp_value].append(data)

    # varied and fixed hp's
    cfg_varied, cfg_fixed = {}, {}
    for key, value in cfg_global.items():
        if key != hp_name:
            if len(value) > 1:
                cfg_varied[key] = value
            else:
                cfg_fixed[key] = value

    # create x,y,hp arrays
    x_values, y_values, hp_values = [], [], []
    for key, value in variable.items():
        x_values.append(np.tile(steps, len(value)))
        y_values.append(np.reshape(value, [-1]))
        hp_values.append([key for _ in range(len(y_values[-1]))])

    x_values = np.hstack(x_values)
    y_values = np.hstack(y_values)
    hp_values = [item for sublist in hp_values for item in sublist]

    # construct dataframe
    df = pd.DataFrame()
    df['number_of_steps'] = x_values
    df[variable_name] = y_values
    df[hp_name] = hp_values

    # plot
    fname = variable_name + '_versus_' + hp_name + '.png'
    title = cfg.env_name
    fpath = os.path.join(experiment_folder, fname)
    footnote = None
    if details:
        footnote = 'varied hyperparameters\n' + str(cfg_varied) + '\n\nfixed hyperparameters\n' + str(cfg_fixed).replace('],','],\n')
    plot_df(df, 'number_of_steps', variable_name, hp_name, title, fpath, footnote, show)

def read_tf_events(dpath, event_name=None):
    """ read tf events """
    fnames = os.listdir(dpath)
    event_fnames = [fname for fname in fnames if 'tf.events.model' in fname]
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in event_fnames]
    out = defaultdict(list)
    steps = []

    if event_name is not None:
        # read in only one tag
        for it in summary_iterators:
            assert event_name in it.Tags()['scalars']
        tags = [event_name]
    else:
        # read all tags
        tags = summary_iterators[0].Tags()['scalars']
        for it in summary_iterators:
            assert it.Tags()['scalars'] == tags

    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]
        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1
            out[tag].append([e.value for e in events])
    return out, steps

def tf_events_to_csv(dpath):
    """ write all tf events to csv files """
    fnames = os.listdir(dpath)
    event_fnames = [fname for fname in fnames if 'tf.events.model' in fname]
    out, steps = read_tf_events(dpath)
    tags, values = zip(*out.items())
    np_values = np.array(values)

    for index, tag in enumerate(tags):
        df = pd.DataFrame(np_values[index], index=steps, columns=event_fnames)
        fpath = os.path.join(dpath, tag.replace("/", "_") + '.csv')
        df.to_csv(fpath)

def one_hot(value, n_classes):
    """ one-hot encoding """
    enc = np.zeros(n_classes, 'uint8')
    enc[value] = 1
    return enc

def cumulative_rewards(reward_batch, gamma):
    """ discounted cumulative sum of rewards for t = 1,2,...,T-1 """
    crewards = []
    for i in range(len(reward_batch)):
        gamma_mask = gamma ** np.arange(len(reward_batch[i]))
        cr = np.flip(np.cumsum(np.flip(gamma_mask * reward_batch[i]), axis=0)) / (gamma_mask + 1e-8)
        crewards.append(cr.tolist())
    return crewards

def advantage_values(obs_batch, reward_batch, done_batch, state_value_batch, gamma, lamda):
    """ generalised advantage estimate of the advantage function
    if lamda = 1: A_t = R_t - V(s_t)
    if lamda = 0: A_t = r_t+1 + gamma * V(s_t+1) - V(s_t)
    max. episode length: T
    obs_batch: s_1, s_2 ..., s_T-1 (i.e. without s_T)
    reward_batch: r_2, r_3, ..., r_T (r_2 is the reward obtained after taking action a_1 in s_1)
    done_batch: d_2, d_3, ..., d_T
    """
    n_episodes = len(obs_batch)
    ep_lengths = [len(obs_batch[i]) for i in range(len(obs_batch))]
    max_ep_length = max(ep_lengths)

    # obtain equal-sized arrays
    obs_shape = obs_batch[0][0].shape[1:]
    obs_arr = np.zeros((n_episodes, max_ep_length, *obs_shape))
    reward_arr = np.zeros((n_episodes, max_ep_length))
    state_value_arr = np.zeros((n_episodes, max_ep_length))
    advantage_arr = np.zeros((n_episodes, max_ep_length))
    done_arr = np.ones((n_episodes, max_ep_length))  # padding with ones

    for i in range(n_episodes):
        obs_arr[i, :ep_lengths[i]] = obs_batch[i]
        reward_arr[i, :ep_lengths[i]] = reward_batch[i]
        done_arr[i, :ep_lengths[i]] = done_batch[i]
        state_value_arr[i, :ep_lengths[i]] = state_value_batch[i]

    advantage_value = 0.  # A_T = 0
    next_state_value = 0.  # set V(s_T) = 0 since done = True

    for t in reversed(range(max_ep_length)):
        # only keep V(s_t+1) if done = False
        mask = 1.0 - done_arr[:, t]
        next_state_value = next_state_value * mask

        # td(0) error: delta_t = r_(t+1) + gamma * V(s_t+1) - V(s_t)
        delta = reward_arr[:, t] + gamma * next_state_value - state_value_arr[:, t]

        # advantage: A_t = delta_t + gamma * lamda * A_t+1
        advantage_value = delta + gamma * lamda * advantage_value
        advantage_arr[:, t] = advantage_value

        # V(s_t)
        next_state_value = state_value_arr[:, t]

    advantage_batch = [advantage_arr[i, :ep_lengths[i]] for i in range(n_episodes)]

    return advantage_batch




