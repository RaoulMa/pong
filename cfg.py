from utils import dotdict

def get_cfg(experiment_folder, env_name, agent_name):
    """ config files for environments and agents """

    ##########################
    #### Four Rooms Maze #####
    ##########################
    if env_name == 'four_rooms_maze':
        cfg_env = dotdict({
            'experiment_folder': experiment_folder,
            'verbose': False,
            'env_name': 'four_rooms_maze',
            'env_max_steps': 21,
            'env_reward': 10.,
            'seed': 1,
            'global_step': 0,
            'reward_discount_factor': 0.99,
            'time_reward': 0.0,
            'n_episodes': 10000,
            'n_batches': 10000,
            'batch_size': 1,
            'log_step': 100,
            'agent_buffer_size': 10000,
            'agent_buffer_batch_size': 64,
            'observation_encoding': 'one_hot',
            'activation': 'tanh',
            'gae_lamda': 0.97,
        })
        if agent_name == 'ppo':
            cfg_agent = dotdict({
                'model_name': 'ppo',
                'agent_d_hidden_layers': [32],
                'agent_learning_rate': 0.001,
                'clip_range': 0.2,
                'baseline_d_hidden_layers': [32],
                'baseline_learning_rate': 0.001,
                'baseline': 'advantage',
            })
        if agent_name == 'vpg':
            cfg_agent = dotdict({
                'model_name': 'vpg',
                'agent_d_hidden_layers': [32],
                'agent_learning_rate': 0.001,
                'baseline_d_hidden_layers': [32],
                'baseline_learning_rate': 0.001,
                'baseline': 'advantage',
            })
        if agent_name == 'dqn':
            cfg_agent = dotdict({
                'model_name': 'dqn',
                'update_target_network_freq': 10,
                'agent_d_hidden_layers': [32],
                'agent_learning_rate': 0.001,
                'epsilon': 1.,
                'epsilon_discount_factor': 0.999,
            })

    #################
    ## CartPole-v0 ##
    #################
    if env_name == "CartPole-v0":
        cfg_env = dotdict({
            'experiment_folder': experiment_folder,
            'verbose': False,
            'env_name': 'CartPole-v0',
            'seed': 1,
            'global_step': 0,
            'reward_discount_factor': 0.99,
            'epsilon': 0.,
            'epsilon_discount_factor': 0.999,
            'n_episodes': 10000,
            'n_batches': 10000,
            'batch_size': 1,
            'log_step': 100,
            'agent_buffer_size': 10000,
            'agent_buffer_batch_size': 64,
            'activation': 'tanh',
            'gae_lamda': 0.97,
            'time_reward': 0,
            'observation_encoding': 'None',
        })
        if agent_name == 'ppo':
            cfg_agent = dotdict({
                'model_name': 'ppo',
                'agent_d_hidden_layers': [32],
                'agent_learning_rate': 0.001,
                'clip_range': 0.2,
                'baseline_d_hidden_layers': [32],
                'baseline_learning_rate': 0.001,
                'baseline': 'advantage',
            })
        if agent_name == 'vpg':
            cfg_agent = dotdict({
                'model_name': 'vpg',
                'agent_d_hidden_layers': [32],
                'agent_learning_rate': 0.001,
                'baseline_d_hidden_layers': [32],
                'baseline_learning_rate': 0.001,
                'baseline': 'advantage',
            })
        if agent_name == 'dqn':
            cfg_agent = dotdict({
                'model_name': 'dqn',
                'update_target_network_freq': 10,
                'agent_d_hidden_layers': [32],
                'agent_learning_rate': 0.001,
            })

    # random search
    if agent_name == 'rs':
        cfg_agent = dotdict({
            'model_name': 'rs',
        })

    return cfg_env, cfg_agent
