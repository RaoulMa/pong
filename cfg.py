from utils import dotdict

def get_cfg(experiment_folder, env_name, agent_name):
    """ config files for environments and agents """

    """
    Pong-v0
    """
    if env_name == 'PongNoFrameskip-v4':
        cfg_env = dotdict({
            'experiment_folder': experiment_folder,
            'verbose': False,
            'env_name': 'PongNoFrameskip-v4',
            'seed': 1,
            'global_step': 0,
            'reward_discount_factor': 0.99,
            'time_reward': 0.0,
            'n_steps': 1000000,
            'log_step': 1000,
            'agent_buffer_size': 100000,
            'agent_buffer_batch_size': 32,
            'observation_encoding': 'None',
            'activation': 'relu',
            'gae_lamda': 0.97,
        })
        if agent_name == 'ppo':
            cfg_agent = dotdict({
                'model_name': 'ppo',
                'batch_size': 4,
                'agent_d_hidden_layers': [32],
                'agent_learning_rate': 0.00025,
                'clip_range': 0.1,
                'baseline_d_hidden_layers': [],
                'baseline_learning_rate': 0.00025,
                'baseline': 'shared_advantage',
            })
        if agent_name == 'vpg':
            cfg_agent = dotdict({
                'model_name': 'vpg',
                'batch_size': 4,
                'agent_d_hidden_layers': [32],
                'agent_learning_rate': 0.00025,
                'baseline_d_hidden_layers': [32],
                'baseline_learning_rate': 0.00025,
                'baseline': 'shared_advantage',
            })
        if agent_name == 'dqn':
            cfg_agent = dotdict({
                'model_name': 'dqn',
                'update_target_network_freq': 1000,
                'agent_d_hidden_layers': [32],
                'agent_learning_rate': 0.00025,
                'baseline': 'None',
                'epsilon_start': 1.0,
                'epsilon_step_range': 10000,
                'epsilon_final': 0.1,
                'agent_buffer_start_size': 10000
            })

    """ 
    Breakout-v0 
    """
    if env_name == 'BreakoutNoFrameskip-v4':
        cfg_env = dotdict({
            'experiment_folder': experiment_folder,
            'verbose': False,
            'env_name': 'BreakoutNoFrameskip-v4',
            'seed': 1,
            'global_step': 0,
            'reward_discount_factor': 0.99,
            'time_reward': 0.0,
            'n_steps': 1000000,
            'log_step': 1000,
            'agent_buffer_size': 1000000,
            'agent_buffer_batch_size': 32,
            'observation_encoding': 'None',
            'activation': 'relu',
            'gae_lamda': 0.97,
        })
        if agent_name == 'ppo':
            cfg_agent = dotdict({
                'model_name': 'ppo',
                'batch_size': 4,
                'agent_d_hidden_layers': [32],
                'agent_learning_rate': 0.00025,
                'clip_range': 0.1,
                'baseline_d_hidden_layers': [],
                'baseline_learning_rate': 0.001,
                'baseline': 'shared_advantage',
            })
        if agent_name == 'vpg':
            cfg_agent = dotdict({
                'model_name': 'vpg',
                'batch_size': 4,
                'agent_d_hidden_layers': [32],
                'agent_learning_rate': 0.00025,
                'baseline_d_hidden_layers': [32],
                'baseline_learning_rate': 0.00025,
                'baseline': 'advantage',
            })
        if agent_name == 'dqn':
            cfg_agent = dotdict({
                'model_name': 'dqn',
                'update_target_network_freq': 1000,
                'agent_d_hidden_layers': [32],
                'agent_learning_rate': 0.00025,
                'epsilon': 1.,
                'epsilon_discount_factor': 0.9999,
                'baseline': 'None',
                'agent_buffer_start_size': 50000
            })

    """ 
    Four-Rooms Maze
    """
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
            'n_steps': 50000,
            'log_step': 1000,
            'agent_buffer_size': 100000,
            'agent_buffer_batch_size': 32,
            'observation_encoding': 'normalise',
            'activation': 'relu',
            'gae_lamda': 0.97,
        })
        if agent_name == 'ppo':
            cfg_agent = dotdict({
                'model_name': 'ppo',
                'batch_size': 4,
                'agent_d_hidden_layers': [32],
                'agent_learning_rate': 0.001,
                'clip_range': 0.1,
                'baseline_d_hidden_layers': [32],
                'baseline_learning_rate': 0.001,
                'baseline': 'advantage',
            })
        if agent_name == 'vpg':
            cfg_agent = dotdict({
                'model_name': 'vpg',
                'batch_size': 4,
                'agent_d_hidden_layers': [32],
                'agent_learning_rate': 0.001,
                'baseline_d_hidden_layers': [32],
                'baseline_learning_rate': 0.001,
                'baseline': 'advantage',
            })
        if agent_name == 'dqn':
            cfg_agent = dotdict({
                'model_name': 'dqn',
                'update_target_network_freq': 1000,
                'agent_d_hidden_layers': [32],
                'agent_learning_rate': 0.001,
                'epsilon': 1.,
                'epsilon_discount_factor': 0.9999,
                'baseline': 'None',
                'agent_buffer_start_size': 1000
            })


    """ 
    CartPole-v0 
    """
    if env_name == "CartPole-v0":
        cfg_env = dotdict({
            'experiment_folder': experiment_folder,
            'verbose': False,
            'env_name': 'CartPole-v0',
            'seed': 1,
            'n_steps': 50000,
            'log_step': 1000,
            'global_step': 0,
            'reward_discount_factor': 0.99,
            'agent_buffer_size': 1000000,
            'agent_buffer_batch_size': 32,
            'activation': 'relu',
            'gae_lamda': 0.97,
            'time_reward': 0,
            'observation_encoding': 'None',
        })
        if agent_name == 'ppo':
            cfg_agent = dotdict({
                'model_name': 'ppo',
                'batch_size': 4,
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
                'batch_size': 4,
                'agent_d_hidden_layers': [32],
                'agent_learning_rate': 0.001,
                'baseline_d_hidden_layers': [32],
                'baseline_learning_rate': 0.001,
                'baseline': 'advantage',
            })
        if agent_name == 'dqn':
            cfg_agent = dotdict({
                'model_name': 'dqn',
                'epsilon': 1.,
                'epsilon_discount_factor': 0.99,
                'update_target_network_freq': 1000,
                'agent_d_hidden_layers': [32],
                'agent_learning_rate': 0.001,
                'baseline': 'None',
                'agent_buffer_start_size': 1000
            })

    """
    random search
    """
    if agent_name == 'rs':
        cfg_agent = dotdict({
            'model_name': 'rs',
        })

    return cfg_env, cfg_agent
