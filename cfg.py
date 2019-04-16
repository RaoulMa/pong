from utils import dotdict

def get_cfg(experiment_folder, env_name, agent_name):
    """ config files for environments and agents """

    """
    Pong-v0
    """
    if env_name == 'Pong-v0':
        cfg_env = dotdict({
            'experiment_folder': experiment_folder,
            'verbose': False,
            'env_name': 'Pong-v0',
            'seed': 1,
            'global_step': 0,
            'reward_discount_factor': 0.99,
            'epsilon': 1.,
            'epsilon_discount_factor': 0.999,
            'time_reward': 0.0,
            'n_episodes': 1000,
            'log_step': 100,
            'agent_buffer_size': 10000,
            'agent_buffer_batch_size': 64,
            'observation_encoding': 'None',
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
                'baseline_d_hidden_layers': [],
                'baseline_learning_rate': 0.001,
                'baseline': 'shared_advantage',
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
            'epsilon': 1.,
            'epsilon_discount_factor': 0.999,
            'time_reward': 0.0,
            'n_episodes': 1000,
            'log_step': 100,
            'agent_buffer_size': 10000,
            'agent_buffer_batch_size': 64,
            'observation_encoding': 'None',
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
                'baseline_d_hidden_layers': [],
                'baseline_learning_rate': 0.001,
                'baseline': 'shared_advantage',
            })

    """ 
    Four Rooms Maze
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
            'n_episodes': 1000,
            'log_step': 100,
            'agent_buffer_size': 10000,
            'agent_buffer_batch_size': 64,
            'observation_encoding': 'one_hot',
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
                'update_target_network_freq': 100,
                'agent_d_hidden_layers': [32],
                'agent_learning_rate': 0.001,
                'epsilon': 1.,
                'epsilon_discount_factor': 0.999,
                'baseline': 'None'
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
            'n_episodes': 1000,
            'log_step': 100,
            'global_step': 0,
            'reward_discount_factor': 0.99,
            'epsilon': 0.,
            'epsilon_discount_factor': 0.999,
            'agent_buffer_size': 10000,
            'agent_buffer_batch_size': 64,
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
                'update_target_network_freq': 100,
                'agent_d_hidden_layers': [32],
                'agent_learning_rate': 0.001,
                'baseline': 'None'
            })

    """
    random search
    """
    if agent_name == 'rs':
        cfg_agent = dotdict({
            'model_name': 'rs',
        })

    return cfg_env, cfg_agent
