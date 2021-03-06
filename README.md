![](images/pong_not_preprocessed.gif)![](images/pong_preprocessed.gif)
### About
Implementation of PPO, DQN, VPG agents in Tensorflow to solve Atari Pong. 

This repository serves the purpose of self-teaching. The implementations 
are not particularly clear, efficient, well tested or numerically stable. We advise 
against using this software for non-didactic purposes.

This software is licensed under the MIT License.

### Installation and Usage
This code is based on [TensorFlow](https://www.tensorflow.org/). Install Python 3 with basic 
packages, then run these commands: 
```Shell
git clone -b master --single-branch https://github.com/RaoulMa/drl.git
python3 -m pip install --user --upgrade pip
python3 -m pip install --user -r requirements.txt 
```

Default hyperparameters are stored in cfg.py, where they can also be modified. To train the model
with default parameters, run the following command:
```Shell
python3 train.py
```

All runs create tensorflow checkpoints and tensorboard event files in the results folder. Already 
trained agents included in the 'results_train' folder. 

### Experiments 
The gifs on top of this page show the original (left) and preprocessed (right) observations for one game play for 
a trained agent. Regarding the training procedure we make some remarks:

- To speed up training we allow for two actions 'left' and 'right'. The actions 'no-op' and 'fire' are discarded.
- Each frame of Pong gets cropped to the playing region, gray-scaled and then resized to 
the shape 84x84x1. Then the background is set to 0 value while the remaining pixels are set 
to 1. 
- One action of the agent is applied on four adjacent frames. The last two frames are combined 
(taking the maximum) to one frame which is enqueued to a fixed-size queue of four frames of 
shape 84x84x4. All four frames are then fed into the agent. Thus, the agent has information 
about frames of previous time steps.
- In case of the DQN agent we decrease epsilon from 1 to 0.02 over 100000 frames. The target Q-network us updated 
every 1000 steps. The Q-network s updated every 4 steps with a batch of 32 transitions uniformly sampled from the
replay buffer.
- In case of the PPO agent we set the clipping range to 0.1 and choose a batch size of 8 episodes. 
- All default hyperparameters can be found in the file 'cfg.py'. 
- Due to computational resorces we have not made a hyperparameter seach and only considered
one fixed seed.

The following plot shows the average reward of the DQN agent while training for 650000 frames/steps. 
Losing or winning yields a reward of -1 or +1. Note, while training the agent epsilon decreases 
from 1 to 0.02 for the first 100000 frames. If we set epsilon to zero the final agent 
has learnt to beat the computer Pong player by more than 99% of the time.
![dqn returns](images/dqn_ext_return_versus_model_name.png)

Next, we see the training curve for the PPO agent. Apart from fluctuations the agent's performance converges 
to an average reward of +1 after 2 million frames. The final agent has learnt to beat the computer Pong 
player by more than 96% of the time. 
![ppo returns](images/ppo_ext_return_versus_model_name.png)

Finally, we look at the VPG agent. The training curve is much more unstable and doesn't seem to converge 
within 2.5 million frames.
![vpg returns](images/vpg_ext_return_versus_model_name.png)


### References

[1] Playing Atari with Deep Reinforcement Learning, Mnih et al. arXiv:1312.5602v1  [cs.LG]  19 Dec 2013

[2] Proximal Policy Optimization Algorithms, Schulman et al. 2017 arXiv:1707.06347v2  [cs.LG]  28 Aug 2017