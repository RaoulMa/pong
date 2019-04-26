# About
Implementation of PPO, DQN, VPG agenst in Tensorflow to solve Atari Pong and CartPole. 
Some of the agents are also implemented in Tensorflow Eager and PyTorch.

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

All runs create tensorflow checkpoints and tensorboard event files in the results folder.

### Environment
We consider the Atari Pong Environment. The gif below shows the unprocessed environment for one game play. 

![](images/pong_not_preprocessed.gif) 

We preprocess each image by resizing, cropping, grayscaling to speed up learning. The following gif shows 
the preprocessed inputs to the agents.

![](images/pong_preprocessed.gif)


### Results
This plot shows the average reward of the DQN agent while training for 1 million frames/steps. 
Losing or winning yields a reward of -1 or +1. Note, while training the agent epsilon decreases 
from 1 to 0.1 for the first 100000 frames. If we set epsilon to zero the final agent 
has learnt to beat the Pong player more than 90% of the time.
![dqn returns](images/dqn_ext_return_versus_model_name.png)



