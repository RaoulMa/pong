import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import gym
import cv2
import sys

class AtariGame(object):
    """ Environment wrapper for Atari Pong and Breakout

    This class simplifies the interaction of the agent with the Pong
    environment. The API follows the OpenAI gym API.

    Each frame (RGB image) of shape (210, 160, 3) will be rescaled to
    grayscale (84,84,1).

    The observation state contains 4 stacked frames and is of shape
    (84,84,4)

    The last frame results from the current action
    while the previous 3 frames from the previous 3 actions.

    Actions for Atari Pong
    0 (no operation)
    1 (fire)
    2 (right)
    3 (left)
    """
    def __init__(self, env_name, seed):
        self.env_name = env_name
        self.seed = seed
        self.env = gym.make(self.env_name)
        self.env.seed(self.seed)

        self.lives = 0
        self.n_plays = 0
        self.done_game = True

        # 4 frame stack
        self.obs = np.zeros((84, 84, 4))
        self.d_observation = (84,84,4)

        # number of actions
        # we fix the number of allowed actions to 2 (left or right)
        self.actions = [2,3]
        self.n_actions = len(self.actions)

        # number of frames per step i.e. the same action
        # is applied n_frames times
        self.n_frames = 4

        # maximal number of plays per game
        # Atari Pong does not limit this number
        self.max_n_plays = 18

    def reset(self):
        """ Reset environment """

        if self.done_game:
            # reset game
            obs = self.env.reset()

            # fire = start the game
            obs, reward, done, info = self.env.step(1)
            obs = self.encode_obs(obs)  # (84, 84, 1)

            # one game (episode) consists of several plays
            self.done_game = False
            self.n_plays = 0
            self.lives = self.env.unwrapped.ale.lives()

            # fill whole stack with current frame
            self.obs[..., 0:] = obs
            self.obs[..., 1:] = obs
            self.obs[..., 2:] = obs
            self.obs[..., 3:] = obs

        return self.obs

    def encode_obs(self, obs):
        """ Convert one frame to gray scale, resize, crop, normalise """

        # use numpy
        obs = obs[35:195] # crop playing area
        obs = obs[::2,::2,0] # downsample by factor of 2
        obs = np.pad(obs,((2,2),(2,2)), 'edge') # pad to 84x84 shape
        obs[obs == 144] = 0  # erase background (background type 1)
        obs[obs == 109] = 0  # erase background (background type 2)
        obs[obs != 0] = 1  # everything else (paddles, ball) just set to 1
        obs = obs.reshape(84, 84, 1).astype(np.float32)

        # use opencv
        #obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        #obs = cv2.resize(obs, (84, 110), interpolation=cv2.INTER_AREA)
        #obs = obs[17:101, :] # crop playing area
        #obs = obs.reshape(84, 84, 1).astype(np.float32)
        #obs = obs / 255.

        return obs

    def step(self, action):
        """ One environmental step

        Args:
            action (int): 0,1,2,3
        Returns:
            self.obs (np.array): observation tensor (84,84,4)
            total_reward (float): sum of all rewards for each interaction
            done (bool): if episode terminated

        Given an action, the action is applied 4 times on the actual game, and
        the last two frames are combined to form the final frame for
        the given action. This final frame is then stacked on the observation
        tensor.
        """

        # map the correct action
        action = self.actions[action]

        # sum of reards over several frames
        total_reward = 0.

        done_play = False # if one play ended
        obs = None
        for i in range(self.n_frames):

            prev_obs = obs
            obs, reward, done, info = self.env.step(action)

            # reward clipping
            reward = np.sign(reward)

            if i == 0:
                prev_obs = obs

            # sum of rewards from each frame
            total_reward += reward

            # maximal number of plays per game
            #if self.n_plays >= self.max_n_plays:
            if done:
                self.done_game = True

            # one play ended if reward -1 or +1 for Pong
            if reward != 0:
                self.n_plays += 1
                done = True

            # episode has terminated
            if done:
                done = True
                break

        #self.obs_unprocessed = obs

        # Take the maximum of each pixel of the last two frames, which
        # might be necessary due to pixel fluctuations
        prev_obs = self.encode_obs(prev_obs)
        obs = self.encode_obs(obs)
        obs = np.max(np.array([prev_obs, obs]), axis=0)

        # Add the maximum of last two frames to the 4-frame stack.
        # Taking the maximum avoids flickering images of Atari
        self.obs[..., -1:] = obs

        return self.obs, total_reward, done, info

    def render(self, fname=None):
        #matplotlib.use('TkAgg')
        #plt.imshow(self.obs[:,:,-1], cmap=cm.gray, vmin=0., vmax=1.)
        #plt.imshow(self.obs_unprocessed)
        #if fname is not None:
        #    plt.savefig(fname)
        #plt.show()
        self.env.render()

    def close(self):
        self.env.close()

