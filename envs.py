import numpy as np
import re
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import gym
import cv2
import sys

class AtariGame(object):
    """ Environment wrapper for Atari Pong and Breakout

    This class simplifies the interaction of the agent with the Breakout
    environment. The API follows the OpenAI gym API.

    Each frame (RGB image) of shape (210, 160, 3) will be rescaled to
    grayscale (84,84,1).

    The observation state contains 4 stacked frames and is of shape
    (84,84,4)

    The last frame results from the current action
    while the previous 3 frames from the previous 3 actions.

    Actions for Atari Pong and Breakout
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

        # remaining lives
        self.lives = 0

        # 4 frame stack
        self.obs = np.zeros((84, 84, 4))
        self.d_observation = (84,84,4)

        # number of actions
        self.n_actions = self.env.action_space.n

        # number of frames per step
        if 'Pong' in self.env_name:
            self.n_frames = 2
        elif 'Breakout' in self.env_name:
            self.n_frames = 4

    def reset(self):
        """ Reset environment """

        # reset game
        obs = self.env.reset()

        # fire = start the game
        obs, reward, done, info = self.env.step(1)
        obs = self.encode_obs(obs)  # (84, 84, 1)

        self.lives = self.env.unwrapped.ale.lives() # 5
        self.rewards = []
        self.dones = []

        # fill whole stack with current frame
        self.obs[..., 0:] = obs
        self.obs[..., 1:] = obs
        self.obs[..., 2:] = obs
        self.obs[..., 3:] = obs

        return self.obs

    def encode_obs(self, obs):
        """ Convert one frame to gray scale, resize, crop, normalise """
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 110), interpolation=cv2.INTER_AREA)
        obs = obs[17:101, :] # crop playing area
        obs = obs.reshape(84, 84, 1).astype(np.float32)
        obs = obs / 255.
        return obs

    def check_if_done(self, done, reward):
        """ check whether episode terminated """
        if 'Pong' in self.env_name:
            if reward < 0.0:
                return True
        elif 'Breakout' in self.env_name:
            if self.lives > self.env.unwrapped.ale.lives():
                return True
        return done

    def reward_shaping(self, reward):
        if 'Pong' in self.env_name:
            return (reward + 1.0/self.n_frames)
        return reward

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
        total_reward = 0.
        done = False
        obs = None

        # number of frames per step
        for i in range(self.n_frames):

            prev_obs = obs
            obs, reward, done, info = self.env.step(action)
            reward = self.reward_shaping(reward)

            if i == 0:
                prev_obs = obs

            # sum of rewards from each frame
            total_reward += reward

            # if a life is lost the episode ends
            done = self.check_if_done(done, reward)

            # episode has terminated
            if done:
                break

        # Take the maximum of each pixel of the last two frames, which is
        # important for the Breakout game, since one frame alone
        # does not contain the full information of the observation.
        obs = np.max(np.array([prev_obs, obs]), axis=0)

        # Add the maximum of last two frames to the 4-frame stack.
        # Taking the maximum avoids flickering images of Atari
        obs = self.encode_obs(obs)
        self.obs[..., -1:] = obs

        return self.obs, total_reward, done, None

    def render(self):
        #matplotlib.use('TkAgg')
        #plt.imshow(self.obs[:,:,-1], cmap=cm.gray, vmin=0., vmax=1.)
        #plt.show()
        self.env.render()

    def close(self):
        self.env.close()

class Maze(object):
    """ Environmental wrapper for custom 2D mazes

    The API follows the OpenAI gym API.
    """
    def __init__(self, layout, max_steps, entry, goal, reward):
        self.max_steps = max_steps
        self.n_actions = 4
        self.d_observation = 2
        self.entry = entry
        self.goal = goal
        self.reward = reward
        self.layout = np.array(layout, dtype=np.int)
        self.n_observations = self.layout.shape[0] * self.layout.shape[1]
        validr, validc = np.nonzero(layout)
        self.valid_positions = sorted(set(zip(validr, validc)))

    def reset(self):
        """ reset environment """
        self.n_steps = 1
        self.position = self.entry
        self.visited = set({self.position})
        self.goal_achieved = False
        return np.array(self.position)

    def get_reward(self):
        """ returns reward """
        if (self.position == self.goal) and not self.goal_achieved:
            reward = self.reward
            self.goal_achieved = True # get reward only once
        else:
            reward = 0.0
        return reward

    def done(self):
        """ returns flag if episode teminated """
        done = self.max_steps <= self.n_steps
        return done

    def step(self, a):
        """ make a step in the environment """
        if a > self.n_actions:
            raise Exception('Invalid action')

        self.n_steps += 1
        # left, right, up, down
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        new_pos_0 = self.position[0] + moves[a][0]
        new_pos_1 = self.position[1] + moves[a][1]

        if (new_pos_0, new_pos_1) in self.valid_positions:
            self.position = (new_pos_0, new_pos_1)

        self.visited.add(self.position)
        reward = self.get_reward()
        done = self.done()
        return np.array(self.position), reward, done, 0.

    def render(self):
        print(self.__repr__())

    def __repr__(self):
        s = []
        for i in range(len(self.layout)):
            for j in range(len(self.layout[0])):
                if (i, j) == self.position:
                    s.append('@')
                else:
                    s.append('.' if self.layout[i, j] else '#')
            s.append('\n')

        return ''.join(s)

    def n_actions(self):
        return self.n_actions

    def d_observation(self):
        return self.d_observation

    def close(self):
        pass

    def animate(self, obs_buffer):
        """ animate agent in maze environment """
        fig = plt.figure()
        ax = plt.gca()
        plt.grid(True)
        nrows, ncols = len(self.layout), len(self.layout[0])
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ims = []
        for row, col in obs_buffer:
            canvas = np.copy(self.layout).astype(np.float)
            canvas[int(row), int(col)] = 0.7
            canvas[self.goal] = 0.4
            img = plt.imshow(canvas, cmap=cm.gray, vmin=0., vmax=1.)
            ims.append([img])

        ani = animation.ArtistAnimation(fig, ims, interval=300, repeat_delay=1000)
        return ani

    def maze_canvas(self):
        """ return canvas of maze environment """
        canvas = np.copy(self.layout).astype(np.float)
        canvas[self.entry] = 0.8
        canvas[self.goal] = 0.4
        return canvas

    def render_maze(self, title='maze', obs_batch=None, fpath=None):
        """ render maze environment """
        fig = plt.figure()
        ax = plt.gca()
        plt.grid(True)
        plt.title(title)
        nrows, ncols = len(self.layout), len(self.layout[0])
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        canvas = self.maze_canvas()

        if obs_batch is not None:
            for obs_list in obs_batch:
                obs_list = np.vstack(obs_list)
                x = obs_list[:,1]
                y = obs_list[:,0]
                plt.plot(x, y , '-', lw=2)

        plt.imshow(canvas, cmap=cm.gray, vmin=0., vmax=1.)
        if fpath!=None:
            plt.savefig(fpath, dpi=300)
        plt.show()
        plt.close()

def make_env(env_string, max_steps, reward=10, variation=False):
    """ make maze environment """
    match = re.match('empty_maze_(\d+)_(\d+)', env_string)
    if match:
        return make_empty_maze(int(match.group(1)), int(match.group(2)), max_steps, reward)

    match = re.match('four_rooms_maze', env_string)
    if match:
        return make_four_rooms_maze(max_steps, reward, variation)

    match = re.match('custom_maze', env_string)
    if match:
        return make_custom_maze(max_steps, reward)
    return None

def make_empty_maze(h, w, max_steps, reward):
    """ Empty maze environment """
    layout = np.ones(shape=(h,w), dtype=np.int)
    layout[0,:] = 0
    layout[-1,:] = 0
    layout[:, 0] = 0
    layout[:, -1] = 0
    entry = (1,1)
    goal = (h-2,w-2)
    return Maze(layout, max_steps, entry, goal, reward)

def make_four_rooms_maze(max_steps, reward, variation=False):
    """ Four rooms maze environment """
    layout = np.ones(shape=(13,13), dtype=np.int)
    layout[0,:] = 0
    layout[-1,:] = 0
    layout[:, 0] = 0
    layout[:, -1] = 0
    layout[:3, 6] = 0
    layout[4:10, 6] = 0
    layout[-2:, 6] = 0
    layout[6,:2] = 0
    layout[6, 3:7] = 0
    layout[7, 6:9] = 0
    layout[7, -3:] = 0
    if variation:
        layout = np.transpose(layout)
    entry = (1,1)
    goal = (9,9)
    return Maze(layout, max_steps, entry, goal, reward)

def make_custom_maze(max_steps, reward):
    """ Custom maze environment """
    layout = np.ones(shape=(7,7), dtype=np.int)
    layout[0,:] = 0
    layout[-1,:] = 0
    layout[:, 0] = 0
    layout[:, -1] = 0
    layout[0:3, 6] = 0
    entry = (2,2)
    goal = (0,4)
    return Maze(layout, max_steps, entry, goal, reward)

if __name__ == '__main__':
    matplotlib.use('TkAgg')
    env = make_env('four_rooms_maze', 20)
    env.render_maze()



