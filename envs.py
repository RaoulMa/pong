import numpy as np
import re
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import sys

class Maze:
    """ class for creating maze environments """
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

    match = re.match('two_rooms_maze', env_string)
    if match:
        return make_two_rooms_maze(max_steps, reward)

    match = re.match('four_rooms_maze', env_string)
    if match:
        return make_four_rooms_maze(max_steps, reward, variation)

    match = re.match('cheese_maze', env_string)
    if match:
        return make_cheese_maze(max_steps, reward)

    match = re.match('custom_maze', env_string)
    if match:
        return make_custom_maze(max_steps, reward)
    return None

def make_empty_maze(h, w, max_steps, reward):
    layout = np.ones(shape=(h,w), dtype=np.int)
    layout[0,:] = 0
    layout[-1,:] = 0
    layout[:, 0] = 0
    layout[:, -1] = 0
    entry = (1,1)
    goal = (h-2,w-2)
    return Maze(layout, max_steps, entry, goal, reward)

def make_cheese_maze(max_steps, reward):
    layout = np.ones(shape=(10,10), dtype=np.int)
    layout[0,:] = 0
    layout[-1,:] = 0
    layout[:, 0] = 0
    layout[:, -1] = 0
    layout[:6, 3] = 0
    layout[-6:, 6] = 0
    entry = (1,1)
    goal = (2,5)
    return Maze(layout, max_steps, entry, goal, reward)


def make_two_rooms_maze(max_steps, reward):
    layout = np.ones(shape=(7,7), dtype=np.int)
    layout[0,:] = 0
    layout[-1,:] = 0
    layout[:, 0] = 0
    layout[:, -1] = 0
    layout[:3, 3] = 0
    layout[4:, 3] = 0
    entry = (1,1)
    goal = (5,5)
    return Maze(layout, max_steps, entry, goal, reward)

def make_four_rooms_maze(max_steps, reward, variation=False):
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



