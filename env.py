# -*- encoding utf8 -*-

import gym
from gym import wrappers
import numpy
import os


class Env:

    def __init__(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError


class OpenAiEnv(Env):

    def __init__(self, env, savegif=False, gifpath='./gif'):
        self.env = gym.make(env)
        self.savegif = savegif

        # save gif
        if self.savegif:
            if not os.path.exists(gifpath):
                os.mkdir(gifpath)
            self.env = wrappers.Monitor(
                self.env, gifpath, force=True, video_callable=lambda epi: epi % 5 == 0)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def sample_action(self):
        return self.env.action_space.sample()

    def get_action_space(self):
        return self.env.action_space.n

    def get_observation_space(self):
        return self.env.observation_space.shape[0]

    def close(self):
        self.env.close()


class CartPoleEnv(OpenAiEnv):

    def __init__(self, env='CartPole-v0', savegif=False, gifpath='./gif'):
        super().__init__(env, savegif, gifpath)
        self.totalsteps = 0

    def reset(self):
        self.totalsteps = 0
        if self.savegif:
            recorder = self.env.stats_recorder
            if recorder is not None:
                recorder.done = True
        return self.env.reset()

    def step(self, action):
        self.totalsteps += 1
        n_state, reward, done, info = self.env.step(action)

        if self.totalsteps <= 190:
            reward = 0
        else:
            reward = 1

        if done:
            reward = 0

        return n_state, reward, done, info


if __name__ == "__main__":
    ENV = 'CartPole-v0'
    env = OpenAiEnv(ENV)
