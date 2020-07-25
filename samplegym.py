# -*- encoding utf8 -*-

# import gym
from env import *

ENV = 'CartPole-v0'

env = OpenAiEnv(ENV)
for i_episode in range(20):
    obsvn = env.reset()
    for i in range(10):
        env.render()
        print(obsvn)
        action = env.sample_action()
        print(action)
        obsvn, reward, done, info = env.step(action)
        if done:
            print("Episode end")
            break

env.close()
