# -*- encoding utf8 -*- 

import gym

ENV = 'CartPole-v0' 

env = gym.make(ENV)
for i_episode in range(20):
    obsvn = env.reset()
    for i in range(10):
        env.render()
        print(obsvn)
        action = env.action_space.sample()
        obsvn, reward, done, info = env.step(action)
        if done:
            print("Episode end")
            break

env.close()