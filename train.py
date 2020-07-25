# -*- encoding utf8 -*-

import random
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from env import *
from network import QNetwork


# random.seed(71)

class ExperienceReplayd:

    def __init__(self, memorysize):
        self.memorysize = memorysize
        self.memory = list()

    def add(self, experience):
        if len(self.memory) == self.memorysize:
            _ = self.memory.pop(0)
        self.memory.append(experience)

    def get(self, ind):
        return self.memory[ind]

    def sample(self, num):
        current_cnt = self.get_cnt()
        if current_cnt < num:
            num = current_cnt
        sample = random.sample(self.memory, num)
        return sample

    def showexperience(self):
        print(self.memory)

    def get_cnt(self):
        return len(self.memory)


def main(env, N_EPISODE=100, MAX_STEPS=300, SUCESS_STEP=200, STOP_MAX_EPISODE=5, EPSILONE_START=1.0, EPSILONE_END=0.01, EPSILONE_DECAY=0.001, GAMMA=0.99, WARMUP=10, MEMORYSIZE=10000, BATCHSIZE=32, PLOT=True):

    def _update_target_parameter(_main_qn, _target_qn):
        _target_qn.model.set_weights(_main_qn.model.get_weights())
        return _target_qn

    def _get_egreedy_actions(epsilon, actions):

        if random.random() < epsilon:
            action_num = actions.shape[1]
            action = random.choice([a for a in range(action_num)])
        else:
            action = np.argmax(actions[0])

        return action

    def _update_parameter(_main_qn, _target_qn, _memory, BATCHSIZE):
        memories = _memory.sample(BATCHSIZE)
        states = np.zeros((BATCHSIZE, statesize))
        targets = np.zeros((BATCHSIZE, actionsize))

        for m_ind, (_state, _action, _reward, _n_state) in enumerate(memories):
            _state_arr = _state.reshape(1, statesize)
            _n_state_arr = _n_state.reshape(1, statesize)
            states[m_ind] = _state_arr

            if not (_n_state_arr == np.zeros((1, statesize))).all(axis=1):
                target = _reward + GAMMA * \
                    np.amax(_target_qn.model.predict(_n_state_arr)[0])

            else:
                target = _reward

            targets[m_ind] = _main_qn.model.predict(_state_arr)
            targets[m_ind][_action] = target

        _main_qn.model.fit(states, targets, epochs=1, verbose=0)

        return _main_qn

    # Get Env parameter
    actionsize = env.get_action_space()
    statesize = env.get_observation_space()

    # Create Network
    main_qn = QNetwork(statesize, actionsize)
    target_qn = QNetwork(statesize, actionsize)

    # Create Memory
    memory = ExperienceReplayd(MEMORYSIZE)

    epsilon = EPSILONE_START

    # Dataholder for plotting
    if PLOT:
        history = {"episode": [], "step": []}

    # Repeat Episode
    total_step = 0
    success_episode = 0
    for i in range(N_EPISODE):
        print("Episode : " + str(i))
        # initialize Env
        state = env.reset()

        # Update target network parameter
        target_qn = _update_target_parameter(main_qn, target_qn)

        # Take actions
        for steps in range(MAX_STEPS):
            total_step += 1
            # Decay Epsilon
            epsilon = EPSILONE_END + \
                (EPSILONE_START - EPSILONE_END) * \
                np.exp(-EPSILONE_DECAY*total_step)
            print("Step : " + str(steps) + " Epsilon : " + str(epsilon))
            state_arr = state.reshape(1, statesize)
            actions = target_qn.model.predict(state_arr)
            choiced_action = _get_egreedy_actions(epsilon, actions)
            n_state, reward, done, info = env.step(choiced_action)

            if done:
                n_state = np.zeros(n_state.shape)

            if steps >= WARMUP:
                m = (state, choiced_action, reward, n_state)
                memory.add(m)

            if memory.get_cnt() > BATCHSIZE:
                main_qn = _update_parameter(
                    main_qn, target_qn, memory, BATCHSIZE)

            if done:
                print("Done:" + str(done))
                success_episode = 0
                break

            state = n_state

            if steps >= SUCESS_STEP - 1:
                success_episode += 1
                print("MAX EPISODE")
                break

        if PLOT:
            history["episode"].append(i)
            history["step"].append(steps)
            epi = history["episode"]
            stp = history["step"]
            plt.plot(epi, stp, 'b')
            plt.title('Max steps per episode')
            plt.legend()
            plt.savefig("plot.png")

        if success_episode >= STOP_MAX_EPISODE:
            print("SUCCESS!!")
            break


if __name__ == "__main__":
    env = CartPoleEnv(savegif=True)
    main(env)
