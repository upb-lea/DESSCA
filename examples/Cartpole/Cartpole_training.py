from gym.core import Wrapper
import numpy as np
import json

from Custom_Cartpole import CartPoleEnv

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

from multiprocessing import Pool
import tensorflow as tf

import os
import sys
sys.path.append("../..")
from dessca import dessca_model


use_dessca = False


class cartpole_reset_wrapper(Wrapper):

    def _ref_pdf(self, X):
        x0 = X[0]
        x1 = X[1]
        x2 = X[2]
        x3 = X[3]
        init_pdf = np.logical_and(
            np.logical_and(
                np.greater(
                    np.minimum(np.sqrt((self.state_high[0] - x0) / self.env.tau * self.delta_v), self.state_high[1]),
                    x1),
                np.less(
                    np.maximum(-np.sqrt((self.state_high[0] + x0) / self.env.tau * self.delta_v), -self.state_high[1]),
                    x1),
            ),
            np.less(np.abs(x3), 1)
        ).astype(float)
        return init_pdf

    def __init__(self, environment, try_nb):
        super().__init__(environment)

        self.try_nb = try_nb
        self.state_saver = None
        self._episode_reward_list = []


        state_constraints = [[  - 1,    1], # +- 2.4
                             [     -7,      7],
                             [ -np.pi, +np.pi],
                             [    -10,     10]]

        self.state_low = np.array(state_constraints)[:, 0]
        self.state_high = np.array(state_constraints)[:, 1]

        if use_dessca:
            self.dessca_model = dessca_model(box_constraints=state_constraints,
                                             reference_pdf=self._ref_pdf,
                                             state_names=["x", "v", "epsilon", "omega"])

        self.delta_v = 0.15

    def step(self, action):
        state, reward, done, _ = self.env.step(action)  # state here is observation

        self.state_saver = np.append(self.state_saver, np.reshape(self.env.state, (4, 1)), axis=1)
        self._reward_list.append(reward)
        self.k += 1

        if use_dessca:
            self.dessca_model.update_coverage_pdf(data=np.transpose([state]))

        done = False
        if np.any(np.abs(self.env.state) > self.state_high):
            reward = -1
            done = True

        norm_state = [(self.env.state[0] - self.state_low[0]) / (self.state_high[0] - self.state_low[0]) * 2 - 1,
                      (self.env.state[1] - self.state_low[1]) / (self.state_high[1] - self.state_low[1]) * 2 - 1,
                      np.cos(self.env.state[2]),
                      np.sin(self.env.state[2]),
                      (self.env.state[3] - self.state_low[3]) / (self.state_high[3] - self.state_low[3]) * 2 - 1,
                      ]

        return norm_state, reward, done, _

    def reset(self, **kwargs):

        if hasattr(self, "_reward_list"):
            self._episode_reward_list.append(self._reward_list)
        self._reward_list = []

        self.k = 0

        self.env.reset()
        if use_dessca:
            self.env.state = self.dessca_model.sample_optimally()
        else:
            while True:
                self.env.state = np.random.uniform(low=self.state_low,
                                                   high=self.state_high)
                if np.abs(self.env.state[3]) < 1:
                    if self.env.state[1] > 0:
                        if np.sqrt((self.state_high[0] - self.env.state[0]) / self.env.tau * self.delta_v) > self.env.state[1]:
                            break
                    if self.env.state[1] < 0:
                        if -np.sqrt((self.state_high[0] + self.env.state[0]) / self.env.tau * self.delta_v) < self.env.state[1]:
                            break

        if self.state_saver is None:
            self.state_saver = np.reshape(np.copy(self.env.state), (4, 1))
        else:
            self.state_saver = np.append(self.state_saver, np.reshape(self.env.state, (4, 1)), axis=1)


        norm_state = [(self.env.state[0] - self.state_low[0]) / (self.state_high[0] - self.state_low[0]) * 2 - 1,
                      (self.env.state[1] - self.state_low[1]) / (self.state_high[1] - self.state_low[1]) * 2 - 1,
                      np.cos(self.env.state[2]),
                      np.sin(self.env.state[2]),
                      (self.env.state[3] - self.state_low[3]) / (self.state_high[3] - self.state_low[3]) * 2 - 1,
                      ]

        return norm_state

    def close(self):
        if hasattr(self, "_reward_list"):
            self._episode_reward_list.append(self._reward_list)
        self.env.close()

        save_dict = {
            "state_history": self.state_saver.tolist(),
            "reward_history": self._episode_reward_list
        }
        if use_dessca:
            experiment = "Dessca"
        else:
            experiment = "Uniform"

        with open('./' + experiment + "/" + experiment + '_' + str(self.try_nb) + '.json', 'w') as json_file:
            json.dump(save_dict, json_file)


def train_agent(idx):
    tf.config.set_visible_devices([], 'GPU')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    env = CartPoleEnv()
    env = cartpole_reset_wrapper(env, idx)

    obs_space = (env.observation_space.shape[0] + 1,)

    nb_actions = env.action_space.n
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + obs_space))
    model.add(Dense(200, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(200, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(200, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(nb_actions, activation='linear', ))

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(eps=0.5),
                                  attr='eps',
                                  value_max=0.5,
                                  value_min=0.01,
                                  value_test=0,
                                  nb_steps=40000)

    memory = SequentialMemory(
        limit=10000,
        window_length=1
    )

    agent = DQNAgent(model=model,
                     nb_actions=nb_actions,
                     gamma=0.99,
                     batch_size=32,
                     memory=memory,
                     memory_interval=1,
                     policy=policy,
                     train_interval=1,
                     target_model_update=0.001,
                     enable_double_dqn=False)

    agent.compile(Adam(lr=1e-3))

    agent.fit(
        env,
        nb_steps=100000,
        nb_max_start_steps=0,
        nb_max_episode_steps=200,
        visualize=False,
        action_repetition=1,
        verbose=2,
        log_interval=10000,
        callbacks=[],
    )

    if use_dessca:
        experiment = "Dessca"
    else:
        experiment = "Uniform"

    agent.save_weights(filepath="./" + experiment + "/" + experiment + "_weights_" + str(idx) + ".hdf5", overwrite=True)
    env.close()


if __name__ == '__main__':
    with Pool() as p:
        p.map(train_agent, range(50))
