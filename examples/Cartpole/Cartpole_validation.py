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

import sys
sys.path.append("../..")
from dessca import dessca_model

state_constraints = [[  - 1.0,    1.0], #+- 2.4
                     [     -7,      7],
                     [ -np.pi, +np.pi],
                     [    -10,     10]]

state_low = np.array(state_constraints)[:, 0]
state_high = np.array(state_constraints)[:, 1]

delta_v = 0.15
tau = 0.02
initial_states = []
for _ in range(1000):
    while True:
        state = np.random.uniform(low=state_low, high=state_high)
        if np.abs(state[3]) < 1:
            if state[1] > 0:
                if np.sqrt((1 - state[0]) / tau * delta_v) > state[1]:
                    break
            if state[1] < 0:
                if -np.sqrt((1 + state[0]) / tau * delta_v) < state[1]:
                    break
    initial_states.append(state)
initial_states = np.transpose(np.array(initial_states))
data = {"initial_states": initial_states.tolist()}
#
# with open("ValidationInits.json", 'w') as json_file:
#     data = json.dump(data, json_file)
#
# sys.exit()

use_dessca = True

with open("ValidationInits.json", 'r') as json_file:
    data = json.load(json_file)
    initial_states = np.copy(data["initial_states"])


class cartpole_reset_wrapper(Wrapper):

    def _ref_pdf(self, X):
        x0 = X[0]
        x1 = X[1]
        x2 = X[2]
        x3 = X[3]

        init_pdf = np.logical_and(
            np.greater(np.minimum(np.sqrt((self.state_high[0] - x0) / self.env.tau * self.delta_v), self.state_high[1]), x1),
            np.less(np.maximum(-np.sqrt((self.state_high[0] + x0) / self.env.tau * self.delta_v), -self.state_high[1]), x1),
            np.less(np.abs(x3), 1)
        ).astype(float)

        return init_pdf

    def __init__(self, environment, try_nb):
        super().__init__(environment)

        self.try_nb = try_nb
        self.state_saver = None
        self._episode_reward_list = []

        state_constraints = [[  - 1.0,    1.0], #+- 2.4
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

        self._init_state_iterator = 0

    def step(self, action):
        state, reward, done, _ = self.env.step(action)  # state here is observation

        self.state_saver = np.append(self.state_saver, np.reshape(self.env.state, (4, 1)), axis=1)
        self._reward_list.append(reward)
        self.k += 1

        if self.k == 200:
            done = True

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
        self.env.state = initial_states[:, self._init_state_iterator]
        self._init_state_iterator += 1

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

        with open('./' + experiment + "/" + experiment + '_Validation_' + str(self.try_nb) + '.json', 'w') as json_file:
            json.dump(save_dict, json_file)

for i in range(50):

    env = CartPoleEnv()
    env = cartpole_reset_wrapper(env, i)

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

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(eps=0.1),
                                  attr='eps',
                                  value_max=0.0,
                                  value_min=0.01,
                                  value_test=0,
                                  nb_steps=25000)

    memory = SequentialMemory(
        limit=5000,
        window_length=1
    )

    agent = DQNAgent(model=model,
                     nb_actions=nb_actions,
                     gamma=0.99,
                     batch_size=16,
                     memory=memory,
                     memory_interval=1,
                     policy=policy,
                     train_interval=1,
                     target_model_update=100,
                     enable_double_dqn=False)

    agent.compile(Adam(lr=1e-4))

    if use_dessca:
        experiment = "Dessca"
    else:
        experiment = "Uniform"

    agent.load_weights(filepath="./" + experiment + "/" + experiment + "_weights_" + str(i) + ".hdf5")

    history = agent.test(env,
                         nb_episodes=1000,
                         action_repetition=1,
                         verbose=0,
                         visualize=False,
                         nb_max_episode_steps=500)

    print(f"Done with Agent {i} / 49")
    env.close()