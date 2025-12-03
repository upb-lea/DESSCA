from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, \
    Concatenate, LeakyReLU
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from gym.wrappers import FlattenObservation
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
import gym_electric_motor as gem
from gym_electric_motor.reference_generators import MultipleReferenceGenerator, ConstReferenceGenerator
from gym_electric_motor.physical_systems import ConstantSpeedLoad
from gym.core import Wrapper
from gym.spaces import Box, Tuple
import h5py
from pathlib import Path
import os
import sys
sys.path.append("../..")
from dessca import dessca_model

from multiprocessing import Pool


use_dessca = False


# globals:
i_n = 230
i_lim = 270
U_dc = 350
r_s = 17.932e-3
psi_p = 65.65e-3
l_d = 0.37e-3
l_q = 1.2e-3
p = 3
omega_lim = 12e3 * 2 * np.pi / 60

memory_buffer_size = 50000

def _ref_pdf(X):
    # states that must be available:
    # i_d, i_q, omega, cos_epsilon, sin_epsilon, u_d-1, u_q-1, i_d^*, i_q^*
    # actual states: i_d, i_q, omega, epsilon, i_d^*, i_q^*
    i_d = X[0] * i_lim
    i_q = X[1] * i_lim

    omega = X[2] * omega_lim # box constraints
    #epsilon = X[3] * np.pi # box constraints
    i_d_ref = X[4] * i_lim
    i_q_ref = X[5] * i_lim

    # plant
    # i_d
    _i_d_upper = np.less(i_d, np.clip((- psi_p + U_dc / (np.sqrt(3) * p * np.abs(omega))) / l_d, None, 0))
    _i_d_lower = np.greater(i_d, np.clip((- psi_p - U_dc / (np.sqrt(3) * p * np.abs(omega))) / l_d, - i_n, None))
    _i_d_allow = np.logical_and(_i_d_lower, _i_d_upper)
    # i_q
    _i_q_c_max = np.sqrt((U_dc / (l_d * np.sqrt(3) * p * np.abs(omega))) ** 2 - ((l_d * i_d + psi_p) / l_q) ** 2)
    _i_q_upper = np.less(i_q, np.minimum(_i_q_c_max, np.sqrt(np.clip(i_n ** 2 - i_d ** 2, 0, None))))
    _i_q_lower = np.greater(i_q, np.maximum(- _i_q_c_max, -np.sqrt(np.clip(i_n ** 2 - i_d ** 2, 0, None))))
    _i_q_allow = np.logical_and(_i_q_lower, _i_q_upper)
    _plant_allow = np.logical_and(_i_d_allow, _i_q_allow).astype(float)

    # reference
    # i_d_ref
    _i_d_upper = np.less(i_d_ref, np.clip((- psi_p + U_dc / (np.sqrt(3) * p * np.abs(omega))) / l_d, None, 0))
    _i_d_lower = np.greater(i_d_ref, np.clip((- psi_p - U_dc / (np.sqrt(3) * p * np.abs(omega))) / l_d, -i_n, None))
    _i_d_ref_allow = np.logical_and(_i_d_lower, _i_d_upper)
    # i_q_ref
    _i_q_c_max = np.sqrt((U_dc / (l_d * np.sqrt(3) * p * np.abs(omega))) ** 2 - ((l_d * i_d_ref + psi_p) / l_q) ** 2)
    _i_q_upper = np.less(i_q_ref, np.minimum(_i_q_c_max, np.sqrt(np.clip(i_n ** 2 - i_d_ref ** 2, 0, None))))
    _i_q_lower = np.greater(i_q_ref, np.maximum(- _i_q_c_max, -np.sqrt(np.clip(i_n ** 2 - i_d_ref ** 2, 0, None))))
    _i_q_ref_allow = np.logical_and(_i_q_lower, _i_q_upper)
    _reference_allow = np.logical_and(_i_d_ref_allow, _i_q_ref_allow)

    _state_filter = np.logical_and(_plant_allow, _reference_allow).astype(float)

    return _state_filter

class AppendLastActionWrapper(Wrapper):

    def __init__(self, environment, agent_idx):
        super().__init__(environment)
        self.observation_space = Tuple((Box(
            np.concatenate((environment.observation_space[0].low[0:3],
                            [-1, -1],
                            environment.observation_space[0].low[4:-1],
                            environment.action_space.low)),
            np.concatenate((environment.observation_space[0].high[0:3],
                            [1, 1],
                            environment.observation_space[0].high[4:-1],
                            environment.action_space.high))
        ), environment.observation_space[1]))

        self.STATE = None
        self.REWARD = []
        self.HISTORY = []

        self.episode_count = 0
        self.agent_idx = agent_idx
        if use_dessca:
            self.folder_name = "Dessca"
        else:
            self.folder_name = "Uniform"

        if use_dessca:
            state_constraints = [[-1, 1],
                                 [-1, 1],
                                 [-1, 1],
                                 [-1, 1],
                                 [-1, 1],
                                 [-1, 1]]
            self.dessca_model = dessca_model(box_constraints=state_constraints,
                                             reference_pdf=_ref_pdf,
                                             state_names=["i_d",
                                                          "i_q",
                                                          "omega",
                                                          "epsilon",
                                                          "i_d^*",
                                                          "i_q^*"],
                                             buffer_size=memory_buffer_size)

    def step(self, action):

        (state, ref), rew, term, info = self.env.step(action)
        state[2] += np.random.normal(0, 0.0001) # measurement noise
        ref[0] += np.random.normal(0, 0.01)
        ref[1] += np.random.normal(0, 0.01)
        state = np.concatenate((state[0:3], [np.cos(state[3] * np.pi), np.sin(state[3] * np.pi)], action))

        i_d = state[0]
        i_q = state[1]
        omega = state[2]
        epsilon = np.arctan2(state[4], state[3]) / np.pi
        i_d_ref = ref[0]
        i_q_ref = ref[1]
        r_d = (np.sqrt(np.abs(i_d_ref - i_d) / 2) + ((i_d_ref - i_d) / 2) ** 2) / 2
        r_q = (np.sqrt(np.abs(i_q_ref - i_q) / 2) + ((i_q_ref - i_q) / 2) ** 2) / 2

        i_total = np.sqrt(i_d ** 2 + i_q ** 2)
        if i_total > i_n / i_lim: # Danger Zone !
            rew = (1 - (i_total - i_n / i_lim) / (1 - i_n / i_lim)) * (1 - 0.9) - (1 - 0.9)
        else:
            rew = (2 - r_d - r_q) / 2 * (1 - 0.9)

        if use_dessca:
            self.dessca_model.update_coverage_pdf(data=np.transpose([i_d, i_q, omega, epsilon, i_d_ref, i_q_ref]))

        if term:
            rew = -1

        self.STATE.append(np.concatenate((state, ref)).tolist())
        self.REWARD.append(rew)

        state[3] *= 0.1
        state[4] *= 0.1

        return (state, ref), rew, term, info

    def reset(self, **kwargs):

        if self.STATE is not None:
            self.HISTORY.append(np.mean(self.REWARD))
            Path(self.folder_name + "/" + self.folder_name + "_" + str(self.agent_idx)).mkdir(parents=True, exist_ok=True)
            with h5py.File(
                    self.folder_name
                    + "/" + self.folder_name
                    + "_" + str(self.agent_idx)
                    + "/" + "training"
                    + "_" + str(self.episode_count)
                    + ".hdf5", "w") as f:
                lim = f.create_dataset("limits", data=np.concatenate((self.limits[0:3], [1, 1],
                                                                      [self.env.physical_system.limits[7],
                                                                       self.env.physical_system.limits[7]],
                                                                      self.limits[0:2]
                                                                      )))
                obs = f.create_dataset("observations", data=self.STATE)
                rew = f.create_dataset("rewards", data=self.REWARD)
                history = f.create_dataset("history", data=self.HISTORY)
                self.episode_count += 1

        state, ref = self.env.reset()

        if not use_dessca:
            eps_0 = np.random.uniform(-1, 1) * np.pi
            omega_0 = np.random.uniform(-1, 1) * self.env.physical_system.limits[0]
            psi_p = self.env.physical_system.electrical_motor.motor_parameter["psi_p"]
            l_d = self.env.physical_system.electrical_motor.motor_parameter["l_d"]
            l_q = self.env.physical_system.electrical_motor.motor_parameter["l_q"]
            p = self.env.physical_system.electrical_motor.motor_parameter["p"]
            u_dc = self.env.physical_system.limits[-1]
            dc_link_d = u_dc / (np.sqrt(3) * l_d * np.abs(omega_0 * p))
            i_d_upper = np.clip(- psi_p / l_d + dc_link_d, None, 0)
            i_d_lower = np.clip(- psi_p / l_d - dc_link_d, -i_n, None)
            i_d_0 = np.random.uniform(i_d_lower, i_d_upper)
            i_q_upper = np.clip(np.sqrt((u_dc / (np.sqrt(3) * omega_0 * p * l_q)) ** 2 -
                                        (l_d / l_q * (i_d_0 + psi_p / l_d)) ** 2), None, np.sqrt(i_n ** 2 - i_d_0 ** 2))
            i_q_lower = np.clip(- i_q_upper, - np.sqrt(i_n ** 2 - i_d_0 ** 2), None)
            i_q_0 = np.random.uniform(i_q_lower, i_q_upper)

            i_d_ref = np.random.uniform(i_d_lower, i_d_upper)
            i_q_upper_ref = np.clip(np.sqrt((u_dc / (np.sqrt(3) * omega_0 * p * l_q)) ** 2 -
                                        (l_d / l_q * (i_d_ref + psi_p / l_d)) ** 2), None, np.sqrt(i_n ** 2 - i_d_ref ** 2))
            i_q_lower_ref = np.clip(- i_q_upper, - np.sqrt(i_n ** 2 - i_d_ref ** 2), None)
            i_q_ref = np.random.uniform(i_q_lower_ref, i_q_upper_ref)
        else:
            # i_d, i_q, omega, epsilon, i_d^*, i_q^*
            dessca_state = self.dessca_model.sample_optimally()
            i_d_0 = dessca_state[0] * self.env.physical_system.limits[2]
            i_q_0 = dessca_state[1] * self.env.physical_system.limits[2]
            omega_0 = dessca_state[2] * self.env.physical_system.limits[0]
            eps_0 = dessca_state[2] * np.pi
            i_d_ref = dessca_state[4] * self.env.physical_system.limits[2]
            i_q_ref = dessca_state[5] * self.env.physical_system.limits[2]

        self.env.physical_system._ode_solver.set_initial_value(np.array([omega_0, i_d_0, i_q_0, eps_0]))
        self.env.reference_generator._sub_generators[0]._reference_value = i_d_ref / self.env.physical_system.limits[2]
        self.env.reference_generator._sub_generators[1]._reference_value = i_q_ref / self.env.physical_system.limits[2]
        self.env.physical_system.mechanical_load._omega = omega_0 / self.env.physical_system.limits[0]

        state[2] = omega_0 / self.env.physical_system.limits[0]
        state[3] = np.cos(eps_0)
        state[0] = i_d_0 / self.env.physical_system.limits[2]
        state[1] = i_q_0 / self.env.physical_system.limits[2]
        state = np.append(state, np.sin(eps_0))
        ref = [i_d_ref / self.env.physical_system.limits[2], i_q_ref / self.env.physical_system.limits[2]]

        state = np.concatenate((state, np.zeros(self.env.action_space.shape)))

        # i_d, i_q, omega, cos_epsilon, sin_epsilon, u_d-1, u_q-1, i_d^*, i_q^*
        self.STATE = [np.concatenate((state, ref)).tolist()]
        self.REWARD = []

        state[3] *= 0.1
        state[4] *= 0.1

        return state, ref


def train_agent(agent_idx):
    tf.config.set_visible_devices([], 'GPU')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    d_generator = ConstReferenceGenerator('i_sd', 0)
    q_generator = ConstReferenceGenerator('i_sq', 0)
    rg = MultipleReferenceGenerator([d_generator, q_generator])

    motor_parameter = dict(
        r_s=r_s, l_d=l_d, l_q=l_q, psi_p=psi_p, p=p, j_rotor=0.06
    )

    limit_values = dict(
        i=i_lim,
        omega=omega_lim,
        u=U_dc,
    )

    nominal_values = {key: 0.7 * limit for key, limit in limit_values.items()}

    env = gem.make(
        'PMSMCont-v1',
        load=ConstantSpeedLoad(omega_fixed=1000 * np.pi / 30),
        control_space='dq',
        ode_solver='scipy.solve_ivp', solver_kwargs={},
        reference_generator=rg,
        reward_weights={'i_sq': 0.5, 'i_sd': 0.5},
        reward_power=0.5,
        observed_states=['i_sd', 'i_sq'],
        tau=1e-4,
        dead_time=True,
        u_sup=U_dc,
        motor_parameter=motor_parameter,
        limit_values=limit_values,
        nominal_values=nominal_values,
        state_filter=['i_sd', 'i_sq', 'omega', 'epsilon'],
        #visualization=MotorDashboard(plots=['i_sd', 'i_sq', 'action_0', 'action_1', 'mean_reward']), visu_period=1,
    )
    env = AppendLastActionWrapper(env, agent_idx=agent_idx)
    env = FlattenObservation(env)

    nb_actions = env.action_space.shape[0]

    window_length = 1
    actor = Sequential()
    actor.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
    actor.add(Dense(64, activation='linear'))
    actor.add(LeakyReLU(alpha=0.1))
    actor.add(Dense(64, activation='linear'))
    actor.add(LeakyReLU(alpha=0.1))
    actor.add(Dense(64, activation='linear'))
    actor.add(LeakyReLU(alpha=0.1))
    # The network output fits the action space of the env
    actor.add(Dense(nb_actions,
                    activation='tanh',
                    #kernel_regularizer=regularizers.l2(1e-2))
              ))

    # Define another artificial neural network to be used within the agent as critic
    # note that this network has two inputs
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(window_length,) + env.observation_space.shape, name='observation_input')
    # (using keras functional API)
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(128, activation='linear')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(128, activation='linear')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(128, activation='linear')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(128, activation='linear')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(1, activation='linear')(x)
    critic = Model(inputs=(action_input, observation_input), outputs=x)
    print(critic.summary())

    # Define a memory buffer for the agent, allows to learn from past experiences
    memory = SequentialMemory(
        limit=memory_buffer_size,
        window_length=window_length
    )

    # Create a random process for exploration during training
    # this is essential for the DDPG algorithm
    random_process = OrnsteinUhlenbeckProcess(
        theta=10,
        mu=0.0,
        sigma=1,
        dt=env.physical_system.tau,
        sigma_min=0.01,
        n_steps_annealing=290000,
        size=2
    )

    # Create the agent for DDPG learning
    agent = DDPGAgent(
        # Pass the previously defined characteristics
        nb_actions=nb_actions,
        actor=actor,
        critic=critic,
        critic_action_input=action_input,
        memory=memory,
        random_process=random_process,

        # Define the overall training parameters
        nb_steps_warmup_actor=2000,
        nb_steps_warmup_critic=1000,
        target_model_update=0.25,
        gamma=0.9,
        batch_size=16,
        memory_interval=1
    )

    # Compile the function approximators within the agent (making them ready for training)
    # Note that the DDPG agent uses two function approximators, hence we define two optimizers here
    agent.compile([Adam(lr=5e-6), Adam(lr=5e-4)])

    # Start training for 75 k simulation steps
    agent.fit(
        env,
        nb_steps=400000,
        nb_max_start_steps=0,
        nb_max_episode_steps=100,
        visualize=True,
        action_repetition=1,
        verbose=2,
        log_interval=10000,
        callbacks=[],
    )

    if use_dessca:
        folder_name = "Dessca"
    else:
        folder_name = "Uniform"

    agent.save_weights(filepath=folder_name +
                                "/" + folder_name +
                                "_" + str(agent_idx) +
                                "/" + folder_name +
                                "_weights.hdf5",
                       overwrite=True)


if __name__ == '__main__':
    with Pool() as p:
        p.map(train_agent, range(50))