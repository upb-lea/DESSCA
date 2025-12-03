from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, \
    Concatenate, LeakyReLU
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
from gym_electric_motor.physical_systems import ExternalSpeedLoad
from gym.core import Wrapper
from gym.spaces import Box, Tuple
import h5py
from pathlib import Path
from multiprocessing import Pool


use_dessca = True


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
tau = 1e-4

rng = np.random.default_rng(42)
i_step = i_n / 8
i_dq_grid = np.array([[], []])
for d_idx in range(9):
    for q_idx in range(17):
        i_d_new = - d_idx * i_step * 0.999
        i_q_new = (q_idx - 8) * i_step * 0.999
        if np.sqrt(i_d_new ** 2 + i_q_new ** 2) < i_n:
            i_dq_grid = np.append(i_dq_grid, np.array([[i_d_new], [i_q_new]]), axis=1)
rng.shuffle(i_dq_grid, axis=1)

step_length_per_point = 300
grid_test_duration_steps = np.shape(i_dq_grid)[1] * step_length_per_point
grid_test_duration_time = grid_test_duration_steps * tau
acceleration_steps = 5000
accel_time = acceleration_steps * tau

last_state_hold_steps = 5000

test_duration = (grid_test_duration_steps + acceleration_steps) * 5 + last_state_hold_steps
max_return = test_duration * 0.1
print(max_return)
print()
print()
print()

def i_dq_validation_profile(k):

    _k = k % (grid_test_duration_steps + acceleration_steps)
    if k >= test_duration - last_state_hold_steps:
        i_d_ref = 0
        i_q_ref = 0
    elif _k < grid_test_duration_steps:
        _point_idx = _k // step_length_per_point
        i_d_ref = i_dq_grid[0, _point_idx] / i_lim
        i_q_ref = i_dq_grid[1, _point_idx] / i_lim
    else:
        i_d_ref = 0
        i_q_ref = 0

    return i_d_ref, i_q_ref

def speed_profile(t):

    niveau0 = 0.0
    niveau1 = 0.15
    niveau2 = 0.5

    if t < grid_test_duration_time:
        omega = niveau0 * omega_lim
    elif t < (grid_test_duration_time + accel_time):
        omega = ((t - grid_test_duration_time) * (niveau1 - niveau0) / accel_time + niveau0) * omega_lim

    elif t < 2 * grid_test_duration_time + accel_time:
        omega = niveau1 * omega_lim
    elif t < 2 * (grid_test_duration_time + accel_time):
        omega = ((t - 2 * grid_test_duration_time - accel_time) * (- niveau1 - niveau1) / accel_time + niveau1) * omega_lim

    elif t < 3 * grid_test_duration_time + 2 * accel_time:
        omega = - niveau1 * omega_lim
    elif t < 3 * (grid_test_duration_time + accel_time):
        omega = ((t - 3 * grid_test_duration_time - 2 * accel_time) * (niveau2 + niveau1) / accel_time - niveau1) * omega_lim

    elif t < 4 * grid_test_duration_time + 3 * accel_time:
        omega = niveau2 * omega_lim
    elif t < 4 * (grid_test_duration_time + accel_time):
        omega = ((t - 4 * grid_test_duration_time - 3 * accel_time) * (- niveau2 - niveau2) / accel_time + niveau2) * omega_lim

    elif t < 5 * grid_test_duration_time + 4 * accel_time:
        omega = - niveau2 * omega_lim
    elif t < 5 * (grid_test_duration_time + accel_time):
        omega = ((t - 5 * grid_test_duration_time - 4 * accel_time) * (niveau0 + niveau2) / accel_time - niveau2) * omega_lim

    else:
        omega = 0.0

    return omega


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

        self.time_idx = 0

        self.episode_count = 0
        self.agent_idx = agent_idx
        if use_dessca:
            self.folder_name = "Dessca"
        else:
            self.folder_name = "Uniform"

    def step(self, action):

        (state, ref), rew, term, info = self.env.step(action)
        self.time_idx += 1

        ref[0], ref[1] = i_dq_validation_profile(self.time_idx)
        state = np.concatenate((state[0:3], [np.cos(state[3] * np.pi), np.sin(state[3] * np.pi)], action))

        i_d = state[0]
        i_q = state[1]
        i_d_ref = ref[0]
        i_q_ref = ref[1]
        r_d = (np.sqrt(np.abs(i_d_ref - i_d) / 2) + ((i_d_ref - i_d) / 2) ** 2) / 2
        r_q = (np.sqrt(np.abs(i_q_ref - i_q) / 2) + ((i_q_ref - i_q) / 2) ** 2) / 2

        i_total = np.sqrt(i_d ** 2 + i_q ** 2)
        if i_total > i_n / i_lim: # Danger Zone !
            rew = (1 - (i_total - i_n / i_lim) / (1 - i_n / i_lim)) * (1 - 0.9) - (1 - 0.9)
        else:
            rew = (2 - r_d - r_q) / 2 * (1 - 0.9)

        if term:
            rew = -1

        self.STATE.append(np.concatenate((state, ref)).tolist())
        self.REWARD.append(rew)

        state[3] *= 0.1
        state[4] *= 0.1

        if self.time_idx == test_duration:
            self.reset()

        return (state, ref), rew, term, info

    def reset(self, **kwargs):

        if self.STATE is not None:
            self.HISTORY.append(np.mean(self.REWARD))
            Path(self.folder_name + "/" + self.folder_name + "_" + str(self.agent_idx)).mkdir(parents=True, exist_ok=True)
            with h5py.File(
                    self.folder_name
                    + "/" + self.folder_name
                    + "_" + str(self.agent_idx)
                    + "/" + "validation"
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

        i_d_0 = 0
        i_q_0 = 0
        eps_0 = 0
        omega_0 = 0
        i_d_ref = 0
        i_q_ref = 0

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

        self.time_idx = 0

        return state, ref


def test_agent(agent_idx):
    d_generator = ConstReferenceGenerator('i_sd', 0)
    q_generator = ConstReferenceGenerator('i_sq', 0)
    rg = MultipleReferenceGenerator([d_generator, q_generator])

    motor_parameter = dict(
        r_s=r_s, l_d=l_d, l_q=l_q, psi_p=psi_p, p=p, j_rotor=0.001
    )

    limit_values = dict(
        i=i_lim,
        omega=omega_lim,
        u=U_dc,
    )

    nominal_values = {key: 0.7 * limit for key, limit in limit_values.items()}

    env = gem.make(
        'PMSMCont-v1',
        load=ExternalSpeedLoad(speed_profile=speed_profile, tau=tau),
        control_space='dq',
        ode_solver='scipy.solve_ivp', solver_kwargs={},
        reference_generator=rg,
        reward_weights={'i_sq': 0.5, 'i_sd': 0.5},
        reward_power=0.5,
        observed_states=None, # ['i_sd', 'i_sq'],
        tau=tau,
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

    # Define a memory buffer for the agent, allows to learn from past experiences
    memory = SequentialMemory(
        limit=0,
        window_length=window_length
    )

    # Create a random process for exploration during training
    # this is essential for the DDPG algorithm
    random_process = OrnsteinUhlenbeckProcess(
        theta=10,
        mu=0.0,
        sigma=0,
        dt=env.physical_system.tau,
        sigma_min=0,
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

    if use_dessca:
        folder_name = "Dessca"
    else:
        folder_name = "Uniform"
    agent.load_weights(filepath=folder_name +
                                "/" + folder_name +
                                "_" + str(agent_idx) +
                                "/" + folder_name +
                                "_weights.hdf5"
                       )

    agent.test(
        env,
        nb_max_start_steps=0,
        nb_max_episode_steps=test_duration,
        nb_episodes=1,
        visualize=False,
        action_repetition=1,
        verbose=2,
        callbacks=[],
    )

if __name__ == '__main__':
    with Pool() as p:
        p.map(test_agent, range(50))
