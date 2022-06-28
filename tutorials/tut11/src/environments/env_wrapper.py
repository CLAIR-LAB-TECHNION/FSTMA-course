"""
Wrappers for Multi-Taxi domain
many types of decoders
"""


from gym import Wrapper, RewardWrapper, ActionWrapper, ObservationWrapper
from gym.spaces import MultiDiscrete, Discrete, Box, MultiBinary
# 'Deep Model Related Imports'
from torch.nn.functional import one_hot
import torch
import numpy as np

TAXI_pickup_dropoff_REWARDS = dict(
    step=-1,
    no_fuel=-1,
    bad_pickup=-1,
    bad_dropoff=-1,
    bad_refuel=-1,
    bad_fuel=-1,
    pickup=100,
    standby_engine_off=0,
    turn_engine_on=-1,
    turn_engine_off=-1,
    standby_engine_on=0,
    intermediate_dropoff=100,
    final_dropoff=100,
    hit_wall=-1,
    collision=-1,
    collided=-1,
    unrelated_action=-1,
)

MAP = [
    "+-----------------------+",
    "| : |F: | : | : | : |F: |",
    "| : | : : : | : | : | : |",
    "| : : : : : : : : : : : |",
    "| : : : : : | : : : : : |",
    "| : : : : : | : : : : : |",
    "| : : : : : : : : : : : |",
    "| | :G| | | :G| | | : | |",
    "+-----------------------+",
]


class ObsWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.map_h = len(self.unwrapped.desc) - 2 - 1
        self.map_w = (len(self.unwrapped.desc[0]) - 1) #// 2 - 1
        self.map_full_obs()
        self.full_map[1,2] = 2

    def reset(self):
        return self._obs_prep(self.env.reset())

    def map_full_obs(self):
        self.full_map = np.zeros((self.map_h, self.map_w),dtype=int)
        a, b, _, c, d = self.unwrapped.state
        self.unwrapped.state = [a, b, [[0, 0]], c, d]
        # self.full_map_set = np.array(dtype=set)
        # self.full_map_set[0,0] = {"one"}


    def step(self, action):
        obs, r, d, i = self.env.step(action)
        return self._obs_prep(obs), r, d, i

    def _obs_prep(self, obs):
        taxi_and_pass = obs[:-self.unwrapped.num_passengers]
        pass_stat = obs[-self.unwrapped.num_passengers:]
        self.state = self.env.unwrapped

        # taxi_and_pass[::2] = taxi_and_pass[::2] / self.map_h
        # taxi_and_pass[1::2] = taxi_and_pass[1::2] / self.map_w
        # pass_stat = one_hot(torch.from_numpy(pass_stat).to(torch.int64) - 1,
        #                     num_classes=3).flatten().numpy().astype(
        #     np.float64)
        return np.concatenate([taxi_and_pass, pass_stat])

    @property
    def observation_space(self):
        obs_space_v = self.env.observation_space.nvec
        taxi_pass_info = obs_space_v[:-self.unwrapped.num_passengers]
        new_obs_space_v = np.concatenate([taxi_pass_info,
                                          [2] * (self.unwrapped.num_taxis + 2) * self.unwrapped.num_passengers])

        return MultiDiscrete(new_obs_space_v)
#
#
# class AddActionWrapper(Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#
#         self.space_size = self.env.action_space.n
#
#     def action_space(self):
#         return Discrete(self.space_size + 3)
#
#     def step(self, action):
#         changed_taxi_actions = {}
#         for taxi in action:
#             if action[taxi] == new action:
#                 changed_taxi_actions[taxi] = action[taxi]
#                 action[taxi] = standby
#
#         ret_vals = self.env.step(action)
#
#         # do new actions for taxis chagned
#
#         # update ret_vals
#
#         # return ret_Valsz
#
#
#         if action > self.space_size:
#             x = 2  # do new action step
#         else:
#             return self.env.step(action)

class SingleTaxiWrapper(Wrapper):
    """
    A wrapper for multi-taxi environments aligning the environments'
    API with OpenAI Gym if using only 1 taxi.
    """

    def __init__(self, env):
        assert env.num_taxis == 1
        super().__init__(env)

    def reset(self):
        # run `reset` as usual.
        # returned value is a dictionary of observations with a single entry
        obs = self.env.reset()

        # a, b, _, c, d = self.unwrapped.state
        # self.unwrapped.state = [a, b, [[0, 0]], c, d]

        # return the single entry value as is.
        # no need for the key (only one agent)
        ret = next(iter(obs.values()))

        return ret

    def step(self, action):
        # step using "joint action" of a single agnet as a dictionary
        step_rets = self.env.step({self.env.taxis_names[0]: action})

        # unpack step return values from their dictionaries
        return tuple(next(iter(ret.values())) for ret in step_rets)


class SinglePassengerPosWrapper(Wrapper):
    '''Same for above, when we also have only one passenger'''

    def __init__(self, env, pass_pos):
        super().__init__(env)
        self.__pass_pos = pass_pos

    def reset(self):
        obs = self.env.reset()
        a, b, _, c, d = self.unwrapped.state
        self.unwrapped.state = [a, b, [[0, 0]], c, d]
        obs[2:4] = self.__pass_pos
        return obs


class TaxiObsPrepWrapper(Wrapper):
    '''Preprocess observations to make data more meaningful for deep networks - encode passenger locations as vectors
    (one hot) and scale according to coordinate system dimensions.'''

    def __init__(self, env):
        super().__init__(env)
        self.map_h = len(self.unwrapped.desc) - 2 - 1
        self.map_w = (len(self.unwrapped.desc[0]) - 1) // 2 - 1

    def reset(self):
        return self._obs_prep(self.env.reset())

    def step(self, action):
        obs, r, d, i = self.env.step(action)
        return self._obs_prep(obs), r, d, i

    def _obs_prep(self, obs):
        taxi_and_pass = obs[:-self.unwrapped.num_passengers].astype(np.float64)
        pass_stat = obs[-self.unwrapped.num_passengers:]

        taxi_and_pass[::2] = taxi_and_pass[::2] / self.map_h
        taxi_and_pass[1::2] = taxi_and_pass[1::2] / self.map_w
        pass_stat = one_hot(torch.from_numpy(pass_stat).to(torch.int64) - 1, num_classes=3).flatten().numpy().astype(
            np.float64)

        return np.concatenate([taxi_and_pass, pass_stat])

    @property
    def observation_space(self):
        obs_space_v = self.env.observation_space.nvec
        taxi_pass_info = obs_space_v[:-self.unwrapped.num_passengers]
        new_obs_space_v = np.concatenate([taxi_pass_info,
                                          [2] * (self.unwrapped.num_taxis + 2) * self.unwrapped.num_passengers])

        return MultiDiscrete(new_obs_space_v)

class EnvWrappper:
    def __init__(self, env, env_agents, num_observation_spaces=1, num_actions=1):
        self.env = env
        self.env_agents = env_agents
        self.num_observation_spaces = num_observation_spaces
        self.num_actions = num_actions
        self.metadata = None

    def get_env(self):
        return self.env

    def get_num_obs(self):
        return self.num_observation_spaces

    def get_num_actions(self):
        return self.num_actions

    def get_env_agents(self):
        return self.env_agents

    def step(self, joint_action):
        return self.env.step(joint_action)


class EnvWrappperGym:

    def __init__(self, env, needs_conv=False):
        super(EnvWrappperGym, self).__init__(env, self.env.possible_agents,
                                             self.env.observation_spaces[env.possible_agents[0]].shape,
                                             self.env.action_spaces[env.possible_agents[0]].n)
        self.needs_conv = needs_conv

    def get_needs_conv(self):
        return self.needs_conv
