from src.agents.agent import Agent, DecisionMaker
from src.environments.env_wrapper import*
# 'Environment Related Imports'
# import tqdm
# import gym
import PIL
import matplotlib.pyplot as plt

from multi_taxi.taxi_environment import TaxiEnv
from gym import Wrapper
from gym.spaces import MultiDiscrete, Box


# 'Deep Model Related Imports'
from torch.nn.functional import one_hot
import torch
import numpy as np
from stable_baselines3 import DQN, PPO

# del model # remove to demonstrate saving and loading
#
# model = DQN.load("dqn_cartpole")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()


class PPODecisionMaker(DecisionMaker):
    def __init__(self, action_space):
        self.space = action_space
        self.model_file_name = 'M_Taxi_PPO1'
        try:
            self.model = PPO.load(self.model_file_name)
        except:
            self.train_model()
        print(f"{self.model.observation_space}")

        self.__is_image_obs = isinstance(self.model.observation_space, Box)

        if self.__is_image_obs:
            self.obs_size = np.prod(self.model.observation_space.shape)
        elif isinstance(self.model.observation_space, MultiDiscrete):
            self.obs_size = len(self.model.observation_space.nvec)
        else:
            raise NotImplemented('other observation spaces not supported')


        # print(f"{self.obs_size}")
        # for obs in self.model.observation_space:
        #     print(f"obs size: {obs}")
        #     print(f"obs size: { obs.n }")

# fix observation space if not fit to trained dimension
    def fit_obs(self, obs):
        temp = [obj for obj in obs[:(self.obs_size-1)]]
        temp.append(obs[-1])
        print(f"{temp}")
        return temp

    def get_action(self, observation):
        if not self.__is_image_obs:
            if (len(observation)!=self.obs_size):
                observation = self.fit_obs(observation)
        action, _states = self.model.predict(observation, deterministic=True)
        return action

    def train_model(self):
        temp_env = TaxiEnv(num_taxis=1, pickup_only=True, observation_type='symbolic')
        temp_env = SingleTaxiWrapper(temp_env)
        temp_env = SinglePassengerPosWrapper(temp_env, pass_pos=[0, 0])
        temp_env.render()
        temp_env.reset()
        temp_env.render()

        model = PPO("MlpPolicy", temp_env, verbose=1)
        model.learn(total_timesteps=200000,n_eval_episodes=20)
        self.model = model
        self.model.save(self.model_file_name)

def from_RGBarray_to_image(obs):
    fig = plt.figure(figsize=(16, 4))
    for ob in obs:
        plt.imshow(obs)
        # if filename is None:
        plt.show()
    # ax = fig.add_subplot(1, len(self.agents), i)
    # i += 1
    # plt.title(title)
    # ax.imshow(observation[agent_name])
    return PIL.Image.frombytes('RGB',
                        fig.canvas.get_width_height(), fig.canvas.tostring_rgb())



if __name__ == '__main__':
    # check code:
    env = TaxiEnv(num_taxis=2, num_passengers=2, observation_type='symbolic')  # pickup_only=True,
    # env = SingleTaxiWrapper(env)
    # env = SinglePassengerPosWrapper(env, pass_pos=[0, 0])
    env = ObsWrapper(env)

    obs = env.reset()
    env.render()

    agents = env.taxis_names
    # obs = (obs[a_n] for a_n in agents)
    # im = from_RGBarray_to_image(obs)
    D_M = PPODecisionMaker(env.action_space)
    env.render()
    print(f"obs:{obs}")
    print(f"next action: { env.index_action_dictionary[D_M.get_action(obs)]}")



