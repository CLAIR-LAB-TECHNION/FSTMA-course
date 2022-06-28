from src.agents.agent import Agent, DecisionMaker
from src.environments.env_wrapper import*
# 'Environment Related Imports'
# import tqdm
# import gym
import PIL
import matplotlib.pyplot as plt

from multi_taxi.taxi_environment import TaxiEnv
from gym import Wrapper
from gym.spaces import MultiDiscrete


# 'Deep Model Related Imports'
from torch.nn.functional import one_hot
import torch
import numpy as np
from stable_baselines3 import DQN




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


class PlannerAstarDecisionMaker(DecisionMaker):
    def __init__(self, action_space):
        self.space = action_space
        self.model_file_name = "M_Taxi_Astar"
        try:
            self.model = DQN.load(self.model_file_name)
        except:
            self.train_model()

    def get_action(self, observation):
        action, _states = self.model.predict(observation, deterministic=True)
        return action

    def train_model(self):
        env = TaxiEnv(num_taxis=1, num_passengers=1, pickup_only=True, observation_type='image')
        env = SingleTaxiWrapper(env)
        # env = SinglePassengerPosWrapper(env, taxi_pos=[0, 0])
        env.render()
        # env = TaxiObsPrepWrapper(env)

        self.model = DQN("MlpPolicy", env, verbose=1)    #  CnnPolicy for images use "
        self.model.learn(total_timesteps=10000, log_interval=4)
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
    env = TaxiEnv(num_taxis=1, observation_type='image')  # pickup_only=True,
    env = SingleTaxiWrapper(env)
    obs = env.reset()

    agents = env.taxis_names
    # obs = (obs[a_n] for a_n in agents)
    # im = from_RGBarray_to_image(obs)
    D_M = LearningDecisionMaker(env.action_space)
    env.render()
    print(f"obs:{type(obs)}")
    print(f"next action: { env.index_action_dictionary[D_M.get_action(obs)]}")



