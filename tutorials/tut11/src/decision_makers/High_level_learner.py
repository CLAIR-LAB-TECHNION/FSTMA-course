"""
TTR - step 4 - High-level-Taxi-env, learning - HL-RL agents
"""



from src.agents.agent import Agent, DecisionMaker
# 'Environment Related Imports'
# import tqdm
# import gym
import PIL
import matplotlib.pyplot as plt
import math

from src.Communication.COM_net import COM_net
from src.agents.agent import DecisionMaker, Action_message_agent, Agent_Com
from src.control.Controller_COM import DecentralizedComController
from src.decision_makers.planners.Com_High_level_Planner import Astar_message_highlevel_DM
from src.decision_makers.planners.map_planner import AstarDM
from src.environments.env_wrapper import EnvWrappper
from src.environments.hirarchical_Wrapper import Multi_Taxi_Task_Wrapper
from multi_taxi import MultiTaxiEnv
from multi_taxi.taxi_environment import TaxiEnv
from src.environments.env_wrapper import*
from gym.spaces import MultiDiscrete


# 'Deep Model Related Imports'
from torch.nn.functional import one_hot
import torch
import numpy as np
from stable_baselines3 import DQN, PPO

MAP2 = [
    "+-------+",
    "| : |F: |",
    "| : | : |",
    "| : : : |",
    "| | :G| |",
    "+-------+",
]

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




class LearningHighLevelDecisionMaker(DecisionMaker):
    def __init__(self, action_space):
        self.space = action_space
        self.model_file_name = "M_Taxi_HL_DQN1"
        try:
            self.model = PPO.load(self.model_file_name)
        except:
            self.train_model()

    def get_action(self, observation):
        action, _states = self.model.predict(observation, deterministic=True)
        return action

    def train_model(self):
        env = MultiTaxiEnv(num_taxis=1, num_passengers=2, domain_map=MAP2, observation_type='symbolic',option_to_stand_by=True)
        env.agents = env.taxis_names
        env = EnvWrappper(env,env.agents)
        env = Multi_Taxi_Task_Wrapper(env)
        # env = SinglePassengerPosWrapper(env, taxi_pos=[0, 0])
        env.env.env.render()
        # env = TaxiObsPrepWrapper(env)
        self.model = PPO("MlpPolicy", env, verbose=1)

        for i in range(20):
            print(f" setting no. : {i}")
            env.reset()
            self.model.learn(total_timesteps=20000,n_eval_episodes=20)
            if i%5==4 : self.model.save(self.model_file_name)
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
    env = MultiTaxiEnv(num_taxis=1, num_passengers=2, domain_map=MAP2, observation_type='symbolic',option_to_stand_by=True)
    env.agents = env.taxis_names
    env = EnvWrappper(env,env.agents)
    env = Multi_Taxi_Task_Wrapper(env)
    obs = env.reset()

    # obs = (obs[a_n] for a_n in agents)
    # im = from_RGBarray_to_image(obs)
    D_M = LearningHighLevelDecisionMaker(env.action_space)
    obs = env.reset()
    env.env.env.render()
    for i in range(50):
        print(f"obs:{type(obs)}")
        a = D_M.get_action(obs)
        print(f"next action: { env.index_action_dictionary[a]}")
        obs, r, done, info = env.step(a)
        env.env.env.render()
        if done: break
    print("that all")



