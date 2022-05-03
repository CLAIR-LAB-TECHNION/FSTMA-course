import random
from time import sleep
import os
import torch
import sys
import itertools
sys.path.append('multi-taxi')
from multi_taxi import MultiTaxiEnv
import numpy as np
import gym

#customized_reward = dict(intermediate_dropoff=20,pickup = 5000,final_dropoff = 1000)
env = MultiTaxiEnv(num_taxis=1,  # 1 taxi agents in the environment
                   num_passengers=8,  # 2 passengers in the environment
                   max_fuel=[30],  # taxi1 has a capacity of 30 fuel units, tax2 has 50, and taxi3 has 25
                   taxis_capacity=None,  # unlimited passenger capacity for all taxis
                   option_to_stand_by=False,  # taxis can turn the engin on/off and perform a standby action
                   observation_type='symbolic',  # accepting symbolic (vector) observations
                   can_see_others=False,
                   pickup_only=True)
                   #rewards_table=customized_reward)  # cannot see other taxis (but also not collision sensitive)

print()
print(f'taxi observation space: {env.observation_space}')
print(f'taxi action space: {env.action_space}')
print(f'taxi possible actions: {env.index_action_dictionary}')

# from multi_taxi.utils import StochasticActionFunction
#
# # define a conditional action distributions for each taxi
# taxi0_action_dist = {
#     'north': {'north': 0.5, 'east': 0.25, 'west': 0.25},
#     'south': {'south': 0.5, 'east': 0.25, 'west': 0.25}
# }
# taxi1_action_dist = {
#     'east': {'east': 0.5, 'north': 0.25, 'south': 0.25},
#     'west': {'west': 0.5, 'north': 0.25, 'south': 0.25}
# }
#
# # create a different stochast
# f0 = StochasticActionFunction(taxi0_action_dist)
# f1 = StochasticActionFunction(taxi1_action_dist)
def encode(obs,num_passengers,num_taxis):
    obs_parsed = obs['taxi_0'][:-num_passengers]
    one_hot_passengers = torch.nn.functional.one_hot(
        torch.Tensor(obs['taxi_0'][-num_passengers:] - 1).to(torch.int64), num_classes=num_taxis + 2)
    obs['taxi_0'] = torch.cat([torch.FloatTensor(obs_parsed), one_hot_passengers.flatten().to(torch.float32)])
    obs['taxi_0'][2] /= 6
    obs['taxi_0'][3] /= 11
def extract(d):
    return list(d.values())[0]
obs = env.reset()
encode(obs,env.num_passengers,env.num_taxis)
obs = extract(obs)
env.render()
def rand_policy(obs):
    return random.randint(0,7)

# iterate and step in environment.
# limit num actions for incomplete policies
policy = rand_policy
import torch
from dpg_S import PolicyNetwork as PN
model = PN(0.004, env.observation_space.shape[0]+2*env.num_passengers, env.action_space.n, None)
model.load_state_dict(torch.load('trained_model_gym'))
model.eval()
actions = []
while(True):
    #action = policy(obs['taxi_0'])
    #trajectory.add_step(obs, action)
    with torch.no_grad():
        probs = model(obs)

    probs =torch.distributions.Categorical(torch.nn.Softmax()(probs))
    action = probs.sample().item()
    print(action)

    # if len(actions) == 10:
    #     while action in actions and actions.count(action) >= 7:
    #         action = probs.sample().item()
    #     actions = []
    # actions.append(action)

    obs, reward, done, info = env.step({'taxi_0':action})
    encode(obs,env.num_passengers,env.num_taxis)

    obs = extract(obs)
    reward = extract(reward)
    done = extract(done)
    info = extract(info)
    sleep(2)
    # action = {'taxi_0':action}
    #
    # obs, reward, done, info = env.step(action)
    # except:
    #     obs, reward, done, info = env.step(action)
    env.render()


    if done:
        break


env.reset()
