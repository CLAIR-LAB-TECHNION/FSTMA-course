
from pettingzoo.mpe import simple_v2
from time import sleep
import numpy as np


def print_env_info(continuous_actions):
    env = simple_v2.env(continuous_actions=continuous_actions)
    env.reset()

    print('continuous actions:' if continuous_actions else 'discrete actions:')

    for i, agent in enumerate(env.agents, 1):
        print(f'- agent {i}: {agent}')
        print(f'\t- observation space: {env.observation_space(agent)}')
        print(f'\t- action space: {env.action_space(agent)}')



print_env_info(continuous_actions=True)
env = simple_v2.parallel_env(continuous_actions=True, max_cycles = 400)
env.reset()

# run twice to show the chnage in the communication vector
#env.action_space(env.possible_agents[0]).shape

obs, _, _, _ = env.step({env.possible_agents[0]:np.array([0.5,0.5,0.5,0.5,0.5])}) # get speaker observation vector
print(f'agent: {env.agents[0]}')
print(f'observation: {obs}')
print()







def speaker_continuous_action(v1, v2, v3):
    return np.array([v1, v2, v3], dtype=np.float32)


# listener continuous action function
def listener_continuous_action(right, left, up, down):
    return np.array([0, right, left, up, down], dtype=np.float32)







# CHANGE ACTION AS NEEDED
chosen_listener_action = listener_continuous_action(right=0.8, left=0.8, up=0.5, down=0.7)



def extract(d):
    return list(d.values())[0]


class RandomPolicy:
    def __init__(self, action_space):
        # choose a policy function for this action space type
        self.policy_fn = self.__continuous_policy
        self.action_space = action_space

    def __call__(self, observation):
        # we completely ignore the observation and create a random valid action.
        return self.policy_fn()

    def __discrete_policy(self):
        # a random number within the discrete action range
        return np.random.randint(self.action_space.n)

    def __continuous_policy(self):
        # a random vector within the continuous range of the appropriate dimensionality
        # convert to the right dtype to avoid clipping warnings (e.g. float64 to float32)
        return np.random.uniform(self.action_space.low, self.action_space.high, self.action_space.shape).astype(
            self.action_space.dtype)


def extract(d):
    return list(d.values())[0]


# run an episode
def run_rand(env):
    rand_policy = RandomPolicy(env.action_space(env.possible_agents[0]))
    obs = env.reset()
    first = True
    while (True):
        # stop if done

        # choose and execute action
        action = rand_policy(obs)
        obs, reward, done, info = env.step({'agent_0': action})
        obs = extract(obs)
        done = extract(done)
        if done:
            break

        # render the environment
        env.render()
    env.close()


for i in range(3):
    run_rand(env)

import torch
from dpg_S import PolicyNetwork as PN
model = PN(0.005, 4, 5, None)
model.load_state_dict(torch.load('trained_model'))
model.eval()


env = simple_v2.parallel_env(continuous_actions=True, max_cycles = 400)
env.reset()
first = True
while(True):
    if first:
        observation, reward, done, info = env.step({'agent_0':np.array([0.5,0.5,0.5,0.5,0.5])})
        observation = extract(observation)
        reward = extract(reward)
        done = extract(done)
        info = extract(info)
        first = False

    # stop if done
    if done:
        break

    # choose and execute action
    with torch.no_grad():
        action = model(observation)
    probs = torch.nn.Softmax()(action)
    observation, reward, done, info = env.step({'agent_0':probs.detach()})
    observation = extract(observation)
    reward = extract(reward)
    done = extract(done)
    info = extract(info)
    sleep(0.1)



    # render the environment
    env.render('human')

env.close()
