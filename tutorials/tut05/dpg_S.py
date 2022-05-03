"""

Policy gradient implementaiton using PyTorch

https://www.youtube.com/watch?v=GOBvUA9lK1Q (sorry)

Details of below:
Two fully connected hidden layers, default 256 neurons each
Relu activation functions, softmax final activation function
Update according to log gradient
With standardized updates

TODO: add option to use cnn instead of fc (e.g. pixel env)
TODO: check if working
"""
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pettingzoo.mpe import simple_v2
from multi_taxi import MultiTaxiEnv
from train_and_eval_S import train

class PolicyNetwork(nn.Module): # The network itself is separate from the agent
    def __init__(self, learning_rate, input_dims, num_actions, mode,emd_dim = 6):
        super(PolicyNetwork, self).__init__()

        self.input_dims = input_dims
        self.learning_rate = learning_rate
        self.num_actions = num_actions
        self.mode = mode
        self.emb_dim = emd_dim


        #mpe net
        h_dim = 8
        self.layers = nn.Sequential(
            nn.Linear(self.input_dims, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, self.num_actions),
        )

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        return self.layers(state)

class DPGAgent(object):
    def __init__(self, learning_rate, input_dims, num_actions,mode, gamma=0.99, identifier=None,\
                action_return_format=None):
        self.gamma = gamma
        self.reward_mem = []
        self.action_mem = []
        self.num_actions = num_actions
        self.gamma = gamma
        self.policy = PolicyNetwork(learning_rate, input_dims, num_actions,mode)
        self.action_return_format = action_return_format
        self.identifier = identifier


    def int_to_vector(self, action):
        """ Turns integer action into one hot vector """
        vec = torch.zeros((self.num_actions))
        vec[action] = 1
        return vec

    def learn(self):
        """ Calculate rewards for the episode, compute the gradient of the loss, update optimizer"""
        self.policy.optimizer.zero_grad()
        G = np.zeros_like(self.reward_mem, dtype=np.float64)

        for t in range(len(self.reward_mem)):
            G_sum = 0
            discount = 1

            for k in range(t, len(self.reward_mem)):
                G_sum += self.reward_mem[k] * discount
                discount *= self.gamma
            G[t] = G_sum

        # standardize updates
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G-mean)/std

        G = T.tensor(G, dtype=T.float).to(self.policy.device)
        loss = torch.Tensor([0])

        baseline = sum(self.reward_mem)/len(self.reward_mem)
        for i in range(len(self.reward_mem)):
            self.reward_mem[i] -= baseline


        for g, logprob in zip(G, self.action_mem):
            loss += -g*logprob

        loss.backward()
        self.policy.optimizer.step()


    """ Training Callbacks """
    def action_callback_mpe(self, observation):
        probs = F.softmax(self.policy.forward(observation))
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()

        log_probs = action_probs.log_prob(action)
        self.action_mem.append(log_probs)

        returned_action = self.int_to_vector(action.item())


        return returned_action
    def action_callback(self, observation): #action callback for gym

        if self.identifier.startswith('agent'): #'agent' is an mpe env identifier
            return self.action_callback_mpe(observation)

        probs = F.softmax(self.policy.forward(observation))
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()

        log_probs = action_probs.log_prob(action)
        self.action_mem.append(log_probs)

        returned_action = action.item()
        if self.action_return_format == 'vector':
            returned_action = self.int_to_vector(returned_action)

        return returned_action
    def experience_callback(self, obs, action, new_obs, reward, done):
        self.reward_mem.append(reward)

    def episode_callback(self):
        """ Reset at the end of an episode"""
        self.learn()
        self.reward_mem = []
        self.action_mem = []

    """ Evaluation Callbacks """
    def policy_callback(self, observation):
        probs = F.softmax(self.policy.forward(observation))
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()

        return action.item()

    def reset(self):
        return
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



import time
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
        time.sleep(0.05)

        # render the environment
        env.render()
    env.close()


def test_continuous_single_agent(mode):

    #disambiguate env (gym/mpe)
    if mode == 'gym':
        env = MultiTaxiEnv(num_taxis=1,  # 1 taxi agents in the environment
                           num_passengers=1,  # 1 passenger in the environment
                           max_fuel=[30],  # taxi1 has a capacity of 30 fuel units, tax2 has 50, and taxi3 has 25
                           taxis_capacity=None,  # unlimited passenger capacity for all taxis
                           option_to_stand_by=False,  # taxis can't turn the engin on/off and perform a standby action
                           observation_type='symbolic',  # accepting symbolic (vector) observations
                           can_see_others=False,
                           pickup_only=True)
                           #rewards_table=customized_reward)  # cannot see other taxis (but also not collision sensitive)
    else:
        env = simple_v2.parallel_env(continuous_actions=True)


    agent = env.possible_agents[0] if mode == 'mpe' else 'taxi_0' #single agent
    num_actions = env.action_space(agent).shape[0] if mode == 'mpe' else env.action_space.n
    num_states = env.observation_spaces[agent].shape[0] if mode == 'mpe' else env.observation_space.shape[0]


    #define agent and model params
    learning_rate = 0.004
    identifier = 'agent_0' if mode == 'mpe' else 'taxi_0'
    dpg_agent = DPGAgent(learning_rate, num_states, num_actions, gamma=0.99, identifier = identifier, mode=mode)


    train(env=env, is_env_multiagent=False, agents=[dpg_agent], max_episode_len=50000, num_episodes=600,
             display=False, save_rate=25, agents_save_path="", train_result_path="")
    torch.save(dpg_agent.policy.state_dict(), f'trained_model_{mode}')



if __name__ == "__main__":
    test_continuous_single_agent('gym')
