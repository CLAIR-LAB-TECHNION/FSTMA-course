import numpy as np
from tqdm import tqdm
from IPython.display import clear_output
import copy as cp


# Multi-Agent Control imports
from mac.control.controller_decentralized import DecentralizedController
from mac.control.controller_centralized import CentralizedController

from src.agents import DqnDecisionMaker
from mac.agents import Agent


def create_decentralized_agent(policy_name, env, agent_name, kwargs, load_agent=False):
    """
    Creates a decentralized agent
    Args:
        policy_name (str): name of RL algorithm
        env: the multi_taxi env
        agent_name (str): name of the agent
        kwargs (dict): dict parameters for RL algorithm

    Returns: agent

    """
    agent_kwargs = cp.copy(kwargs)
    agent_kwargs['input_dims'] = env.get_observation_space(agent_name).shape
    agent_kwargs['n_actions'] = env.get_action_space(agent_name).n
    agent_kwargs['env_name'] = agent_kwargs['env_name'] + '__' + agent_name

    if policy_name == 'dqn':
        agent = Agent(DqnDecisionMaker(**agent_kwargs))
    else:
        print('... invalid operation ...')
        return
    print(f'... initialize a decentralized {policy_name} agent for agent {agent_name} ...')
    if load_agent:
        agent.get_decision_maker().load_models()
        print(f'... load a decentralized {policy_name} agent for agent {agent_name} ...')
    return agent


def create_centralized_agent(policy_name, env, agent_name, kwargs, load_agent=False):
    """
    Creates a centralized agent
    Notes:
        Compute the dimension of the joint state space and the joint action space
            [
             agent_0_x, agent_0_y, ..., agent_n_x, agent_0_y,
             pass_0_loc_x, pass_0_loc_y, pass_0_des_x, pass_0_des_y, ...
             pass_m_loc_x, pass_m_loc_y, pass_m_des_x, pass_m_des_y,
             pass_0_status, ..., pass_m, status
            ]

    Args:
        policy_name (str): name of RL algorithm
        env: the multi_taxi env
        agent_name (str): name of the agent
        kwargs (dict): dict parameters for RL algorithm

    Returns: agent

    """

    # compute the observation space
    agents_ids = env.get_env_agents()
    observation_space = list(env.get_observation_space(agents_ids[0]).shape)
    observation_space[0] += 2 * (len(agents_ids) - 1)
    kwargs['input_dims'] = tuple(observation_space)

    # compute the dimension of the joint action space
    kwargs['n_actions'] = int(env.get_action_space(agents_ids[0]).n ** len(agents_ids))

    # name the agent for saving purposes
    kwargs['env_name'] = kwargs['env_name'] + '__' + agent_name
    if policy_name == 'dqn':
        agent = Agent(DqnDecisionMaker(**kwargs))
    else:
        print('... invalid operation ...')
        return
    print(f'... initialize a centralized {policy_name} agent for agent {agent_name} ...')

    if load_agent:
        agent.get_decision_maker().load_models()
        print(f'... load a centralized {policy_name} agent for agent {agent_name} ...')
    return agent


class DecentralizedRlController(DecentralizedController):
    def __init__(self, env, agents):
        # initialize super class
        super().__init__(env, agents)

    def get_training_joint_action(self, observation):
        """Returns the joint action
        Args:
            observation (dict): the current observatins
        Returns:
            dict: the actions for the agents
        """
        observation = {agent_id: self.agents[agent_id].get_observation(obs)
                       for agent_id, obs in observation.items()}

        joint_action = {}
        for agent_name in self.agent_ids:
            action = self.agents[agent_name].get_decision_maker().get_training_action(observation[agent_name])
            joint_action[agent_name] = action

        return joint_action

    def train(self, render=False, max_iteration=None, max_episode=1):
        """Train the controller on the environment given in the init,
        with the agents given in the init
        Args:
            render (bool, optional): Whether to render while training. Defaults to False.
            max_iteration ([type], optional): Number of steps to train. Defaults to infinity.
            max_episode (int, optional): Number of episodes to train. Defaults to 1.
        """
        episode_index = 0
        self.total_rewards = []
        self.agents_epsilons = {}
        self.agents_reward = {}
        for agent_id in self.agent_ids:
            self.agents_epsilons[agent_id] = []
            self.agents_reward[agent_id] = {'best_avg_reward': float('-inf'), 'avg_reward': [-2], 'rewards': []}

        for episode in tqdm(range(max_episode)):

            self.total_rewards.append([])
            done = {'__all__': False}
            index = 0
            observation = self.env.get_env().reset()

            while not done['__all__']:
                index += 1
                if max_iteration is not None and index > max_iteration:
                    break

                # display environment
                if render:
                    self.env.render()

                # assert observation is in dict form
                observation = self.env.observation_to_dict(observation)

                # get actions for all agents and perform
                joint_action = self.get_training_joint_action(observation)
                observation_, reward, done, info = self.perform_joint_action(joint_action)

                # store the joint transition
                self.store_joint_transition(observation, joint_action, reward, observation_, done)

                # distribute rewards
                self.distribute_joint_reward(reward)

                # save agents checkpoints
                self.save_checkpoints()

                # perform joint learning for all agents
                self.get_agents_epsilons()
                self.joint_learning()

                # save rewards
                self.total_rewards[episode].append(reward)

                # check if all agents are done
                if all(done.values()):
                    break

                # step the state forward
                observation = observation_

            if render:
                self.env.render()

            episode_index += 1
            clear_output()

        return self.total_rewards, self.agents_epsilons

    def distribute_joint_reward(self, joint_reward, evaluate=False):
        """
        Saves rewards for all agents for analysis and computes the rewards averages
        Args:
            joint_reward (dict): dict of agents rewards
            evaluate:

        Returns: None

        """
        for agent_id in self.agent_ids:
            if evaluate:
                self.eval_agents_reward[agent_id]['rewards'].append(joint_reward[agent_id])
                self.eval_agents_reward[agent_id]['avg_reward'].append(
                    np.mean(self.eval_agents_reward[agent_id]['rewards'][-100:]))
            else:
                self.agents_reward[agent_id]['rewards'].append(joint_reward[agent_id])
                self.agents_reward[agent_id]['avg_reward'].append(
                    np.mean(self.agents_reward[agent_id]['rewards'][-100:]))

    def save_checkpoints(self):
        """
        Save checkpoint for all agents with improved averaged performances
        Returns: None

        """
        for agent_id in self.agent_ids:
            if self.agents_reward[agent_id]['avg_reward'][-1] > self.agents_reward[agent_id]['best_avg_reward']:
                self.agents_reward[agent_id]['best_avg_reward'] = self.agents_reward[agent_id]['avg_reward'][-1]
                self.agents[agent_id].get_decision_maker().save_models()

    def joint_learning(self):
        """
        Training the neural networks of each one of the agents
        Returns: None

        """
        for agent_id in self.agents:
            self.agents[agent_id].get_decision_maker().learn()

    def get_agents_epsilons(self):
        """
        Saves the epsilons of training for analysis
        Returns: None

        """
        for agent_id in self.agents:
            epsilon = self.agents[agent_id].get_decision_maker().epsilon
            self.agents_epsilons[agent_id].append(epsilon)

    def store_joint_transition(self, joint_observation, joint_action, joint_reward, joint_observation_, joint_done):
        """
        Breaks the joint transition into individual transitions and saves in each agent's replay buffer
        Args:
            joint_observation (dict):
            joint_action (dict):
            joint_reward (dict):
            joint_observation_ (dict):
            joint_done (dict):

        Returns: None

        """
        for agent_id in self.agents:
            # break the joint variables to individual variables
            observation = joint_observation[agent_id]
            action = joint_action[agent_id]
            reward = joint_reward[agent_id]
            observation_ = joint_observation_[agent_id]
            done = joint_done[agent_id]

            # store the individual transition per agent
            self.agents[agent_id].get_decision_maker().store_transition(observation, action, reward, observation_, done)

    def evaluate(self, render=False, max_iteration=None, max_episode=1):
        """Evaluate the controller on the environment given in the init,
        with the agents given in the init
        Args:
            render (bool, optional): Whether to render while training. Defaults to False.
            max_iteration ([type], optional): Number of steps to train. Defaults to infinity.
            max_episode (int, optional): Number of episodes to train. Defaults to 1.
        """
        episode_index = 0
        self.eval_total_rewards = []
        self.eval_agents_reward = {}
        for agent_id in self.agent_ids:
            self.eval_agents_reward[agent_id] = {'best_avg_reward': float('-inf'), 'avg_reward': [-2], 'rewards': []}

        for episode in range(max_episode):

            self.eval_total_rewards.append([])
            done = {'__all__': False}
            index = 0
            observation = self.env.get_env().reset()

            while not done['__all__']:
                index += 1
                if max_iteration is not None and index > max_iteration:
                    break

                # display environment
                if render:
                    self.env.render()

                # assert observation is in dict form
                observation = self.env.observation_to_dict(observation)

                # get actions for all agents and perform
                joint_action = self.get_joint_action(observation)
                observation_, reward, done, info = self.perform_joint_action(joint_action)

                # distribute rewards
                self.distribute_joint_reward(reward, evaluate=True)

                # save rewards
                self.eval_total_rewards[episode].append(reward)

                # check if all agents are done
                if all(done.values()):
                    break

                # step the state forward
                observation = observation_

            if render:
                self.env.render()

            episode_index += 1

        return self.eval_total_rewards


class CentralizedRlController(CentralizedController):
    def __init__(self, env, central_agent):
        # initialize super class
        super().__init__(env, central_agent)

        self.agent_name = list(central_agent.keys())[0]
        self.action_space = env.get_action_space(self.agent_ids[0])

    def get_joint_action(self, decomposed_observation):
        """Returns the joint actions of all the agents

        Args:
            decomposed_observation ([dict]): The central agent observation

        Returns:
            dict: dict of central action
        """
        action = self.central_agent[self.agent_name].get_decision_maker().get_action(
            decomposed_observation[self.agent_name])
        return {self.agent_name: action}

    def get_training_joint_action(self, decomposed_observation):
        """Returns the joint actions of all the agents

                Args:
                    decomposed_observation ([dict]): The central agent observation

                Returns:
                    dict: dict of central action
                """
        action = self.central_agent[self.agent_name].get_decision_maker().get_training_action(
            decomposed_observation[self.agent_name])
        return {self.agent_name: action}

    def train(self, render=False, max_iteration=None, max_episode=1):
        """Train the controller on the environment given in the init,
        with the central_agent given in the init
        Args:
            render (bool, optional): Whether to render while training. Defaults to False.
            max_iteration ([type], optional): Number of steps to train. Defaults to infinity.
            max_episode (int, optional): Number of episodes to train. Defaults to 1.
        """
        episode_index = 0
        self.total_reward = []
        self.agent_epsilon = {self.agent_name: []}
        self.agent_reward = {self.agent_name: {
            'best_avg_reward': float('-inf'),
            'avg_reward': [-2 * len(self.agent_ids)],
            'rewards': []}}

        for episode in tqdm(range(max_episode)):

            self.total_reward.append([])
            decomposed_done = {self.agent_name: False}
            index = 0
            observation = self.env.get_env().reset()

            while not decomposed_done[self.agent_name]:
                index += 1
                if max_iteration is not None and index > max_iteration:
                    break

                # display environment
                if render:
                    self.env.render()

                # assert observation is in dict form
                observation = self.env.observation_to_dict(observation)

                # decompose the observation
                decomposed_observation = self.decompose_observation(observation)

                # get the central agent action
                decomposed_action = self.get_training_joint_action(decomposed_observation)

                # compute the action for each agent
                joint_action = self.decode_action(decomposed_action)

                # perform the joint action
                observation_, reward, done, info = self.perform_joint_action(joint_action)

                # decompose observation_ adn reward
                decomposed_observation_ = self.decompose_observation(observation_)
                decomposed_reward = self.decompose_reward(reward)
                decomposed_done[self.agent_name] = done['__all__']

                # store the joint transition
                self.store_decomposed_transition(decomposed_observation, decomposed_action, decomposed_reward,
                                                 decomposed_observation_, decomposed_done)

                # save the central agent checkpoint
                self.save_checkpoint()

                # save epsilon for analysis and perform learning for the central agent
                self.learn()

                # save rewards
                self.total_reward[episode].append(decomposed_reward)
                self.agent_epsilon[self.agent_name].append(
                    self.central_agent[self.agent_name].get_decision_maker().epsilon)

                # check if all agents are done
                if all(done.values()):
                    break

                # step the state forward
                observation = observation_

            if render:
                self.env.render()

            episode_index += 1
            clear_output()

        return self.total_reward, self.agent_epsilon

    def decompose_reward(self, joint_reward, evaluate=False):
        """
        Decomposes the joint reward into a central agent single reward
        Args:
            joint_reward (dict): dictionary of al agents rewards
            evaluate: TBD

        Returns:
            dict of central agent reward
        """
        decomposed_reward = 0
        for agent_id in joint_reward:
            decomposed_reward += joint_reward[agent_id]

        if evaluate:
            self.eval_agent_reward[self.agent_name]['rewards'].append(decomposed_reward)
            self.eval_agent_reward[self.agent_name]['avg_reward'].append(
                np.mean(self.eval_agent_reward[self.agent_name]['rewards'][-100:]))
        else:
            self.agent_reward[self.agent_name]['rewards'].append(decomposed_reward)
            self.agent_reward[self.agent_name]['avg_reward'].append(
                np.mean(self.agent_reward[self.agent_name]['rewards'][-100:]))
        return {self.agent_name: decomposed_reward}

    def decode_action(self, decomposed_action, num_agents=None):
        """Decodes the action from the RL algorithm into the environment format

        Args:
            decomposed_action (dict): The action from the RL algo
            num_agents (int): computed inside the function
        Returns:
            dict: dictionary of the joint action
        """
        # decode action
        action = decomposed_action[self.agent_name]
        num_actions = self.action_space.n
        num_agents = len(self.agent_ids)
        actions = []
        for ind in range(num_agents):
            actions.append(action % num_actions)
            action = action // num_actions

        # arrange actions into dictionary
        actions = actions[::-1]
        joint_action = {}
        for i, agent_id in enumerate(self.agent_ids):
            joint_action[agent_id] = actions[i]

        return joint_action

    def save_checkpoint(self):
        """
        Saves central agent's checkpoint if average reward is better then best averaged reward
        Returns: None

        """
        if self.agent_reward[self.agent_name]['avg_reward'][-1] > self.agent_reward[self.agent_name]['best_avg_reward']:
            self.agent_reward[self.agent_name]['best_avg_reward'] = self.agent_reward[self.agent_name]['avg_reward'][-1]
            self.central_agent[self.agent_name].get_decision_maker().save_models()

    def learn(self):
        """
        Trains the neural network of the central agent
        Returns:

        """
        self.central_agent[self.agent_name].get_decision_maker().learn()

    def decompose_observation(self, observation):
        """
        Decompose the observations of all taxis to a single central agent observation of the form
            [taxi_0_x, taxi_0_y, ..., taxi_n_x, taxi_n_y,
             pass_0_loc_x, pass_0_loc_y, pass_0_des_x, pass_0_des_y,
             ...,
             pass_m_loc_x, pass_m_loc_y, pass_m_des_x, pass_m_des_y,
             pass_0_stat, ..., pass_m_stat]

        Args:
            observation (dict): dict of joint observation of all the taxis

        Returns:
            dict of central agent observation
        """
        deco_observation = []
        for agent_id in self.agent_ids:
            agent_id_observation = list(observation[agent_id])
            agent_id_row = agent_id_observation[0]
            agent_id_col = agent_id_observation[1]
            deco_observation += [agent_id_row, agent_id_col]

        agent_id_observation = list(observation[self.agent_ids[0]])
        deco_observation += agent_id_observation[2:]

        return {self.agent_name: np.array(deco_observation)}

    def store_decomposed_transition(self, decomposed_observation, decomposed_action, decomposed_reward,
                                    decomposed_observation_, decomposed_done):
        """
        Stores the central agent's transition
        Args:
            decomposed_observation (dict):
            decomposed_action (dict):
            decomposed_reward (dict):
            decomposed_observation_ (dict):
            decomposed_done (dict):

        Returns: None

        """
        for agent_id in self.central_agent:
            # break the joint variables to individual variables
            observation = decomposed_observation[agent_id]
            action = decomposed_action[agent_id]
            reward = decomposed_reward[agent_id]
            observation_ = decomposed_observation_[agent_id]
            done = decomposed_done[agent_id]

            # store the individual transition per agent
            self.central_agent[agent_id].get_decision_maker().store_transition(
                observation, action, reward, observation_, done)

    def evaluate(self, render=False, max_iteration=None, max_episode=1):
        """Evaluate the controller on the environment given in the init,
            with the central_agent given in the init
        Args:
            render (bool, optional): Whether to render while training. Defaults to False.
            max_iteration ([type], optional): Number of steps to train. Defaults to infinity.
            max_episode (int, optional): Number of episodes to train. Defaults to 1.
        """

        episode_index = 0
        self.eval_total_reward = []
        self.eval_agent_reward = {self.agent_name: {
            'best_avg_reward': float('-inf'),
            'avg_reward': [-2 * len(self.agent_ids)],
            'rewards': []}}

        for episode in range(max_episode):

            self.eval_total_reward.append([])
            decomposed_done = {self.agent_name: False}
            index = 0
            observation = self.env.get_env().reset()

            while not decomposed_done[self.agent_name]:
                index += 1
                if max_iteration is not None and index > max_iteration:
                    break

                # display environment
                if render:
                    self.env.render()

                # assert observation is in dict form
                observation = self.env.observation_to_dict(observation)

                # decompose the observation
                decomposed_observation = self.decompose_observation(observation)

                # get the central agent action
                decomposed_action = self.get_joint_action(decomposed_observation)

                # compute the action for each agent
                joint_action = self.decode_action(decomposed_action)

                # perform the joint action
                observation_, reward, done, info = self.perform_joint_action(joint_action)

                # decompose observation_ adn reward
                decomposed_observation_ = self.decompose_observation(observation_)
                decomposed_reward = self.decompose_reward(reward, evaluate=True)
                decomposed_done[self.agent_name] = done['__all__']

                # save rewards
                self.eval_total_reward[episode].append(decomposed_reward)

                # check if all agents are done
                if all(done.values()):
                    break

                # step the state forward
                observation = observation_

            if render:
                self.env.render()

            episode_index += 1

        return self.eval_total_reward

