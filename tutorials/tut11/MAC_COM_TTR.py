"""
TTR - step 1 - simple use of com-module and com-agent, showing no com results
"""


import math

from src.Communication.COM_net import COM_net
from src.agents.agent import DecisionMaker, Action_message_agent, Agent_Com, RandomDecisionMaker
from src.control.Controller_COM import DecentralizedComController
from src.decision_makers.planners.map_planner import AstarDM
from src.environments.env_wrapper import EnvWrappper, TAXI_pickup_dropoff_REWARDS
from multi_taxi import MultiTaxiEnv



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

"""
Builds Multi_taxi env
"""
env = MultiTaxiEnv(num_taxis=2, num_passengers=2, domain_map=MAP, observation_type='symbolic')

# env = SingleTaxiWrapper(env)
obs = env.reset()

# env.render()

def build_multi_env(env):
    env.agents = env.taxis_names
    # print(f"{env.agents}\n")
    env.action_spaces = {
        agent_name: env.action_space for agent_name in env.agents
    }
    env.observation_spaces = {
        agent_name: env.observation_space for agent_name in env.agents
    }
    env.possible_agents = [agent for agent in env.agents]
    return EnvWrappper(env, env.agents)


environment = build_multi_env(env)
#
print('EnvironmentWrapper created')

# making agent class that communicates and heads towards 1 passenger (pickup->dropoff)
"""
in order to use com module: 
    - implement Agent_com class (inherit from Agent_com
    - make sure to implement set_data_func - that decides what is the data that the agent will transmit whenever it called
    - u can implement your recieve_func - that decides what to do with a recieved data
        defualt is the union_func that add the message data to the observation 
"""

class Heading_message_agent(Agent_Com):

    def __init__(self, decision_maker : AstarDM , sensor_function =None, message_filter = None, AgentName = None, bandW = math.inf, union_recieve = True):
        super().__init__(decision_maker , sensor_function, message_filter, AgentName, bandW, union_recieve)
        self.last_action = None

    def set_data_func(self, obs):
        data = (self.decision_maker.taking_passenger,len(self.decision_maker.active_plan))
        return data

    # todo - implement your recive_func
    def set_recive_func(self, obs, message):
        pass

    # saves last action of the agent - not necessary for com module
    def set_last_action(self, action):
        self.last_action = action
#
#

"""
part 1 - 
agents sends com but dont use it
"""
# after having our com-Astar-agents class we can set our env agents into a dicentralized_agents dict
env_agents = environment.get_env_agents()
decentralized_agents = {agent_name: Heading_message_agent(AstarDM(env ,single_plan=True, Taxi_index=int(agent_name[-1]), domain_map=MAP) ,AgentName=agent_name)             # Agent(LearningDecisionMaker(env.action_space))  # can use diffrent DM
                        for agent_name in env_agents}


"""
Simple use of communication network - 
    - build one using COM_net() - defualt architecture is - all masseges sent to all other agents
            *U can use more options - see at COM_net() class doc.
"""
com = COM_net()
""" 
    - initailze our new controller (DecentralizedComController) - using our env, our agents and our com module
        - this controller will perform all joint action and message delieveries at any time-step
"""
controller = DecentralizedComController(environment, decentralized_agents, com)
"""
activate 
"""
controller.run(render=True, max_iteration=50)
print("Thats all - part 1")

"""
part 1.1 - use com
"""

env.reset()

environment = build_multi_env(env)
#
print('EnvironmentWrapper created')

"""
in order to use com module: 
    - implement Agent_com class (inherit from Agent_com
    - make sure to implement set_data_func - that decides what is the data that the agent will transmit whenever it called
    - u can implement your recieve_func - that decides what to do with a recieved data
        defualt is the union_func that add the message data to the observation 
"""

class Heading_message_agent(Agent_Com):

    def __init__(self, decision_maker , sensor_function =None, message_filter = None, AgentName = None, bandW = math.inf, union_recieve = True):
        super().__init__(decision_maker , sensor_function, message_filter, AgentName, bandW, union_recieve)
        self.last_action = None

    def set_data_func(self, obs):
        data = (self.last_action)
        return data

    # todo - implement your recive_func (defualt is union with obs)
    # def set_recive_func(self, obs, message):
    #     pass

    # saves last action of the agent - not necessary for com module
    def set_last_action(self, action):
        self.last_action = action

# after having our com-agent class we can set our env agents into a dicentralized_agents dict with a Random DM
env_agents = environment.get_env_agents()
decentralized_agents = {agent_name: Heading_message_agent(RandomDecisionMaker(env.action_space) ,AgentName=agent_name)             # Agent(LearningDecisionMaker(env.action_space))  # can use diffrent DM
                        for agent_name in env_agents}


"""
Simple use of communication network - 
    - build one using COM_net() - defualt architecture is - all masseges sent to all other agents
            *U can use more options - see at COM_net() class doc.
"""
com = COM_net()
""" 
    - initailze our new controller (DecentralizedComController) - using our env, our agents and our com module
        - this controller will perform all joint action and message delieveries at any time-step
"""
controller = DecentralizedComController(environment, decentralized_agents, com)
"""
activate 
"""
controller.run(render=True, max_iteration=3)





"""
part 2 - 
com_emerge_use of message for getting better + introducing com-module
"""
from src.decision_makers.planners.MA_com_planner import Astar_message_DM



"""
Builds Multi_taxi env
"""
m = MAP
env = MultiTaxiEnv(num_taxis=3, num_passengers=5, domain_map=m, observation_type='symbolic',rewards_table=TAXI_pickup_dropoff_REWARDS ,option_to_stand_by=True)
obs = env.reset()

environment = build_multi_env(env)
#
print('EnvironmentWrapper created')

# making agent class that communicates and heads towards 1 passenger (pickup->dropoff)
"""
in order to use com module: 
    - implement Agent_com class (inherit from Agent_com
    - make sure to implement set_data_func - that decides what is the data that the agent will transmit whenever it called
    - u can implement your recieve_func - that decides what to do with a recieved data
        defualt is the union_func that add the message data to the observation 
"""

class Heading_message_agent(Agent_Com):

    def __init__(self, decision_maker : Astar_message_DM , sensor_function =None, message_filter = None, AgentName = None, bandW = math.inf, union_recieve = False):
        super().__init__(decision_maker , sensor_function, message_filter, AgentName, bandW, union_recieve)
        self.last_action = None
        self.last_message = None

    def set_data_func(self, obs):
        data = (self.decision_maker.taking_passenger,len(self.decision_maker.active_plan))
        return data

    # implement our recive_func
    def set_recive_func(self, obs, message):
        self.last_message = message
        self.decision_maker.save_last_message(message)
        # self.decision_maker.updateplan_message(message)

    # saves last action of the agent - not necessary for com module
    def set_last_action(self, action):
        self.last_action = action
#
#


# after having our com-Astar-agents class we can set our env agents into a dicentralized_agents dict
env_agents = environment.get_env_agents()
decentralized_agents = {agent_name: Heading_message_agent(Astar_message_DM(env ,single_plan=True, Taxi_index=int(agent_name[-1]), domain_map=m) ,AgentName=agent_name)             # Agent(LearningDecisionMaker(env.action_space))  # can use diffrent DM
                        for agent_name in env_agents}


"""
Simple use of communication network - 
    - build one using COM_net() - defualt architecture is - all masseges sent to all other agents
            *U can use more options - see at COM_net() class doc.
"""
com = COM_net()
""" 
    - initailze our new controller (DecentralizedComController) - using our env, our agents and our com module
        - this controller will perform all joint action and message delieveries at any time-step
"""
controller = DecentralizedComController(environment, decentralized_agents, com)
"""
activate 
"""
#communicate first
controller.send_recieve()

#run (communication inside after each time_click)
controller.run(render=True, max_iteration=250,reset=True)
print("Thats all - part 2")

"""
part 2.2 - 
agents learn now - hierarchical_tasks
"""

# import hierarchical_tasks multi taxi env
from src.environments.hirarchical_Wrapper import Multi_Taxi_Task_Wrapper
from src.decision_makers.planners.Com_High_level_Planner import Astar_message_highlevel_DM
"""
Builds Multi_taxi env
"""
m = MAP
env = MultiTaxiEnv(num_taxis=3, num_passengers=5, domain_map=m, observation_type='symbolic',rewards_table=TAXI_pickup_dropoff_REWARDS ,option_to_stand_by=True)

obs = env.reset()

environment = build_multi_env(env)

environment =  Multi_Taxi_Task_Wrapper(environment)
#
print('EnvironmentWrapper Multi_Taxi_Task_Wrapper created')

# making agent class that communicates and heads towards 1 passenger (pickup->dropoff)

class Heading_message_agent(Agent_Com):

    def __init__(self, decision_maker : Astar_message_highlevel_DM , sensor_function =None, message_filter = None, AgentName = None, bandW = math.inf, union_recieve = False):
        super().__init__(decision_maker , sensor_function, message_filter, AgentName, bandW, union_recieve)
        self.last_action = None
        self.last_message = None

    def set_data_func(self, obs):
        data = (self.decision_maker.taking_passenger,len(self.decision_maker.active_plan))
        return data

    # implement our recive_func
    def set_recive_func(self, obs, message):
        self.last_message = message
        self.decision_maker.save_last_message(message)
        # self.decision_maker.updateplan_message(message)

    # saves last action of the agent - not necessary for com module
    def set_last_action(self, action):
        self.last_action = action


# run again - high-level
env_agents = environment.get_env_agents()
decentralized_agents = {agent_name: Heading_message_agent(Astar_message_highlevel_DM(env ,single_plan=True, Taxi_index=int(agent_name[-1]), domain_map=m) ,AgentName=agent_name)             # Agent(LearningDecisionMaker(env.action_space))  # can use diffrent DM
                        for agent_name in env_agents}


"""
Simple use of communication network - 
    - build one using COM_net() - defualt architecture is - all masseges sent to all other agents
            *U can use more options - see at COM_net() class doc.
"""
com = COM_net()
""" 
    - initailze our new controller (DecentralizedComController) - using our env, our agents and our com module
        - this controller will perform all joint action and message delieveries at any time-step
"""
controller = DecentralizedComController(environment, decentralized_agents, com)
"""
activate 
"""
#communicate first
controller.send_recieve()

#run (communication inside after each time_click)
controller.run(render=True, max_iteration=250,reset=True)

# SHOW REWARDS
reward = controller.total_rewards
totals = {}
for r in reward[0]:
    totals[r] = 0
total=0
for r in reward:
    for key, value in r.items():
        totals[key]+=value
        total+=value
print(f"----------------------------------\n total reward of all agents: {total}, {totals} \n----------------------------------")
print("Thats all - part 2.2")

"""
part 3 - 
agents learn now - high level PPO, low-level a-star
"""

from src.decision_makers.High_level_learner import LearningHighLevelDecisionMaker

env = MultiTaxiEnv(num_taxis=1, num_passengers=2, domain_map=MAP2, observation_type='symbolic', option_to_stand_by=True)
env.agents = env.taxis_names
env = EnvWrappper(env, env.agents)
env = Multi_Taxi_Task_Wrapper(env)
obs = env.reset()

D_M = LearningHighLevelDecisionMaker(env.action_space)
obs = env.reset()
env.env.env.render()
for i in range(50):
    print(f"obs:{type(obs)}")
    a = D_M.get_action(obs)
    print(f"next action: {env.index_action_dictionary[a]}")
    obs, r, done, info = env.step(a)
    env.env.env.render()
    if done: break
print("that all - part 3")

"""
part 4 - 
speaker-listener com as action
speaker - speaks right if used, listener can ask for 'help'
#### Listener new-wrapper :
The speaker observation is of type `Box(-inf, inf, (9,), float32)` 
1. listener agent velocity X
2. listener agent velocity Y
3. red landmark X pos - listener agent X pos
4. red landmark Y pos - listener agent Y pos
5. blue landmark X pos - listener agent X pos
6. blue landmark Y pos - listener agent Y pos
7. green landmark X pos - listener agent X pos
8. green landmark Y pos - listener agent Y pos
9. communication channel 1 - if com action was choosen last - speaker tells destination (1/2/3) else - 0
Discrete ACTIONS:
* 0 - do nothing and ask for communicate (com penalty)
* 1 - push left (add velocity in negative x-axis direction)
* 2 - push right (add velocity in positive x-axis direction)
* 3 - push down (add velocity in negative y-axis direction)
* 4 - push up (add velocity in positive t-axis direction)
"""

import numpy as np
from gym import Wrapper
from copy import deepcopy
from pettingzoo.mpe import simple_speaker_listener_v3
from stable_baselines3 import PPO

# speaker action-to-index dict
SPEAKER_DISCRETE_ACTIONS = {

    'A': 0,
    'B': 1,
    'C': 2,
    'nothing' : 3,

}

# listener action-to-index dict
LISTENER_DISCRETE_ACTIONS = {
    'nothing': 0,
    'left':    1,
    'right':   2,
    'down':    3,
    'up':      4
}

from gym.spaces import Discrete, Box

"""#### Custom gym Wrapper
"""
class ListenerOnlyCOMWrapper(Wrapper):

    def __init__(self, env, com_allways_on = False, com_reward = -0.5):
        super().__init__(env)
        self.com_allways_on = com_allways_on
        self.com_reward = com_reward

        # reset to skip speaker before new game
        self.obs = self.reset()

        # set single agent list
        self.agents = self.agents[1:]

        self.observation_space = self.get_observation_space()

        #set action space (listener)
        self.action_space = env.action_spaces[self.agents[0]]

    def get_observation_space(self):
        return Box(low=-np.inf,high=np.inf , shape= (len(self.obs),), dtype=np.float32)

    def reset(self):
        super().reset()


        # skip speaker action
        # self.__step_speaker()
        # TODO - fix if discrete
        self.__step_speaker()
        self.obs,_,_,_ = self.env.last()
        self.obs = self.fix_obs(self.obs, self.com_allways_on)
        return self.obs

    def step(self, action):
        # CHANGE ACTION AS NEEDED
        com = action == 0
        super().step(action)
        ob, _, done, _ = self.env.last()  # do listener action

        # skip speaker action - if com==True, speaker 'speaks' else - 0 speak will be performed
        self.__step_speaker()

        step_rets = self.env.last()

        if self.com_allways_on:  com = True

        self.obs = self.fix_obs(step_rets[0], com)

        reward_com = 0
        if com:
            # get -0.5 reward for com use TODO adjust reward if needed
            if not self.com_allways_on: reward_com = self.com_reward
        else:
            self.obs[-1] = SPEAKER_DISCRETE_ACTIONS['nothing']

        return (self.obs,step_rets[1] + reward_com, step_rets[2], step_rets[3])


    # def step(self, action):
    #     # CHANGE ACTION AS NEEDED
    #     com = action == 0
    #     super().step(action)
    #     _, _, done, _ = self.env.last()  # do listener action
    #     # step_rets = super().step(action)
    #
    #     # unpack step return values from their dictionaries
    #     # step_rets = tuple(next(iter(ret.values())) for ret in step_rets)
    #     if done: return None
    #     if com:
    #         # skip speaker action - if com==True, speaker 'speaks' else - 0 speak will be performed
    #         self.__step_speaker()
    #     else:
    #         # TODO - fix if discrete
    #         super().step(SPEAKER_DISCRETE_ACTIONS['A'])
    #     self.obs, _, _, _ = self.env.last()  # do listener action



    def fix_obs(self, obs, com):
        obs = np.array(obs[:-2])
        if not com: obs[-1] = SPEAKER_DISCRETE_ACTIONS['nothing']
        obs[-1] = (obs[-1]+1) % (len(SPEAKER_DISCRETE_ACTIONS))
        return obs


    def __step_speaker(self):
        goal_color, _, done, _ = self.env.last()

        # speaker is done before the listener.
        if done:
            return

        # step with the correct action type
        if self.env.unwrapped.continuous_actions:
            super().step(goal_color)
        else:
            super().step(np.argmax(goal_color))

env = simple_speaker_listener_v3.env(max_cycles=200, continuous_actions=False)
env = ListenerOnlyCOMWrapper(env, com_allways_on=False)
print(f'custom wrapped environment: {env}')

#train PPO agent method
def train_model(env):
    env_copy = deepcopy(env)
    env_copy.reset()
    model = PPO("MlpPolicy", env_copy, verbose=1,gamma=0.97)
    for batch in range(200):
        print(f"batch: {batch}")
        model.learn(total_timesteps=8000, n_eval_episodes=40)
        env_copy.reset()
    return model

def get_ppo_agent(is_com_active):
    if is_com_active:
        model_file_name = 'S-L_PPO_base'
    else:
        model_file_name = 'S-L_PPO_com2'
    try:
        model = PPO.load(model_file_name)
    except:
        model = train_model(env)
        model.save(model_file_name)
    return model

# run (n_iter) episodes after training and render
def run(env,model,n_iter):
    for m in range(n_iter):
        observation = env.reset()
        env.render()
        total_rewards = 0
        for i in range(env.unwrapped.max_cycles):
            # choose an action and execute
            action = model.predict(observation)
            action = action[0]
            print(f'step {i}')
            print(f'observation: {observation}')
            observation, reward, done, info = env.step(action)

            # log everything
            c = "no_com"
            if action == 0 or env.com_allways_on: c = "com active"
            print(f'action:      {action}  com:{c}')
            print(f'reward:      {reward}')
            print()

            # if done, the episode is complete. no more actions can be taken
            if done:
                break

            env.render()
        print(f"total rewards: {total_rewards}")
        env.close()

model = get_ppo_agent(env.com_allways_on)

run(env,model,10)

print("that all - part 4")
