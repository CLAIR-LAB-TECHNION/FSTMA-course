import math
from abc import ABC, abstractmethod

import numpy
import numpy as np

from src.Communication.COM_net import Message


class Agent:

    def __init__(self, decision_maker, sensor_function =None, message_filter = None, AgentName = None):
        self.decision_maker = decision_maker
        self.sensor_function = sensor_function
        self.message_filter = message_filter
        self.agent_name = AgentName

    def get_decision_maker(self):
        return self.decision_maker

    def get_observation(self, state):

        return self.sensor_function(state)


class Agent_Com(Agent, ABC):

    def __init__(self, decision_maker, sensor_function =None, message_filter = None, AgentName = None, bandW = math.inf, union_recieve = True):
        super().__init__(decision_maker, sensor_function, message_filter, AgentName)
        self.max_bandW = bandW
        self.union_recieve = union_recieve

    def check_send_data(self, data):
        # todo fix BandW
        # assert data.__sizeof__() > self.max_bandW, f"size of message :{data.__sizeof__()} greater than bandW : {self.max_bandW}"
        return data

    @abstractmethod
    def set_data_func(self, obs=None):
        pass

    def transmit(self,obs = None, **kwargs):
        data = self.set_data_func(obs)
        self.check_send_data(data)
        m = Message(data=data,author=self.agent_name)
        return m

    def union_Recieve_func(self, obs : numpy.ndarray, messages : list):
        # print(f"{type(messages)}")
        all_m = [m.data for m in messages]
        obs = numpy.append(obs,np.array(all_m))
        return obs

    def recieve(self, obs, message):
        self.check_send_data(message)

        #if self.message_filter!=None:
            # use for filter message

        if self.union_recieve:
            try:
                data = self.union_Recieve_func(obs, message)
                # print("message recieved as [obs+message]")
            except:
                data = [obs,message]
                # print("message recieved as [obs,message]")
        else:
            data = self.set_recive_func(obs, message)
        self.new_obs = data
        return data

    def set_recive_func(self, obs, message):
        pass


class Action_message_agent(Agent_Com):

    def __init__(self, decision_maker, sensor_function =None, message_filter = None, AgentName = None, bandW = math.inf, union_recieve = True):
        super().__init__(decision_maker, sensor_function, message_filter, AgentName, bandW, union_recieve)
        self.last_action = None

    def set_last_action(self, action):
        self.last_action = action

    def set_data_func(self, obs):
        data = self.last_action
        return data


"""
An abstract class for choosing an action, part of an agent.
(An agent can have one or several of these)
"""
class DecisionMaker:

    def __init__(self):
        pass

    def get_action(self, observation):
        pass

    """
    Functions for training:
    """
    def get_train_action(self, observation):
        pass

    def update_step(self, obs, action,new_obs, reward, done):
        pass

    def update_episode(self, batch_size=0):
        pass

class RandomDecisionMaker:
    def __init__(self, action_space):
        self.space = action_space

    def get_action(self, observation):
        if type(self.space) == dict:
            return {agent: self.space[agent].sample() for agent in self.space.keys()}
        else:
            return self.space.sample()


