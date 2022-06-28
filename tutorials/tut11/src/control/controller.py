from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy
from gym.spaces import Box


class Controller(ABC):
    """An abstract controller class, for other controllers
    to inherit from
    """

    # init agents and their observations
    def __init__(self, environment, agents):
        self.environment = environment
        self.agents = agents
        self.__is_image_obs = isinstance(self.environment.get_env().observation_space, Box)

    def run(self, render=False, max_iteration=None):
        """Runs the controller on the environment given in the init,
        with the agents given in the init

        Args:
            render (bool, optional): Whether to render while runngin. Defaults to False.
            max_iteration ([type], optional): Number of steps to run. Defaults to infinity.
        """
        done = False
        index = 0
        observation = self.environment.get_env().reset()
        self.total_rewards = []
        while done is not True:
            index += 1
            if max_iteration is not None and index > max_iteration:
                break

            # get actions for each agent to perform
            joint_action = self.get_joint_action(observation)

            # display environment
            if render:
                env_str = self.environment.get_env().render()
                # todo check obs type - if image False, symbolic-True
                self.render_obs_next_action(joint_action,observation,not self.__is_image_obs)

            # perform agents actions
            observation, reward, done, info = self.perform_joint_action(joint_action)
            self.total_rewards.append(reward)
            done = all(value == True for value in done.values())
            if done:
                break

        if render:
            self.environment.get_env().render()

    def perform_joint_action(self, joint_action):
        return self.environment.step(joint_action)
        # return self.environment.get_env().step(joint_action)

    def get_joint_action(self, observation):
        pass

    def render_obs_next_action(self, joint_action, observation, symbolic=True):

        # if  isinstance(list(observation.values())[0], numpy.ndarray) :

        if symbolic:

            for agent_name in self.agents.keys():
                print(f"{agent_name} obs:\n {observation[agent_name]} , action: {joint_action[agent_name]}")
                # print(f"{agent_name} obs:\n {observation[agent_name]} , action: {self.environment.get_env().index_action_dictionary[joint_action[agent_name]]}")
        else:
            print(f"{ type(list(observation.values())[0])}")
            fig = plt.figure(figsize=(16,4))
            i = 1
            for agent_name in self.agents.keys():
                title = str(agent_name) + ": " + str(self.environment.get_env().index_action_dictionary[joint_action[agent_name]])

                ax = fig.add_subplot(1, len(self.agents), i)
                i += 1
                plt.title(title)
                ax.imshow(observation[agent_name])
            plt.show()