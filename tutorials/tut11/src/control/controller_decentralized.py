from .controller import Controller




class DecentralizedController(Controller):

    def __init__(self, env, agents):
        # initialize super class
        super().__init__(env, agents)

    def get_joint_action(self, observation):
        """Returns the joint action

        Args:
            observation (dict): the current observatins

        Returns:
            dict: the actions for the agents
        """

        joint_action = {}
        for agent_name in self.agents.keys():
            print(f"{observation[agent_name]}")       # check use
            action = self.agents[agent_name].get_decision_maker().get_action(observation[agent_name])
            joint_action[agent_name] = action
            # print(f"agent:{agent_name}, action: {action}\n")

        # if self.render:
        #     self.render_obs_next_action(joint_action,observation)

        return joint_action

    # def render_obs_next_action(self, joint_action, observation):
    #     fig = plt.figure(figsize=(17,4))
    #     i=1
    #     temp_taxi_env = TaxiEnv(num_taxis=3, observation_type="image")
    #     for agent_name in self.agents.keys():
    #         title = str(agent_name) + ": " + str(temp_taxi_env.index_action_dictionary[joint_action[agent_name]])
    #
    #         ax = fig.add_subplot(1, len(self.agents), i)
    #         i += 1
    #         plt.title(title)
    #         ax.imshow(observation[agent_name])
    #     plt.title(title)
    #     print(plt)
    #
    #     plt.show()

