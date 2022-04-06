# use a different wrapper base class for parallel environments
from pettingzoo.utils.wrappers import BaseParallelWraper


class SingleAgentParallelEnvGymWrapper(BaseParallelWraper):
    """
    A wrapper for single-agent parallel environments aligning the environments'
    API with OpenAI Gym.
    """

    def reset(self):
        # run `reset` as usual.
        # returned value is a dictionary of observations with a single entry
        obs = self.env.reset()

        # return the single entry value as is.
        # no need for the key (only one agent)
        return next(iter(obs.values()))

    def step(self, action):
        # step using "joint action" of a single agnet as a dictionary
        step_rets = self.env.step({self.env.agents[0]: action})

        # unpack step return values from their dictionaries
        return tuple(next(iter(ret.values())) for ret in step_rets)

    @property  # make property for gym-like access
    def action_space(self, _=None):  # ignore second argument in API
        # get action space of the single agent
        return self.env.action_space(self.env.possible_agents[0])

    @property  # make property for gym-like access
    def observation_space(self, _=None):  # ignore second argument in API
        # get observation space of the single agent
        return self.env.observation_space(self.env.possible_agents[0])
