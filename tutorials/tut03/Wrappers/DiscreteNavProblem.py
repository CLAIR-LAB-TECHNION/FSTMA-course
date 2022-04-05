from copy import deepcopy

from AI_agents.Search import utils
from AI_agents.Search.problem import Problem
# listener action-to-index dict
NAVIGATOR_DISCRETE_ACTIONS = {
    'nothing': 0,
    'left':    1,
    'right':   2,
    'down':    3,
    'up':      4
}

# nav_state(State):
# key will be ((new_idx_coords),env_copy)

#    def get_key(self):
#        return self.key[0]


class DiscreteNavProblem(Problem):
    # initial state is the starting state_key AKA our new node coordinates
    def __init__(self, navigator, initial_state, constraints):
        super().__init__(initial_state, constraints)
        self.navigator = navigator
        self.counter = 0

    # get the actions that can be applied at the current node
    def get_applicable_actions(self, node):
        return NAVIGATOR_DISCRETE_ACTIONS.values()

    # get (all) succesor states of an action and their
    def get_successors(self, action, node):

        navigator_copy = deepcopy(node.state)
        navigator_copy.env.step(action)
        return [utils.Node(navigator_copy, node.state, action, self.get_action_cost(action, node.state))]

    def get_action_cost(self, action, state):
        return 1

    def is_goal_state(self, state):
        if state.is_terminal:
            return True
        else:
            return False

    def apply_action(self, action):
        state, reward, done, info = self.env.step(int(action))
        return [state, reward, done, info]
