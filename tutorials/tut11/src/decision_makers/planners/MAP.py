import numpy as np
from multi_taxi import MultiTaxiEnv as TaxiEnv
from ai_dm.Search.best_first_search import best_first_search, breadth_first_search, a_star
from src.decision_makers.planners.problem import Problem
import ai_dm.Search.utils as utils
import ai_dm.Search.heuristic
from IPython.display import display, clear_output
import time
from copy import deepcopy
import itertools

# from MaHeuristic import MultiAgentsHeuristic, taxi_heuristic


class MapProblem(Problem):
    """Problem superclass
       supporting COMPLETE
    """
    def __init__(self, env, init_state, heur_func=None, constraints=[]):
        super().__init__(init_state, constraints)
        self.env = env
        self.num_taxis = self.env.num_taxis
        self.taxis_available = {taxi_id: "taxi_" + str(taxi_id) for taxi_id in range(self.num_taxis)}
        self.heur = heur_func

    def get_state(self):
        return list_to_tuple(self.env.state)

    def set_state(self, state):
        self.env.state = tuple_to_list(state)

    def get_action(self, action_str):   #action_str(tuple) - action_str[i] represent the taxi_i action
        return {v: action_str[i] for i, v in enumerate(self.taxis_available.values())}

    def step(self, action_str):
        return self.env.step(self.get_action(action_str))

    def get_action_cost(self, action, state):
        return 1

    def is_goal_state(self, state):
        state.is_terminal = sum(state.key[4]) == self.num_taxis
        return state.is_terminal

    """
    a comparison between 2 values of the heuristic
    param match1: one value
    param match2: other value
    return: True if match 1 <= match 2. otherwise false
    """

    def is_better_or_equal(self, node_a, node_b):

        eval_node_a = self.heur(node_a)
        eval_node_b = self.heur(node_b)
        sorted_values_a = sorted(eval_node_a[1], reverse=True)
        sorted_values_b = sorted(eval_node_b[1], reverse=True)
        path_cost_a = node_a.path_cost
        path_cost_b = node_b.path_cost
        for i in range(len(eval_node_a[1])):
            if sorted_values_a[i] + path_cost_a < sorted_values_b[i] + path_cost_b:
                return True
            elif sorted_values_a[i] + path_cost_a > sorted_values_b[i] + path_cost_b:
                return False
        return True

    def is_state_equal(self, state1, state2):
        taxis1 = state1[0]
        taxis2 = state2[0]
        status1 = state1[4]
        status2 = state2[4]

        for t1, t2, s1, s2 in zip(taxis1, taxis2, status1, status2):
            if t1 == t2:
                # status changed from on_taxi -> has_arrived
                if s1 != s2 and s2 == 1:

                    return False
                # unexecuted pickup or dropoff
                if status1 == status2 or s2 == 2:
                    return True

        return False

    def can_do_action_from_state(self, state, action_str):
        action = self.get_action(action_str)

        self.set_state(state)
        obs, _, done, _ = self.env.step(action)
        new_state = self.get_state()
        if done['__all__']:
            self.env.reset()
            self.set_state(state)
        if self.is_state_equal(state, new_state):
            return False
        else:
            return True

    def apply_action(self, state, action):
        self.set_state(state)
        _, reward, done, info = self.step(action)
        new_state = self.get_state()
        return new_state, reward, done, info

    def evaluate(self, node):
        val = node.path_cost
        if self.heur is not None:
            val += np.array(self.heur(node))
        return val

    # get the actions that can be applied at the current node
    def get_applicable_actions(self, node):
        # action_list = [i for i in range(4)]
        actions = self.env.available_actions_indexes

        return [joint_action for joint_action in itertools.product(actions, repeat=self.num_taxis)
                if self.can_do_action_from_state(node.state.key, joint_action)]

    # get successor state of an action
    def get_successors(self, action, node):
        next_state, reward, done, info = self.apply_action(node.state.key, action)
        next_state = utils.State(next_state, False)
        for a in action:
            if a > 4:
                next_state.is_terminal = self.is_goal_state(next_state)
        return [
            utils.Node(next_state, parent=node, action=action, path_cost=node.path_cost + 1)]


def list_to_tuple(x):
    lolol = deepcopy(x)
    result = list()
    while lolol:
        lol = lolol.pop(0)
        if type(lol[0]) is not list:
            result.append(tuple(lol))
        else:
            local_res = list()
            while lol:
                l = lol.pop(0)
                local_res.append(tuple(l))
            result.append(tuple(local_res))

    return tuple(result)


def tuple_to_list(x):
    lolol = list(x)
    result = list()
    while lolol:
        lol = lolol.pop(0)
        if type(lol[0]) is not tuple:
            result.append(list(lol))
        else:
            lol = list(lol)
            local_res = list()
            while lol:
                l = lol.pop(0)
                local_res.append(list(l))
            result.append(list(local_res))

    return result

def run_plan(state,plan,problem):
    new_state = deepcopy(state)
    problem.env.reset()
    problem.set_state(state)
    problem.env.render()
    for action in plan:
        time.sleep(0.25)
        new_state = problem.step(eval(action))
        clear_output(wait=True)
        problem.env.render()

    return new_state
#
# MAP2 = [
#     "+-------+",
#     "| : |F: |",
#     "| : | : |",
#     "| : : : |",
#     "| | :G| |",
#     "+-------+",
# ]
# taxi_env = TaxiEnv(num_taxis=2, num_passengers=2, domain_map=MAP2)
#
# # Make sure it works with our API:
# h = MultiAgentsHeuristic(taxi_heuristic)
# initial_state=deepcopy(taxi_env.state)
# map_problem = MapProblem(taxi_env, list_to_tuple(initial_state))
# # plan = a_star(problem=map_problem,)
# taxi_env.render()
#
# # plan = breadth_first_search(map_problem)
# plan = a_star(problem=map_problem, heuristic_func=h)
# [best_value, best_node, path, explored_count, ex_terminated] = plan
# final_state = run_plan(initial_state, path, map_problem)
#
# print(plan)
# taxi_env.reset()
# map_problem.set_state(best_node.state.key)
# taxi_env.render()

# taxi_env.reset()
# map_problem.set_state(initial_state)
# taxi_env.render()
# map_problem = MapProblem(taxi_env, list_to_tuple(initial_state), heur_func=h)
# plan = a_star(problem=map_problem,heuristic_func=h)
# [best_value, best_node, path, explored_count, ex_terminated] = plan
# print(plan)
# taxi_env.reset()
# map_problem.set_state(best_node.state.key)
# taxi_env.render()