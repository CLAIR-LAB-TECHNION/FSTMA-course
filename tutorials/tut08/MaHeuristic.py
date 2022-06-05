import random

import numpy as np
from multi_taxi import MultiTaxiEnv
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
import AI_agents.Search.utils as utils


def allocate_tasks(values_mat):
    """
    param values_mat: a matrix A, s.t Aij is the value of agent i, when his task is j.
    return the best makespan match v and its value. vi is the task of the agent i in the match.
    """

    tasks_num = values_mat.shape[1]

    sorted_rewards = sorted(values_mat.reshape(-1))

    low = 0
    high = len(sorted_rewards) - 1

    match = [0]

    while high > low:
        mid = int((high + low) / 2)

        reward_mat_copy = values_mat.copy()
        reward_mat_copy[reward_mat_copy > sorted_rewards[mid]] = 0
        match = maximum_bipartite_matching(csr_matrix(reward_mat_copy), perm_type='column')
        if np.sum(match[match > 0]) < tasks_num:
            low = mid + 1
        else:
            high = mid

    weights = [values_mat[i][match[i]] for i in range(len(match)) if match[i] >= 0]
    match = [match[i[0]] for i in sorted(enumerate(weights), reverse=True, key=lambda x: x[1])]
    return np.array(sorted(weights, reverse=True)), tuple(match)


class MultiAgentsHeuristic:
    def __init__(self, single_agent_heuristic):
        self.h = single_agent_heuristic

    def __call__(self, node):
        """
        return an object that presents the joint heuristic of this state, by using the heuristic of one agent and one task
        """
        state = node.state.key
        taxis_src = state[0]
        passengers_src = state[2]
        passengers_dst = state[3]
        passengers_status = state[4]
        values_mat = np.array([[self.h(taxi_src, passenger_src, passenger_dst, passenger_status)
                                for passenger_src, passenger_dst, passenger_status
                                in zip(passengers_src, passengers_dst, passengers_status)]
                               for taxi_src in taxis_src])

        values, match = allocate_tasks(values_mat)
        return tuple(values + node.path_cost)  #, match


def manhattan_distance(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def taxi_heuristic(taxi_src, passenger_src, passenger_dst, passenger_status):
    """
    manhatten distance to from the taxi's source to the passenger's source, and from there to the passenger's destination
    """
    is_waiting = passenger_status == 2
    not_waiting = passenger_status != 2
    has_arrived = passenger_status == 1
    not_arrived = passenger_status != 1
    in_taxi = passenger_status > 2
    return (manhattan_distance(taxi_src, passenger_src) + manhattan_distance(passenger_src, passenger_dst)) * is_waiting \
           + manhattan_distance(taxi_src, passenger_dst) * in_taxi + (5 - 3*has_arrived - 2*not_waiting)


# if __name__ == '__main__':
#     env_instance = MultiTaxiEnv(num_taxis=2,  # 3 taxi agents in the environment
#                                 num_passengers=2,  # 2 passengers in the environment
#                                 max_fuel=None,  # taxi1 has a capacity of 30 fuel units, tax2 has 50, and taxi3 has 25
#                                 taxis_capacity=None,  # unlimited passenger capacity for all taxis
#                                 option_to_stand_by=False,
#                                 # taxis can turn the engin on/off and perform a standby action
#                                 observation_type='symbolic',  # accepting symbolic (vector) observations
#                                 can_see_others=True)  # cannot see other taxis (but also not collision sensitive)
#
#     env_instance.render()
#     state = env_instance.state
#     print(state)
#
#     match = allocate_tasks(np.array([[7, 8, 2], [1, 9, 3], [3, 8, 12], [4, 7, 2]]))
#     h = MultiAgentsHeuristic(taxi_heuristic)
#     node = utils.Node(state=utils.State(state, False), parent=None, action=None, path_cost=1)
#    print(h(node))
