"""
Wrappers for Multi-Taxi domain
changing actions and tasks types of decoders
"""
import queue
from copy import deepcopy

from ai_dm.Search.best_first_search import a_star
from colorama import Fore
from gym import Wrapper, RewardWrapper, ActionWrapper, ObservationWrapper
from gym.spaces import MultiDiscrete, Discrete, Box, MultiBinary
# 'Deep Model Related Imports'
from multi_taxi.taxi_environment import MAP2, MAP
from torch.nn.functional import one_hot
import torch
import numpy as np
from multi_taxi import MultiTaxiEnv
from src.decision_makers.planners.MAP import MapProblem
from src.environments.env_wrapper import EnvWrappper , MAP
from collections import deque


Pass_color_dict = {
    0 : 'None' ,
    1 : 'Yellow' ,
    2 : 'Red' ,
    3 : 'White' ,
    4 : 'Green' ,
    5 : 'Cyan' ,
    6 : 'Blue' ,
    7 : 'Magneta'
}

Text_color = {
    0 : Fore.BLACK ,
    1 : Fore.YELLOW ,
    2 : Fore.RED ,
    3 : Fore.WHITE,
    4 : Fore.GREEN ,
    5 : Fore.CYAN,
    6 : Fore.BLUE,
    7 : Fore.MAGENTA
}


"""
In order to be compatible with both AI_agents library and the TaxiEnvironment we have to switch the state represntation between list of lists and tuple of tuples
"""

def list_to_tuple(x):
    lolol = deepcopy(x) # list of lists of lists
    result = list()
    while lolol:
        lol = lolol.pop(0)
        # single list
        if type(lol[0]) is not list:
            result.append(tuple(lol))

        # list of lists
        else:
            local_res = list()
            while lol:
                l = lol.pop(0)
                local_res.append(tuple(l))
            result.append(tuple(local_res))

    return tuple(result)


def tuple_to_list(x):
    totot = list(x)
    result = list()
    while totot:
        tot = totot.pop(0)
        if type(tot[0]) is not tuple:
            result.append(list(tot))
        else:
            tot = list(tot)
            local_res = list()
            while tot:
                t = tot.pop(0)
                local_res.append(list(t))
            result.append(list(local_res))

    return result

"""run plan in the adapted simulation"""
def run_plan(state, plan, problem):
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

def str_to_int(s):
    return int(''.join(x for x in s if x.isdigit()))

def plan_2_pass(plan):
    """
    takes a plan and return a tuple of (pass that is taking, lenght)
    """
    p = str_to_int(plan[-1])-4
    l = len(plan)
    return (p,l)

def plan_2_color_pass(plan):
    """
    takes a plan and return a tuple of (pass that is taking, lenght)
    """
    p = str_to_int(plan[-1])-4
    l = len(plan)
    return (Pass_color_dict[p],l)




"""
solves Astar
generate simulation in order to solve single plan of a taxi and return the plan
"""

def joint_simulation(state, h, print_simulation=False, domain_m=MAP2):
    taxi_P = MultiTaxiEnv(num_taxis=len(state[0]), num_passengers=len(state[2]), domain_map=domain_m)
    map_p = MapProblem(taxi_P, list_to_tuple(state))
    map_p.set_state(state)

    [_, _, path, explored_count, _] = a_star(problem=map_p, heuristic_func=h)
    if print_simulation:
        run_plan(state,path,map_p)
        print('explored_count:', explored_count)
        print('path length:', len(path))
    return path

"""### But what about the heuristic?
Heuristics
"""

class MultiAgentsHeuristic:
    def __init__(self, single_agent_heuristic, aggr_func ):
        self.h = single_agent_heuristic
        self.aggr_func = aggr_func

    def __call__(self, node):
        """
        return an object that presents the joint heuristic of this state, by using the heuristic of one agent and one task
        """
        state = node.state.key
        taxis_src = state[0]
        passengers_src = tuple_to_list(state[2])
        passengers_dst = tuple_to_list(state[3])
        passengers_status = state[4]
        # # remove passengered that reached dest
        # for i in range(len(passengers_status)-1,-1,-1):
        #     if passengers_status[i]==1 :
        #         del passengers_src[i]
        #         del passengers_dst[i]
        # passengers_src = list_to_tuple(passengers_src)
        # passengers_dst = list_to_tuple(passengers_dst)
        if len(taxis_src) == 1:
            if len(passengers_src)==0: return 0
            m = min(manhattan_heuristic(0,taxis_src[0],passengers_src[i],passengers_dst[i],passengers_status[i]) for i in range(len(passengers_src)))
            return m + node.path_cost
        values_mat = np.array([[self.h(taxi_id, taxi_src, passenger_src, passenger_dst, passenger_status)
                                for passenger_src, passenger_dst, passenger_status
                                in zip(passengers_src, passengers_dst, passengers_status)]
                               for taxi_id, taxi_src in enumerate(taxis_src)])

        # values, match = allocate_tasks(values_mat)
        g_score = self.aggr_func(values_mat) + node.path_cost
        return g_score


# def manhatten_dist1(r1, c1, r2, c2):
#     # calssic manhatten dist |row1 - row2| + |col1 - col2|
#     return abs(r1 - r2) + abs(c1 - c2)
#
#
# def manhatten_heuristic1(node,env):
#     # decode state integer to interpretable values
#     taxi_row, taxi_col, passenger_idx, dest_idx = env.decode(node.state.get_key())
#     locs = env.unwrapped.locs
#
#     # split to 2 cases where the passenger is in the taxi and not in the taxi.
#     if passenger_idx == 2:
#         # dist from the taxi to the destination
#         return manhatten_dist1(taxi_row, taxi_col, *locs[dest_idx]) + 1  # include dropoff
#     elif passenger_idx == dest_idx:
#         # passenger has reached the destination. this is a goal state
#         return 0
#     else:
#         # dist from the taxi to the passenger and from the passenger to the destination
#         passenger_dist = manhatten_dist1(taxi_row, taxi_col, *locs[passenger_idx])
#         dest_dist = manhatten_dist1(*locs[passenger_idx], *locs[dest_idx])
#         return passenger_dist + dest_dist + 2  # include pickup and dropoff actions

""" transform obs of a taxi into a state for search"""
def obs_to_state(obs):
    state = []
    if isinstance(obs,dict):
        obs_list = obs['taxi_0']
    else: obs_list = obs
    num_pass = (len(obs_list)-2)//5
    state.append([tuple(obs_list[0:2])])
    state.append([numpy.inf])
    state.append([])
    for i in range(2,2+2*num_pass,+2):
        state[2].append([obs_list[i],obs_list[i+1]])
    state.append([])
    for i in range(2+2*num_pass,2+4*num_pass,+2):
        state[3].append([obs_list[i],obs_list[i+1]])
    state.append(tuple(obs_list[2+4*num_pass:]))
    return state

# calssic manhatten dist |row1 - row2| + |col1 - col2| from 2 positions
def manhattan_distance(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])

def manhattan_heuristic(taxi_id, taxi_src, passenger_src, passenger_dst, passenger_status):
    """
    manhatten distance to from the taxi's source to the passenger's source, and from there to the passenger's destination
    """
    is_waiting = passenger_status == 2
    not_waiting = passenger_status != 2
    has_arrived = passenger_status == 1
    not_arrived = passenger_status != 1
    in_taxi = taxi_id + 3 == passenger_status
    return (manhattan_distance(taxi_src, passenger_src) + manhattan_distance(passenger_src, passenger_dst)) * is_waiting \
           + manhattan_distance(taxi_src, passenger_dst) * in_taxi + (2 - has_arrived - not_waiting)

def closest_passenger(values_mat):
    return np.min(values_mat)
#
# """same state?"""
# def is_state_equal(state1, state2):
#     taxis1 = state1[0]
#     taxis2 = state2[0]
#     status1 = state1[4]
#     status2 = state2[4]
#
#     for t1, t2, s1, s2 in zip(taxis1, taxis2, status1, status2):
#         if t1 == t2:
#             # status changed from on_taxi -> has_arrived
#             if s1 != s2 and s2 == 1:
#
#                 return False
#             # unexecuted pickup or dropoff
#             if status1 == status2 or s2 == 2:
#                 return True
#
#     return False

""" fix path of a single taxi to a specific (i) passenger (correct dropoffx action"""
def fix_path_dropoff(path,index):
    num_pass = str_to_int(path[-1]) + index
    st = "(" + str(num_pass) + ",)"
    path[-1] = st
    return path



class Multi_Taxi_Task_Wrapper(Wrapper):

    def __init__(self, env : EnvWrappper):
        super().__init__(env)
        # 1 for each pass, 1 for defualt/stndby
        if env.env.num_taxis==1:
            self.single=True
            self.action_space = Discrete(env.env.num_passengers + 1)
        else:
            self.single = False
            self.action_space = MultiDiscrete([env.env.num_passengers + 1] * self.env.env.num_taxis)
        self.observation_space = env.env.observation_space
        self.agents_plans = { str_to_int(agent) : [0,deque()] for agent in self.env_agents}
        self.index_action_dictionary = { 0 : 'stbdy', 1:'handling pass 1', 2:'handling pass 2', 3:'handling pass 3', 4:'handling pass 4', 5:'handling pass 5' }

    def plan_all_way(self,taxi_pos,pass_loc,pass_dst, pass_indx):
        taxi_P = MultiTaxiEnv(num_taxis=1, num_passengers=1, domain_map=MAP)
        state = [[taxi_pos], taxi_P.state[1],[pass_loc], [pass_dst], [2]]
        map_p = MapProblem(taxi_P, list_to_tuple(state))
        mah = MultiAgentsHeuristic(single_agent_heuristic=manhattan_heuristic, aggr_func=closest_passenger)
        [_, _, path, explored_count, _] = a_star(problem=map_p,heuristic_func=mah)
        path[-1] = "(" + str((4 + pass_indx)) +",)"
        return path

    def plan_drop_only(self, taxi_pos, pass_dst, pass_indx):
        taxi_P = MultiTaxiEnv(num_taxis=1, num_passengers=1, domain_map=MAP)
        state = [[taxi_pos], taxi_P.state[1], [taxi_pos], [pass_dst], [3]]
        map_p = MapProblem(taxi_P, list_to_tuple(state))
        mah = MultiAgentsHeuristic(single_agent_heuristic=manhattan_heuristic, aggr_func=closest_passenger)
        [_, _, path, explored_count, _] = a_star(problem=map_p, heuristic_func=mah)
        path[-1] = "(" + str((4 + pass_indx)) + ",)"
        return path

    def reset(self):
        # run `reset` as usual.
        # returned value is a dictionary of observations with a single entry
        obs = self.env.env.reset()
        self.agents_plans = {i : [0,deque()] for i in range(len(self.env_agents))}
        if self.single: obs= obs['taxi_0']
        return obs

    def get_single_taxi_pass(self,taxi_index,pass_index):
        taxi_loc = self.env.env.state[0][taxi_index]
        pass_loc = self.env.env.state[2][pass_index - 1]
        pass_dest = self.env.env.state[3][pass_index - 1]
        pass_stt = self.env.env.state[4][pass_index - 1]
        return taxi_loc,pass_loc,pass_dest, pass_stt

    def JointAction_list_to_dict(self, lst):
        ret = {}
        for agent in self.env_agents:
            ret[agent] = lst[str_to_int(agent)]
        return ret

    def JointAction_dict_to_list(self, dct):
        j_act = [0] * len(dct)
        for x in dct:
            j_act[str_to_int(x)] = dct[x]
        return j_act

    def step(self,  joint_action):
        if self.single: joint_action=[joint_action]
        atom_joint_action = []
        changed = False
        if isinstance(joint_action,dict):
            joint_action = self.JointAction_dict_to_list(joint_action)
        for index, act in enumerate(joint_action):
                if act == 0 :
                    if self.agents_plans[index][0]!=0:
                        changed = True
                    # case - stndby act
                    a_act = self.env.env.num_actions-1
                elif act == self.agents_plans[index][0]:
                    # case heading to same pass as start
                    try:
                        a_act = self.agents_plans[index][1].popleft()
                        while isinstance(a_act,list):
                            a_act = a_act[0]
                    except:
                        changed = True
                        taxi_loc, pass_loc, pass_dest, pass_stt = self.get_single_taxi_pass(index, act)
                        if pass_stt == 2:
                            plan = self.plan_all_way(taxi_loc, pass_loc, pass_dest, act)  # act - pass#
                            a_plan = deque()
                            self.agents_plans[index][1] = a_plan
                            for p in plan:
                                a_plan.append(p)
                            a_act = str_to_int(a_plan.popleft())
                            while isinstance(a_act, list):
                                a_act = a_act[0]
                        elif pass_stt == index + 3:
                            plan = self.plan_drop_only(taxi_loc, pass_dest, act)  # act - pass#
                            a_plan = deque()
                            self.agents_plans[index][1] = a_plan
                            for p in plan:
                                a_plan.append(p)
                            a_act = str_to_int(a_plan.popleft())
                        else:
                            a_act = self.env.env.num_actions - 1  # stby action
                else:
                    changed = True
                    taxi_loc, pass_loc, pass_dest, pass_stt = self.get_single_taxi_pass(index, act)
                    if pass_stt == 2:
                        plan = self.plan_all_way(taxi_loc, pass_loc, pass_dest, act)   #act - pass#
                        a_plan = deque()
                        self.agents_plans[index][1] = a_plan
                        for p in plan:
                            a_plan.append(p)
                        a_act = str_to_int(a_plan.popleft())
                        while isinstance(a_act,list):
                            a_act = a_act[0]
                    elif pass_stt == index+3:
                        plan = self.plan_drop_only(taxi_loc, pass_dest, act)   #act - pass#
                        a_plan = deque()
                        self.agents_plans[index][1] = a_plan
                        for p in plan:
                            a_plan.append(p)
                        a_act = str_to_int(a_plan.popleft())
                    else:
                        a_act  = self.env.env.num_actions-1          # stby action
                if isinstance(a_act,str):
                    a_act = str_to_int(a_act)
                self.agents_plans[index][0] = a_act
                atom_joint_action.append(a_act)
        last_status= deepcopy(self.env.env.state[4])
        ret = self.env.step(self.JointAction_list_to_dict(atom_joint_action))
        # calculate apdated rewards
        rewards = {}
        cur_status= self.env.env.state[4]
        for key, value in self.JointAction_list_to_dict(joint_action).items():
            if value==0 :
                rewards[key] = 0
            elif (cur_status[value-1] == 3 + str_to_int(key)) and last_status[value-1]==2:
                rewards[key] = 50
            elif (cur_status[value-1] == 1) and (last_status[value-1]== 3 + str_to_int(key)):
                rewards[key] = 100
            else: rewards[key] = -1
        if self.single:
            rewards = sum(r for r in rewards.values())
            return (ret[0]['taxi_0'],rewards,ret[2]['taxi_0'],ret[3])
        return (ret[0],rewards,ret[2],ret[3])







if __name__ == '__main__':
    """
    Builds Multi_taxi env
    """
    env = MultiTaxiEnv(num_taxis=2, num_passengers=2, domain_map=MAP, observation_type='symbolic')

    # env = SingleTaxiWrapper(env)
    obs = env.reset()

    # env.render()

    # # Make sure it works with our API:
    env.agents = env.taxis_names
    # print(f"{env.agents}\n")
    env.action_spaces = {
        agent_name: env.action_space for agent_name in env.agents
    }
    env.observation_spaces = {
        agent_name: env.observation_space for agent_name in env.agents
    }
    env.possible_agents = [agent for agent in env.agents]
    #
    # # env = SingleTaxiWrapper(env)
    # # env = SinglePassengerPosWrapper(environment, taxi_pos=[0, 0])
    environment = Multi_Taxi_Task_Wrapper(EnvWrappper(env, env.agents))
    #
    print('EnvironmentWrapper created')

    a = environment.action_space.sample()
    environment.step(a)
    # path = environment.plan_drop_only([0,0],[2,3],4)
    # print(f"{path}")

