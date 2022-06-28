'''
map_planner - Taxi domain planner - using Astar - make short term plan of picking and dropping best passenger
'''

# not needed with provided conda environment
# !pip install git+https://github.com/sarah-keren/multi_taxi
# !pip install git+https://github.com/sarah-keren/AI_agents

import numpy

from src.Communication.COM_net import COM_net
from src.agents.agent import DecisionMaker, Action_message_agent
from src.control.Controller_COM import DecentralizedComController
from src.environments.env_wrapper import EnvWrappper

"""## Environment
We'll test our MultiAgentPlanning algorithms over the taxi environment we've already seen with the use of best_first_search as the search backbone for A* and BFS.

"""

import time
import numpy as np

from IPython.display import clear_output
from multi_taxi import MultiTaxiEnv
from ai_dm.Search.best_first_search import breadth_first_search, a_star
from src.decision_makers.planners.MAP import MapProblem
from copy import deepcopy
from colorama import Fore, Style

MAP2 = [
    "+-------+",
    "| : |F: |",
    "| : | : |",
    "| : : : |",
    "| | :G| |",
    "+-------+",
]

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


def manhatten_dist1(r1, c1, r2, c2):
    # calssic manhatten dist |row1 - row2| + |col1 - col2|
    return abs(r1 - r2) + abs(c1 - c2)


def manhatten_heuristic1(node,env):
    # decode state integer to interpretable values
    taxi_row, taxi_col, passenger_idx, dest_idx = env.decode(node.state.get_key())
    locs = env.unwrapped.locs

    # split to 2 cases where the passenger is in the taxi and not in the taxi.
    if passenger_idx == 2:
        # dist from the taxi to the destination
        return manhatten_dist1(taxi_row, taxi_col, *locs[dest_idx]) + 1  # include dropoff
    elif passenger_idx == dest_idx:
        # passenger has reached the destination. this is a goal state
        return 0
    else:
        # dist from the taxi to the passenger and from the passenger to the destination
        passenger_dist = manhatten_dist1(taxi_row, taxi_col, *locs[passenger_idx])
        dest_dist = manhatten_dist1(*locs[passenger_idx], *locs[dest_idx])
        return passenger_dist + dest_dist + 2  # include pickup and dropoff actions

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

"""same state?"""
def is_state_equal(state1, state2):
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

""" fix path of a single taxi to a specific (i) passenger (correct dropoffx action"""
def fix_path_dropoff(path,index):
    num_pass = str_to_int(path[-1]) + index
    st = "(" + str(num_pass) + ",)"
    path[-1] = st
    return path




class Astar_message_highlevel_DM(DecisionMaker):
    """
    Class AstarDM - Decision maker.
    input:
        env - MultiTaxi full env
        single_plan - is single plan needed or not (joint plan)
        Taxi_index (int) - taxi index in multi taxi domain (Taxi_0 -> 0 etc.)
        domain_map - important - make sure it is equal to env map

    output:
        - create set of plans from the taxi (i) to each passenger
        - defualt - chooses the shortest plan as active plan
        - at each get_action() - pops the next action from the actice plan
        - when dropoff (out of active_plan) -> replan for the next available passengers

    """
    def __init__(self,env : MultiTaxiEnv, single_plan = True, Taxi_index=0, domain_map=MAP2, defualt_action = "last_action"):
        if (defualt_action == "last_action"):
            self.defualt_action = env.action_space.n-1
        else: self.defualt_action = 0
        self.last_message = None
        self.changed = False
        self.space = env.action_space
        self.init_state = deepcopy(env.state)
        self.single_plan = single_plan
        self.Taxi_index = Taxi_index
        self.plans = []
        self.map = domain_map
        self.active_plan = []
        self.activeted = False
        self.no_more_plans = False
        self.taking_passenger = None


    def get_action(self, observation):
        if self.no_more_plans:
            return 0 #self.defualt_action
        self.init_state = obs_to_state(observation)
        if self.single_plan:
            temp = [x for x in self.init_state[4]]
            self.init_state = [self.init_state[0], self.init_state[1],self.init_state[2], self.init_state[3], temp]
        if (len(self.active_plan) == 0):
            try:
                self.plans = []
                self.replan()
                self.activeted = True
            except:
                print(f"agent_{self.Taxi_index}: no plan found")
        if (self.updateplan_message(self.last_message)):
            try:
                print(Text_color[self.Taxi_index+1] + "changing plan acording to message" + Fore.BLACK)
                self.active_plan = []
                self.plans = []
                self.replan()
                self.activeted = True
            except:
                print(f"agent_{self.Taxi_index}: no plan found")
        # check active plan for replan
        # if len(self.active_plan) == 0:
        #     self.updateplan_message(self.last_message)
        #     self.replan(self.state)
        # else:
        #     if not self.activeted:
        #         if is_state_equal(list_to_tuple(state), self.map_problem.get_state()): self.activeted = True
        #         else: self.replan(state)
        if self.no_more_plans:
            print(f"agent_{self.Taxi_index}: no action available - using defualt")
            self.taking_passenger = 0
            return 0 #self.defualt_action
        action = str_to_int(self.active_plan.pop(0))
        return self.taking_passenger             #action

    def save_last_message(self, message):
        self.changed = False
        if self.last_message != None:
            for i in range(len(message)):
                if (message[i].data[0] != self.last_message[i].data[0]):
                    self.changed = True
                    continue
        else:
            self.changed = True
        self.last_message = message


    """
    makes the short-term plans (for each available passenger) and activate the shortest (defualt) path for a passenger
    choosen passenger is helt at self.taking_passenger
    """
    def get_short_term_plans(self, state, h, print_simulation=False):
        states_pass = []
        init_states = []
        finished_pass = []
        for i in range(len(state[2])):
            states_pass.append( [state[2][i],state[3][i],state[4][i]])
            if state[4][i]==1: finished_pass.append(i)
            init_states.append([state[0], state[1], [state[2][i]], [state[3][i]], [state[4][i]]])

        taxi_P = MultiTaxiEnv(num_taxis=1, num_passengers=1, domain_map=self.map)

        if len(init_states) == 0:
            # print(f"agent_{self.Taxi_index}: no plan / no more available passengers")
            self.no_more_plans = True
        for i in range(len(init_states)):
            if (i in finished_pass): continue
            map_p = MapProblem(taxi_P, list_to_tuple(init_states[i]))
            # map_p.set_state(init_states[i])

            [_, _, path, explored_count, _] = a_star(problem=map_p, heuristic_func=h)
            if print_simulation:
                run_plan(state,path,map_p)
                print('explored_count:', explored_count)
                print('path length:', len(path))

            #fix all dropoff action to the right passenger
            if len(path)>0:
                path = fix_path_dropoff(path,i)
                self.plans.append(path)
            else:
                print(f"agent_{self.Taxi_index}: no available plan found for passenger {i}")

        if len(self.plans)==0:
            print (f"agent_{self.Taxi_index}: no available plan was found (for all)")
            self.no_more_plans = True
            return
        # set shortest plan as active plan
        self.active_plan = min(self.plans,key=len )
        self.taking_passenger = plan_2_pass(self.active_plan)[0]
        print(Text_color[self.Taxi_index+1] + f"{Pass_color_dict[self.Taxi_index+1]} Taxi_{self.Taxi_index} (re)planning results:")
        print("  # taking " + Text_color[self.taking_passenger] + f"{Pass_color_dict[self.taking_passenger]} passenger"
              + Fore.BLACK + f", len. of plan: {len(self.active_plan)} -> active:{self.active_plan}")
        print(f"  # all plans:{[plan_2_color_pass(p) for p in self.plans]}" + Fore.BLACK)

    """
    train_plans - u can implement any other planning method - and use it instead of get_short_term_plans()
    """
    def train_plans(self):
# A* code with a simple reliable heuristic like the distance between the closest (taxi, passenger) pair
        s = deepcopy(self.init_state)
        mah = MultiAgentsHeuristic(single_agent_heuristic=manhattan_heuristic,aggr_func=closest_passenger)
        # self.plans = joint_simulation(s,mah,print_simulation=True)
        self.get_short_term_plans(s,mah,print_simulation=False)

    def replan(self):
        # replan after current plan has ended
        self.train_plans()
        self.activeted = True

    def updateplan_message(self,message):
        # fix plan according to other Texis messages
        my_plan_size = len(self.active_plan)
        if my_plan_size==0:
            print("no plan for me , no need to check plan w others")
            return False
        for m in message:
            d = m.data
            if d[0]==self.taking_passenger:
                if my_plan_size>d[1]:
                    self.init_state[4] = list(self.init_state[4])
                    self.init_state[4][d[0]-1]=1
                    return True
        return False






if __name__ == '__main__':
    env = MultiTaxiEnv(num_taxis=2, num_passengers=2, domain_map=MAP2, observation_type='symbolic')

    # env = SingleTaxiWrapper(env)
    obs = env.reset()

    # env.render()

    # a = planner.get_action(obs)
    # print(f"{a}")
    # b = planner.get_action(obs)


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
    environment = EnvWrappper(env, env.agents)
    #
    print('EnvironmentWrapper created')
    #
    #
    env_agents = environment.get_env_agents()
    decentralized_agents = {agent_name: Action_message_agent(AstarDM(env,single_plan=False),AgentName=agent_name)             # Agent(LearningDecisionMaker(env.action_space))  # can use diffrent DM
                            for agent_name in env_agents}
    #
    # # Here, the action to perform is collected by each agent
    #
    # # ![decenterlized.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA5QAAAIOCAYAAADHvELZAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAhdEVYdENyZWF0aW9uIFRpbWUAMjAyMjowNToxNiAxNTo1MTozNPGOQ4QAALtVSURBVHhe7N0HeBTVwgbgb0khPZteKQkJgYQWeq+CFCmCoiIgiqJeG4rgRb1ey/X3Wqhi16sURcSGgPSeEAIhQAKBgCEQQhpJtiSbsin7z+zOJhtIspslUvR7H+dhz5ydnTNnZn3my5mZlanVah2ukpGRgaioKKlEREREREREdK0W0r9ERERERERETcJASURERERERFZhoCQiIiIiIiKrMFASERERERGRVRgoiYiIiIiIyCoMlERERERERGQVBkoiIiIiIiKyCgMlERERERERWYWBkoiIiIiIiKzCQElERERERERWYaAkIiIiIiIiqzBQEhERERERkVUYKImIiIiIiMgqDJRERERERERkFQZKIiIiIiIisgoDJREREREREVmFgZKIiIiIiIiswkBJREREREREVmGgJCIiIiIiIqvIKrUFOul1jTNns9GhfYBUIiIiIiIiIroWRyiJiIiIiIjIKk0KlLb2XvqJiIiIiIiIiCOUREREREREZBUGSiIiIiIiIrIKAyVdQ6erwLHP5mDK/N+RpbvmmU10m9BmHcRnCx7F4jjuw+ZyYe0reOK9TThbxD4lIiIiEjVroNy/6BBs+x00TH1O41ftrXXSpdNl4dM7DfeB2joEI7j7vZi3JgVFt1ho0sW9ieCIt3G0mdu1/6WONffB2to/gV/LG/r8ShSkn0LSySwUS3P+LEff7IPglw5KJWHbNalYO/9+RLcLg1+HUZj+7n5kVzdvP9ys/rW8/69fcfKXuGf4AsT4340xHaSZAp3mLNa9NB09O3REq85j8ND7B5q/fw+9hZBO7yCxmfv3wMIucHEL0k/Ork9hw1X9Z66+OfgPugs9cj/B6FGvYWdetTSXiIiI6O9LHyjFnwkxnRqaZ07vx6JxcUN3XPwsEL2lebeiez4/iot/HMDGt3oiad40vHGgTKr5a+v9zz24mJ6MizGvNrp/ZDJH3PHfWJzb8ijay2TS3D+fTleF40tm46ljnfD2z9sRs/pxuKx9CM9+nyu949Zmrn8t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fIU3Rjexf0Pvxr+fkeGbycPRuesADHj/qFTROIvX7zgCCz4chGMLRqFDRA+ER8zC6kypTtRAvXH/fdDpCP457g4MmfYhroz/EssfaFr/VuXH4U3hOJW/9DTGuEozTdk6wFXuCrmHC+pe0E9ERET01yOr1Bbozzj1ATJLBUdHZ/j5+SE3NwelpRoEB8qFEzFDWBN/s08kLKP/l4huTeLvWa78x8P4V1IoXl+zCrMjLf+DCzUsZ8vrmPLMz3B79HOseakPvJrwhywiIiKiv6KaQCmqqqrGxUuFcHZ2gaa4CG1ae+kvhzVioCS6feh05cg9vhtp8tEYEMLg0xzyj+1GikcvDGzj0qSrIoiIiIj+quoESlFpqVYIlQVo08oLjo51L9hioCQiIiIiIiKja34ozRgirw6TRERERERERKauCZSNEUcmOTpJREREREREoiYFSiIiIiIiIiIjmVqtrnMPpSgjIwNRUU387TkiIiIiIiL6W+EIJREREREREVmFgZKIiIiIiIiswkBJREREREREVmGgJCIiIiIiIqswUBIREREREZFVGCiJiIiIiIjIKgyUREREREREZBUGSiIiIiIiIrIKAyURERERERFZhYGSiIiIiIiIrMJASURERERERFZhoCQiIiIiIiKrMFASERERERGRVRgoiYiIiIiIyCoMlERERERERGQVBkoiIiIiIiKyCgMlERERERERWYWBkoiIiIiIiKzCQElERERERERWYaAkIiIiIiIiqzBQEhERERERkVUYKImIiIiIiMgqDJRERERERERkFQZKIiIiIiIisgoDJREREREREVmFgZKIiIiIiIiswkB5U1RD8Uk8Dj5/GSXSnKa53uWbh/byfnw6734sipVmEJmR/u1czPnvz0gtkmYQERER0W3ttg+UCR/uQMB7eVLJUlrEfRWHLkM2Q9ZzE2SDjmO/VHNjVKP8fBGKTpagUprTNGaWj03C7+2SUSgV/wyapI8wefBTOOB/P8ZGSTOvU8aXH+GjqYv104p7f8VZrVRxjXJc/ulHrJq11PDeGduQIdXQDXDwZbTu8BqOSsWmCBh8D3rmLMWo4S9gZ55OmktEREREtyuZWq3WZWSYPx2Pimqm1NDMxEA5vrQrshf4SnMsoLyIScPTEPVRLzwZagOZzBZevvZwkKpve2KgnClD37TO8JRmNauyBPyr32Sk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RERERERGYUBJRERERERERmFASUREREREREZhQElERERERERGYUBJRERERERERmFASUREREREREZhQElERERERERGYUBJRERERERERmFASUREREREREZhQElERERERERGYUBJRERERERERmFASUREREREREZhQElERERERERGYUBJRERERERERmFASUREREREREZhQElERERERERGYUBJRERERERERmFASUREREREREZhQElERERERERGYUBJRERERERERmFASUREREREREZhQElERERERERGYUBJRERERERERmFASUREREREREZhQElERERERERGYUBJRERERERERmFASUREREREREZhQElERERERERGYUBJRERERERERmFASUREREREREZhQElERERERERGYUBJRERERERERmFASUREREREREZhQElERERERERGYUBJRERERERERmFASUREREREREZhQElERERERERGYUBJRERERERERmFASUREREREREYA/h/iWuqXGRndcwAAAABJRU5ErkJggg==)
    #
    # # Let's run the code
    #
    # # In[14]:
    com = COM_net()
    controller = DecentralizedComController(environment, decentralized_agents,com)
    controller.run(render=True, max_iteration=30)
