# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + slideshow={"slide_type": "skip"}
from multi_taxi import MultiTaxiEnv
import GTPyhop
from copy import deepcopy
from functools import partial
from IPython.display import display, clear_output
import time

# + [markdown] slideshow={"slide_type": "slide"}
# # Heirarchical Planning tutorial

# + [markdown] slideshow={"slide_type": "slide"}
# ### Note on presenting with RISE

# + [markdown] slideshow={"slide_type": "fragment"}
# References
# * Notebook created by Yonatan Medan 2022
# * Heirarchical planning algorithm based on http://www.cs.umd.edu/~nau/papers/nau2021gtpyhop.pdf by Dana Nau et al
# * Using [GTPyhop](https://github.com/dananau/GTPyhop) as an engine for heirarchical planning search


# + [markdown] slideshow={"slide_type": "slide"}
# ### The environmet definition

# + slideshow={"slide_type": "-"}
env_args = dict(num_taxis=1,                    # 1 taxi agents in the environment
                num_passengers=3,             # 3 passengers in the environment
                max_fuel = [30],      # taxi1 has a capacity of 30 fuel units
                taxis_capacity=[3],          # taxi1 has a capacity of 3 passangers
                option_to_stand_by=False,      
                observation_type='symbolic',  # accepting symbolic (vector) observations
                can_see_others=False)

# + [markdown] slideshow={"slide_type": "slide"}
# ## Calculating (roughly) the env state space

# + slideshow={"slide_type": "fragment"}
NUMBER_OF_ROWS = 7
NUMBER_OF_COLS = 12
TAXI_POSITIONS = NUMBER_OF_ROWS*NUMBER_OF_COLS
FUEL_LEVEL = 30
ONE_PASSANGER_POSITION = NUMBER_OF_ROWS*NUMBER_OF_COLS
NUMBER_OF_PASSANGERS = 3
PASSANGER_STATE = 3
ONE_PASSANGER_DESTINATION = NUMBER_OF_ROWS*NUMBER_OF_COLS

# + slideshow={"slide_type": "fragment"}
TAXI_STATE_SPACE = TAXI_POSITIONS*FUEL_LEVEL*\
                   ONE_PASSANGER_POSITION*NUMBER_OF_PASSANGERS*\
                   PASSANGER_STATE*NUMBER_OF_PASSANGERS*\
                   ONE_PASSANGER_DESTINATION*NUMBER_OF_PASSANGERS

# + slideshow={"slide_type": "fragment"}
TAXI_STATE_SPACE

# + [markdown] slideshow={"slide_type": "slide"}
# # 1,440,270,720 posible states (ROUGHLY)

# + [markdown] slideshow={"slide_type": "skip"}
#  As seen above the environment state space is very big and making it hard to solve the environment using regular approches

# + slideshow={"slide_type": "skip"}
env_instance = MultiTaxiEnv(**env_args)         # cannot see other taxis (but also not collision sensitive)


# + [markdown] slideshow={"slide_type": "slide"}
# ## Action space

# + tags=[] slideshow={"slide_type": "-"}
#### actions
SOUTH = "south"
NORTH = "north"
EAST = "east"
WEST = "west"
PICKUP = "pickup"
DROP_OFF_0 = "dropoff0"
DROP_OFF_1 = "dropoff1"
DROP_OFF_2 = "dropoff2"
DROP_OFF = [DROP_OFF_0, DROP_OFF_1, DROP_OFF_2]
# TURN_ENGINE_ON = "turn_engine_on"
# TURN_ENGINE_OFF = "turn_engine_off"
# STANDBY = "standby"
REFUEL = "refuel"


# + [markdown] slideshow={"slide_type": "slide"}
# ### How can we use heirarchy to lower the search complexity?

# + [markdown] slideshow={"slide_type": "skip"}
# At a heigh level a taxi driver can do one of the actions below:
#  * Pickup a passange
#  * Drop Passanger
#  * Fuel The taxi
#

# + [markdown] slideshow={"slide_type": "slide"}
# <img src="img/RL_tutorial_multiTaxi_be_taxi.png" width="900">

# + [markdown] slideshow={"slide_type": "skip"}
# Going one level deeper Picking up a passanger can be broken in to 3 sub methods:
#  * pickup passanger 1
#  * pickup passanger 2
#  * pickup passanger 3

# + [markdown] slideshow={"slide_type": "slide"}
# <img src = "img/RL_tutorial_multiTaxi_pickup_pass.png" width="900">
#

# + [markdown] slideshow={"slide_type": "skip"}
# Picking up a passanger 1 can be broken into 2 task preforemed one after the other:
#  * Drive to passnager 1 location
#  * pickup passanger
#  
# notice how picking up the passanger is a low level action which can be preforemed on the environment.

# + [markdown] slideshow={"slide_type": "slide"}
# <img src="img/RL_tutorial_multiTaxi_pickup_pass_1.png" width="900">

# + [markdown] slideshow={"slide_type": "skip"}
# Drive to method can be broken down in a recoursive way:
# * try to drive east then apply method Drive to from the new location
# * try to drive north then apply method Drive to from the new location
# * try to drive west then apply method Drive to from the new location
# * try to drive south then apply method Drive to from the new location
#
# Notice this traverses the Graph in an DFS way.
#

# + [markdown] slideshow={"slide_type": "slide"}
# <img src="img/RL_tutorial_multiTaxi_drive_to.png" width="900">

# + [markdown] slideshow={"slide_type": "slide"}
# <img src="img/RL_tutorial_multiTaxi_drive_to_starting.png" width="850">

# + [markdown] slideshow={"slide_type": "skip"}
# ### Pseudo Code translated from the paper. 
# Go through the code and notice the following:
#  * If we only use actions and one heirarchy e.g Be a taxi drive methods are the raw actions. this code boils to regular DFS.
#  * The magic happends in refine_task_and_continue
#      * we can look at each method of the task as an high level action
#      * each method is applied to state to get all of its subtasks.
#      * subtasks are prepended instead of the heigher level task.
#      * we apply the recursive seek apply function of the new refined list of tasks.
#      

# + slideshow={"slide_type": "slide"}
def plan(state, tasks):
    return seek_plan(state, tasks, [])


# + slideshow={"slide_type": "slide"}
# This planner tries to finish all tasks and return a plan to do so.
def seek_plan(state, tasks, plan):
    if len(tasks) == 0:
        return plan # Finished all task hurray!!! return the plan
    first_task = tasks[0] 
    the_rest_of_the_tasks = tasks[1:]
    
    if is_action(first_task):
        return apply_action_and_continue(state, first_task, the_rest_of_the_tasks, plan)
    
    if is_high_level_task(first_task):
        return refine_task_and_continue(state, first_task, the_rest_of_the_tasks, plan)


# + slideshow={"slide_type": "slide"}
def apply_action_and_continue(state,action,remaining_tasks, plan):
    if can_do_action(state, action):
        next_state = preform_action(state, action)
        return seek_plan(next_state, remaining_tasks, plan+[action])
    else:
        return FAIL


# + slideshow={"slide_type": "slide"}
def refine_task_and_continue(state, first_task, remaining_tasks, plan):
    # each task can have several methods. we try them one by one in a DFS manner
    for task_method in first_task.get_methods():# get all methods (heigher level actions) that are aplicable for this task.
        sub_tasks = task_methods(state) # get a list of sub task from the heigher level task
        plan = seek_plan(state, sub_tasks + remaining_tasks, plan) # recursivly call seek_plan with the refined tasks
        if plan is not FAIL:# this fails if planner was not able to apply sub_tasks
            return plan
    return FAIL # all methods failed 


# + [markdown] slideshow={"slide_type": "skip"}
# ### Define helper class to parse the passanger state

# + tags=[] slideshow={"slide_type": "skip"}
class PassengerState:
    def __init__(self,location,destination,state):
        self.row= location[0]
        self.col = location[1]
        self.dest_row = destination[0]
        self.dest_col = destination[1]
        self.is_arrived = True if state == 1 else False
        self.is_on_taxi = True if state == 3 else False
        self.is_waiting_for_taxi = True if state == 2 else False
        
    def __str__(self):
        return str(self.__dict__)
    
    def __repr__(self):
        return str(self.__dict__)
    
        
    



# + [markdown] slideshow={"slide_type": "skip"}
# ### Define a helper class which simplified working with the environment in a planning pradigm
# To plan using an environment we need to be able to get the succesor function (transition function) and reset the environment to traverse and backtrace on the search graph.
# this class helps achiving this by exposing the following importent methods:
# * get_state() get the state of the environment
# * set_state(state) updates the environment to a given state.
# * can_do_action_from_state(state, action) check an action can be applied from a certien states
# * etc
#

# + tags=[] jupyter={"source_hidden": true} slideshow={"slide_type": "skip"}
class TaxiProblem:
    def __init__(self,env):
        self. env = env
        self.visited_locations = set()
        
    def get_state(self):
        return deepcopy(self.env.state)

    def set_state(self,state):
        self.env.reset()
        self.env.state = deepcopy(state)
        
    @classmethod
    def is_state_equal(cls, state1, state2):
        return state1 == state2
    
    def get_action(self,action_str):
        return {"taxi_0":self.env.action_index_dictionary[action_str]}
    
    def step(self, action_str):
        return self.env.step(self.get_action(action_str))
        
    
    def can_do_action_from_state(self, state, action_str):
        action = self.get_action(action_str)
        # save current state
        old_state = self.get_state()
        self.set_state(state)
        self.env.step(action)
        new_state = self.get_state()
    
        #go back to previews_state
        self.set_state(old_state)
        if TaxiProblem.is_state_equal(state, new_state):
            return False
        else:
            return True
        
    def apply_action_from_state(self,state, action_str):
        # save current state
        old_state = self.get_state()
        
        self.set_state(state)
        self.step(action_str)
        new_state = self.get_state()
        
        #go back to previews_state s
        self.set_state(old_state)
        return new_state
        
    @classmethod
    def get_taxi_position(cls, state):
        return state[0][0]
    
    @classmethod
    def get_taxi_feul(cls,state):
        return state[1][0]
    
    @classmethod
    def get_passengers_data(cls,state):
        taxi_locations,taxis_fuel, passenger_location,passenger_dest, passenger_state = state
        return [PassengerState(passenger_location[i],passenger_dest[i], passenger_state[i]) for i in range(3)]
    
    def reset_visited_location(self):
        self.visited_locations = set()
        
    def add_location_to_visited_locations(self,coordinate):
        self.visited_locations.add(tuple(coordinate))

    def remove_location_from_visited(self,coordinate):
        self.visited_locations.discard(tuple(coordinate))
        
    def taxi_has_visited(self, coordinate):
        return tuple(coordinate) in self.visited_locations
    
    def get_available_driving_directions(self, state):
        directions = [SOUTH,WEST,EAST,NORTH]
        return [direction for direction in directions if self.can_do_action_from_state(state, direction)]
        
    def get_infinite_fuel_driving_succesors(self, state):
        ## change full to max
        state_copy = deepcopy(state)
        state_copy[1][0] = 30
        return [(self.apply_action_from_state(state_copy,action), action) for action in self.get_available_driving_directions(state_copy)]

# + slideshow={"slide_type": "slide"}
taxiP = TaxiProblem(env_instance) 

# + [markdown] slideshow={"slide_type": "skip"}
# ### A look at the environment
#  * The yellow rectengle represents the taxi location.
#  * P represents the location of a passanger.
#  * D reprsents a the destination of the passanger with the same color.
#  * F represents a Fuel station.
#  * G represents a Gas station. not used in this setting.
#  

# + slideshow={"slide_type": "slide"}
# env_instance.step(taxiP.get_action(REFUEL))
env_instance.render()


# + [markdown] slideshow={"slide_type": "slide"}
# ### Actions

# + [markdown] slideshow={"slide_type": "skip"}
# In this section we define the raw actions which can be applied to the environment<br>
# Each action can be broken in to 2 steps 
# 1. Precondition - checks if the action is applicable in the given state
# 2. Appling the action
#
# We define 4 actions
# 1. drive
# 2. drop_passanger
# 3. pickup
# 4. refuel

# + slideshow={"slide_type": "fragment"}
## defining actions
def drive(state, direction, problem):
    # Pre-condition
    if TaxiProblem.get_taxi_feul(state)>0 and problem.can_do_action_from_state(state,direction):
        # apply action
        new_state = problem.apply_action_from_state(state, direction)
        return new_state


# + slideshow={"slide_type": "fragment"}
## defining actions
def drop_passanger(state, pass_number, problem):
    passanger_data = TaxiProblem.get_passengers_data(state)[pass_number]
    # Pre-condition
    if passanger_data.is_on_taxi and [passanger_data.dest_row, passanger_data.dest_col] == TaxiProblem.get_taxi_position(state):
        new_state = problem.apply_action_from_state(state, DROP_OFF[pass_number])
        return new_state


# + slideshow={"slide_type": "subslide"}
def pickup(state, problem):
    if problem.can_do_action_from_state(state,PICKUP):
        new_state = problem.apply_action_from_state(state, PICKUP)
        return new_state
    
def refuel(state, problem):
    if problem.can_do_action_from_state(state, REFUEL):
        new_state = problem.apply_action_from_state(state, REFUEL)
        return new_state


# + slideshow={"slide_type": "fragment"}
def define_actions():
    GTPyhop.declare_actions(drive, drop_passanger, pickup, refuel)
# -



# + [markdown] slideshow={"slide_type": "slide"}
# ### Methods

# + [markdown] slideshow={"slide_type": "skip"}
# In this section we are defining methods.<br>
#  * A method is a high level action which returns a subset of task to be applied.<br>
#  * A task can have several methods of achiving the task.<br>
#
#
# Here we only define the drive_to task. drive_to has 4 sub methods as explained near the drive_to illustration above

# + slideshow={"slide_type": "fragment"}
def define_methods_v1():
    drive_to_methods = [partial(drive_to_starting_direction, direction=direction) for direction in [SOUTH,WEST,NORTH,EAST]]
    GTPyhop.declare_task_methods("drive_to", *drive_to_methods)


# + slideshow={"slide_type": "fragment"}
def drive_to_starting_direction(state, coordinate, problem, direction):
    #Pre conditions
    if TaxiProblem.get_taxi_position(state) == coordinate:
        return [] # we got to the location no need to do anything
    else:
        return [("drive", direction, problem), ("drive_to", coordinate, problem)]


# + slideshow={"slide_type": "skip"}
GTPyhop.verbose = 0


# + tags=[] slideshow={"slide_type": "skip"}
def run_plan(state,plan,action_dict,problem):
    new_state = deepcopy(state)
    problem.set_state(state)
    problem.env.render()
    for action in plan:
        time.sleep(0.25)
        new_state = action_dict[action[0]](new_state,*action[1:])
        problem.set_state(new_state)
        clear_output(wait = True)
        problem.env.render()
        

    return new_state


# + tags=[] slideshow={"slide_type": "slide"}
def find_and_run_plan(initial_state, methods, problem):
    problem.set_state(initial_state)
    # Find a plan
    plan = GTPyhop.find_plan(initial_state, methods)
    # run the plan in the environment and render the outcome
    final_state = run_plan(initial_state,plan, GTPyhop.current_domain._action_dict, problem)
    clear_output(wait = True)
    taxiP.env.render()
    return final_state, plan


# + slideshow={"slide_type": "slide"}
multie_taxi = GTPyhop.Domain("MultieTaxi")

define_actions()
define_methods_v1()

# + [markdown] slideshow={"slide_type": "slide"}
# ### First try - finding a plan to go to location [5,5]

# + tags=[] slideshow={"slide_type": "subslide"}
taxiP.env.reset()
find_and_run_plan(taxiP.get_state(), [("drive_to", [5,5], taxiP)], taxiP)


# + [markdown] slideshow={"slide_type": "skip"}
# What happend?
#  * Question: Notice how most of the times the taxi reaches the end of its fuel tank, can you think of why?
#  * Answer: Notice how at each taxi location the fuel can have a different value, making two states where the taxi is in the same location but the fuel level is different complitly different for the algorithm. combining with the DFS approch we expand the node of the search graphs by deepening it thus doing the same actions again and again until the fuel runs out.

# + [markdown] slideshow={"slide_type": "slide"}
# ### How to solve out of fuel problem?
#  * using BFS to search the taxi location space only (with constant fuel level)
#  * Notice each node can be defined by taxi location only making the state space very small (7x12)
#  * using BFS we are guaranteed to to find the shortest path
#  * notice how we used another algorithm here to solve the lower level task of driving to a given location - this can be applied to many areas, where lower level tasks can be sloved in a simple way.

# + slideshow={"slide_type": "slide"}
def bfs(state, coordinate, problem):
    coordinate = tuple(coordinate)
    visited_states = set()
    current_location = tuple(TaxiProblem.get_taxi_position(state))
    visited_states.add(current_location)

    queue = [(state, [])]
    while len(queue)>0:
        current_state, plan = queue.pop(0)
        current_location = tuple(TaxiProblem.get_taxi_position(current_state))
        #NOTICE: resets fuel to full to evoid going to the same location twice
        for next_state, action  in problem.get_infinite_fuel_driving_succesors(current_state):
            next_location = TaxiProblem.get_taxi_position(next_state)
            if tuple(coordinate)==tuple(next_location):
                return plan + [action]
            else:
                if tuple(next_location) not in visited_states:
                    visited_states.add(tuple(next_location))
                    queue.append((next_state,plan+[action]))


# + slideshow={"slide_type": "slide"}
def drive_to_shortest_path(state, coordinate, problem):
    shortest_path = [("drive", direction, problem) for direction in bfs(state, coordinate, problem)]
    return shortest_path


# + [markdown] slideshow={"slide_type": "slide"}
# ### The whole picture, defining what it is to be a taxi driver:

# + slideshow={"slide_type": "fragment"}
def drive_to_waiting_passanger(state, problem, passanger_idx):
    passenger = TaxiProblem.get_passengers_data(state)[passanger_idx]
    if passenger.is_waiting_for_taxi:
        return [("drive_to",[passenger.row,passenger.col], problem)]
    


# + slideshow={"slide_type": "subslide"}
def define_drive_to_waiting_passanger():
    drive_to_waiting_passanger_methods = [partial(drive_to_waiting_passanger, passanger_idx = i) for i in range(3)]
    GTPyhop.declare_task_methods("drive_to_waiting_passanger", *drive_to_waiting_passanger_methods)


# + slideshow={"slide_type": "subslide"}
def pickup_passanger(state, problem):
    passenger_data = TaxiProblem.get_passengers_data(state)
    ### optimize by picking the closest passanger
    return [("drive_to_waiting_passanger",problem),("pickup", problem)]


# + slideshow={"slide_type": "subslide"}
def fuel_at_station(state,problem, fuel_station_coordinate):
    return [("drive_to", fuel_station_coordinate,problem),("refuel",problem)]


# + slideshow={"slide_type": "subslide"}

def define_drive_to_fuel_method():
    fuel_statation_locations = [[0,2],[0,10]]
    fuel_station_methods = [partial(fuel_at_station, fuel_station_coordinate=coordinate) for coordinate in fuel_statation_locations]
    GTPyhop.declare_task_methods("fuel_at_station", *fuel_station_methods)


# + slideshow={"slide_type": "subslide"}
def drive_to_fuel_method(state, problem):
    return [("fuel_at_station",problem)]


# + slideshow={"slide_type": "slide"}
def drive_and_drop_passanger_i(state, problem, passanger_idx):
    passenger = TaxiProblem.get_passengers_data(state)[passanger_idx]
    return [("drive_to", [passenger.dest_row,passenger.dest_col], problem), ("drop_passanger",passanger_idx,problem)]


# + slideshow={"slide_type": "slide"}
def define_drop_passanger_method():
    drive_and_drop_passanger_methods = [partial(drive_and_drop_passanger_i, passanger_idx = i) for i in range(3)]
    GTPyhop.declare_task_methods("drive_passanger_and_drop", *drive_and_drop_passanger_methods)


# + slideshow={"slide_type": "skip"}
def drive_passanger_and_drop(state, problem):
    return [("drive_passanger_and_drop", problem)]


# + slideshow={"slide_type": "skip"}
def all_passangers_down(state):
    passenger_data = TaxiProblem.get_passengers_data(state)
    done = True
    for passanger in passenger_data:
        if not passanger.is_arrived:
            done = False
        
    return done


# + slideshow={"slide_type": "slide"}
def define_do_taxi_methods():
    GTPyhop.declare_task_methods(
        "do_taxi_methods",
        pickup_passanger,
        drive_passanger_and_drop,
        drive_to_fuel_method)


# + [markdown] slideshow={"slide_type": "skip"}
# This is the heighst level of heirarchy. 
# * First we check if all passagers are down we are done
# * Else we do a high level taxi method then recursivly return to "Be a taxi driver" 

# + slideshow={"slide_type": "slide"}
def be_taxi_driver(state, problem):
    if all_passangers_down(state):
        return []
    else:
        return [("do_taxi_methods",problem),("be_taxi_driver", problem)]


# + slideshow={"slide_type": "slide"}
def define_methods_v2():
    GTPyhop.declare_task_methods("drive_to", drive_to_shortest_path)
    
    define_drive_to_waiting_passanger()
    GTPyhop.declare_task_methods("pickup_passanger", pickup_passanger)
    define_drive_to_fuel_method()
    define_drop_passanger_method()
    define_do_taxi_methods()
    GTPyhop.declare_task_methods("be_taxi_driver", be_taxi_driver)


# + [markdown] slideshow={"slide_type": "slide"}
# # Lets test our heirarchical planner!

# + slideshow={"slide_type": "slide"}
# Define the domain, actions and methods
GTPyhop.Domain("MultiTaxiv2")
define_actions()
define_methods_v2()

# + slideshow={"slide_type": "slide"}
taxiP.env.reset()
find_and_run_plan(taxiP.get_state(),[("drive_to", [0,0], taxiP)], taxiP)

# + slideshow={"slide_type": "slide"}
taxiP.env.reset()
find_and_run_plan(taxiP.get_state(),[("drive_to_waiting_passanger", taxiP)], taxiP)
# -



# + slideshow={"slide_type": "slide"}
taxiP.env.reset()
initial_state = taxiP.get_state()
find_and_run_plan(taxiP.get_state(),[("fuel_at_station", taxiP)], taxiP)

# + slideshow={"slide_type": "slide"}
find_and_run_plan(taxiP.get_state(),[("pickup_passanger", taxiP),("drive_passanger_and_drop", taxiP)], taxiP)

# + slideshow={"slide_type": "slide"}
taxiP.env.reset()
find_and_run_plan(taxiP.get_state(),[("be_taxi_driver", taxiP)], taxiP)

# + [markdown] slideshow={"slide_type": "slide"}
# ### Questions?
