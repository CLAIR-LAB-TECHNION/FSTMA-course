import os
import sys
import torch as T
import numpy as np
import random
import logging
from pathlib import Path
from datetime import datetime
from mac.environments.env_wrapper import EnvWrappperMultiTaxi
from multi_taxi import MultiTaxiEnv
from src.utils_controller import (DecentralizedRlController, CentralizedRlController,
                                  create_decentralized_agent, create_centralized_agent)
from src.utils import plot_multi_taxi_learning_curve
sys.path.append("./src")

# seeding everything for experimental reproducibility
SEED = 42
np.random.seed(SEED)    # numpy
T.manual_seed(SEED)     # torch
random.seed(SEED)       # random

simple_MAP = [
    "+---------+",
    "| : : : : |",
    "| : : : : |",
    "| : : : : |",
    "| : : : : |",
    "| : : : : |",
    "+---------+",
]


if __name__ == '__main__':
    load_agent = True
    train = False
    evaluate = True
    base_dir = './tmp'
    directory_to_load = '2022_06_12__00_57_05'

    # directory of experiment
    now = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    experiment_directory = now if not load_agent else directory_to_load

    # if False run Centralized control, else run Decentralized control
    decentralized_control_simulation = True

    # multi_taxi parameters
    num_taxis = 2
    num_passengers = 2
    can_see_others = False
    domain_map = simple_MAP
    pickup_only = True

    # DQN parameters
    kwargs = {
        'gamma': 0.99,
        'epsilon': 1,
        'lr': 1e-4,
        'mem_size': 100000,
        'batch_size': 8,
        'eps_min': 0.01,
        'eps_dec': 2e-5,
        'replace': 4000,
        'algo': 'dqn',
        'env_name': 'multi_taxi'
    }

    # controller parameters
    max_iteration = 100
    max_episode = 100

    # create unique directory for saving
    kwargs['chkpt_dir'] = base_dir
    kwargs['chkpt_dir'] = os.path.join(kwargs['chkpt_dir'], experiment_directory)

    # set up the experiment directory for saving
    if not load_agent:
        Path(kwargs['chkpt_dir']).mkdir(parents=True, exist_ok=True)

    # set up a logger
    logging.basicConfig(filemode='a')
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(filename=os.path.join(kwargs['chkpt_dir'], 'experiment_log.log'))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    log.addHandler(fh)
    log.info('-------------- Experiment Log ---------------')
    log.info(f"{SEED = }")
    log.info(f"Controller: {'Decentralized' if decentralized_control_simulation else 'Centralized'}")
    log.info("---- Environment parameters ----")
    log.info(f"{num_taxis = }")
    log.info(f"{num_passengers = }")
    log.info(f"{can_see_others = }")
    log.info("domain_map = ")
    for line in domain_map:
        log.info(f"{line}")
    log.info(f"{pickup_only = }")
    log.info("---- RL algorithm parameters ----")
    log.info(f"{kwargs = }")
    log.info("---- Controller parameters ----")
    log.info(f"{load_agent = }")
    log.info(f"Path: {directory_to_load if load_agent else ''}")
    log.info(f"{evaluate = }")
    log.info(f"{train = }")
    log.info(f"{max_iteration = }")
    log.info(f"{max_episode = }")

    # set up the environment
    taxi_env = MultiTaxiEnv(num_taxis=num_taxis, num_passengers=num_passengers, can_see_others=can_see_others,
                            domain_map=domain_map, pickup_only=pickup_only)

    # wrap environment to be MAC compatible
    taxi_env = EnvWrappperMultiTaxi(taxi_env)

    # render
    taxi_env.render()

    # set up and run the controller
    if decentralized_control_simulation:
        # set up a dictionary of RL decision makers, one for each taxi
        dqn_agents = {}
        for agent_id in taxi_env.get_env_agents():
            dqn_agents[agent_id] = create_decentralized_agent('dqn', taxi_env, agent_id, kwargs, load_agent)

        # initialize a decentralized controller
        decentralized_rl_controller = DecentralizedRlController(env=taxi_env, agents=dqn_agents)

        if train:
            # train
            total_rewards, agents_epsilons = decentralized_rl_controller.train(render=False,
                                                                               max_iteration=max_iteration,
                                                                               max_episode=max_episode)

            # plot training process results
            plot_multi_taxi_learning_curve(total_rewards, agents_epsilons,
                                           filename=os.path.join(kwargs['chkpt_dir'], 'learning_curve_' + now))

        if evaluate:
            # evaluate
            eval_total_rewards = decentralized_rl_controller.evaluate(render=True, max_iteration=10, max_episode=3)

            # plot evaluation process results
            plot_multi_taxi_learning_curve(eval_total_rewards,
                                           filename=os.path.join(kwargs['chkpt_dir'], 'evaluation_curve_' + now))

    else:
        # set up a dictionary of a single decision makers, one for all taxi
        central_dqn_agent = {}
        agent_id = 'central_agent'
        central_dqn_agent[agent_id] = create_centralized_agent('dqn', taxi_env, agent_id, kwargs, load_agent)

        # initialize a centralized controller
        centralized_rl_controller = CentralizedRlController(env=taxi_env, central_agent=central_dqn_agent)

        if train:
            # train
            total_reward, agent_epsilon = centralized_rl_controller.train(render=False, max_iteration=max_iteration,
                                                                          max_episode=max_episode)

            # plot training process results
            plot_multi_taxi_learning_curve(total_reward, agent_epsilon,
                                           filename=os.path.join(kwargs['chkpt_dir'], 'learning_curve_' + now))

        if evaluate:
            # evaluate
            eval_total_rewards = centralized_rl_controller.evaluate(render=True, max_iteration=10, max_episode=3)

            # plot evaluation process results
            plot_multi_taxi_learning_curve(eval_total_rewards,
                                           filename=os.path.join(kwargs['chkpt_dir'], 'evaluation_curve_' + now))

    # close logger
    log.info("-------------- Experiment Done --------------")
    log.removeHandler(fh)
