import numpy as np
import time
import torch
import pickle as pickle

def iterate(obj):
    """ Iterate over obj = dict || list. Return [(idx, key)].
    Agents will be indexed by idx, environment objects indexed by key."""
    pairs = []
    for idx, key in enumerate(obj.keys()) if isinstance(obj, dict) \
            else zip(range(len(obj)), range(len(obj))):
        pairs.append((idx, key))
    return pairs


def run_episode_single_agent(env, agent, max_episode_len,display = False):
    """ Runs an episode of a single_agent environment """
    obs = env.reset()
    identifier = agent.identifier
    obs = obs[identifier]
    total_rewards = 0.0  # total rewards is agent rewards
    train_steps = 0

    for _ in range(max_episode_len):
        if display:
            env.render()
            time.sleep(0.15)
        action = agent.action_callback(obs)
        action = {identifier: action}
        new_obs, reward, done, info = env.step(action)

        #extract numerical values from representation
        new_obs = new_obs[identifier]
        reward = reward[identifier]
        done = done[identifier]



        agent.experience_callback(obs, action, new_obs, reward, done)

        total_rewards += reward

        obs = new_obs
        train_steps += 1

        if done:
            break

    return total_rewards, [total_rewards], train_steps


def run_episode_multi_agent(env, agents, max_episode_len, method, display):
    """ Runs an enpisode of a multi agent environment """
    obs = env.reset()
    total_rewards = 0.0
    agent_rewards = [0.0 for _ in range(len(agents))]
    train_steps = 0

    for _ in range(max_episode_len):
        if display:
            env.render()
            time.sleep(0.15)

        actions = [agents[idx].action_callback(obs_i) for (idx, obs_i) in enumerate(obs)]
        # usually will need to do some transformations here. For now assume that environment just takes integers and uses lists.
        new_obs, rewards, done, info = env.step(actions)

        if method == 'train':
            for idx, obs_i in enumerate(obs):
                agents[idx].experience_callback(obs[idx], actions[idx], new_obs[idx], rewards[idx], done[idx])

        for idx, _ in enumerate(obs):
            agent_rewards[idx] += rewards[idx]
            total_rewards += rewards[idx]

        obs = new_obs
        train_steps += 1

        terminal = False
        if type(done) == type(list):
            terminal = all(done)
        elif type(done) == type(dict):
            terminal = done['__all__']

        if terminal:
            break

    return total_rewards, agent_rewards, train_steps


def train(env, is_env_multiagent, agents, max_episode_len, num_episodes, display, save_rate, agents_save_path,
          train_result_path):
    method = 'train'
    run(env, is_env_multiagent, agents, max_episode_len, num_episodes, method, display, save_rate, agents_save_path,
        train_result_path)


def evaluate(env, is_env_multiagent, agents, max_episode_len, num_episodes, display, save_rate, agents_save_path,
             train_result_path):
    method = 'evaluate'
    run(env, is_env_multiagent, agents, max_episode_len, num_episodes, method, display, save_rate, agents_save_path,
        train_result_path)


def run(env, is_env_multiagent, agents, max_episode_len, num_episodes, method, display, save_rate, agents_save_path,
        train_result_path):
    if agents_save_path: import dill  # used to save the agents themselves, pickle bad at serializing objecst
    if train_result_path: import pickle

    episode_rewards = [0.0]
    agent_rewards = [[0.0] for _ in range(len(agents))]
    final_ep_rewards = []
    final_ag_ep_rewards = [[] for _ in range(len(agents))]

    episode_step = 1
    train_steps = 0

    print("Starting iterations...")
    t_time = time.time()

    action_list = []
    for i in range(num_episodes):
        if is_env_multiagent:
            ep_results = run_episode_multi_agent(env, agents, max_episode_len, method, display,action_list)
        else:
            ep_results = run_episode_single_agent(env, agents[0], max_episode_len)


        t_reward, a_rewards, t_steps = ep_results
        train_steps += t_steps

        episode_rewards[-1] += t_reward
        for (idx, a_reward) in enumerate(a_rewards):
            agent_rewards[idx][-1] += a_reward

        for agent in agents:
            agent.episode_callback()

        if len(episode_rewards) % save_rate == 0:
            final_ep_rewards.append(np.mean(episode_rewards[-save_rate:]))
            for i, rew in enumerate(agent_rewards):
                final_ag_ep_rewards[i].append(np.mean(rew[-save_rate:]))

            print("steps: {}, episodes: {}, mean episode reward:{}, time:{}".format(
                train_steps, len(episode_rewards), final_ep_rewards[-1], time.time() - t_time
            ))

            if agents_save_path:  # if save path provided, save agents to agents_save_path
                with open(agents_save_path, "wb") as fp:
                    pickle.dump(agents, fp)

            if train_result_path:  # if train_result_path provided, save results to train_result_path
                save_obj = dict()

                save_obj["final_ep_rewards"] = final_ep_rewards
                save_obj["final_ag_ep_rewards"] = final_ag_ep_rewards
                save_obj["all_ep_rewards"] = episode_rewards
                save_obj["all_ag_rewards"] = agent_rewards

                with open(train_result_path, "wb") as fp:
                    pickle.dump(save_obj, fp)

            t_time = time.time()

        episode_rewards.append(0)
        for (idx, a_reward) in enumerate(a_rewards):
            agent_rewards[idx].append(0)

        episode_step += 1

    print("Finished a total of {} episodes.".format(len(episode_rewards)))

    if agents_save_path:
        print("Agent saved to {}.".format(agents_save_path))

    if train_result_path:
        # final_ep_rewards, final_ag_rewards, episode_rewards, agent_rewards
        print("Train results saved to {}.".format(train_result_path))