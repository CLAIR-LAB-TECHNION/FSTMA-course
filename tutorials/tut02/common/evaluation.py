import sys

from tqdm.auto import tqdm


def evaluate_policy(env, policy, num_episodes=100, max_actions_per_episode=25, seed=None):
    """
    evaluates a given policy on an initialized taxi environment.
    :param env: a gym taxi environment (v3)
    :param policy: a function that, given a taxi environment state, returns a valid action.
    :param num_episodes: The number of episodes to run during evaluation
    :param max_actions_per_episode: The number of time steps before the episode is ended environment is reset.
    :param seed: a random seed for the environment to enable reproducible results.
    :return: a tuple (total_reward, mean_reward), where `total_reward` is the sum of all rewards achieved in all
             episodes and `mean_reward` is the mean reward per episode.
    """
    # set random seed if given
    if seed is not None:
        env.seed(seed)

    # iterate episodes and accumulate rewards
    all_episode_rewards = 0
    for _ in tqdm(range(num_episodes)):

        # reset env and get initial observation
        obs = env.reset()

        # iterate time steps and accumulate episode rewards
        total_rewards = 0
        for _ in range(max_actions_per_episode):

            # get policy action
            action = policy(obs)

            # perform policy step and accumulate rewards
            obs, reward, done, _ = env.step(action)
            total_rewards += reward

            if done:
                # if task completed, end episode early.
                break

        # accumulate rewards for all episodes
        all_episode_rewards += total_rewards

    # flush excess tqdm output
    sys.stderr.flush()

    return all_episode_rewards, all_episode_rewards / num_episodes
