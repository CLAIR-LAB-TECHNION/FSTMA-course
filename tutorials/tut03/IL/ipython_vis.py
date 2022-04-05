import time

from IPython.display import clear_output

DEFAULT_ANIMATION_SLEEP = 0.2
MAX_POLICY_ACTIONS = 25


def animate_policy(env, policy, max_actions=MAX_POLICY_ACTIONS, sleep=DEFAULT_ANIMATION_SLEEP):
    """
    Animates a policy below the jupyter notebook cell in which it was run. Runs infinitely until interrupted by raising
    a KeyboardInterrupt.
    :param env: a gym taxi environment (v3)
    :param policy: a function that, given a taxi environment state, returns a valid action.
    :param max_actions: The number of time steps before the episode is ended environment is reset.
    :param sleep: sleep time between renderings. controls the speed of animation.
    """

    # keep track of completed episodes and collected rewards
    num_episodes_completed = 0
    all_episode_rewards = 0

    try:  # catch intentional interrupts
        while True:  # loop forever until interrupted

            # reset env and get first observation and render the environment
            obs = env.reset()
            env.render()

            # iterate and collect rewards
            total_rewards = 0
            for _ in range(max_actions):  # run until completion or until
                # get policy action
                action = policy(obs)

                # step and count reward
                obs, reward, done, _ = env.step(action)
                total_rewards += reward

                # clear canvas and render new state
                clear_output(wait=True)
                env.render()
                time.sleep(sleep)

                if done:
                    # task completed. end episode
                    break

            # count ended episode and aggregate rewards
            num_episodes_completed += 1
            all_episode_rewards += total_rewards

            # cleanup and wit for next episode
            clear_output(wait=True)
            time.sleep(sleep)

    except KeyboardInterrupt:
        pass
    finally:
        # output final results
        print(f'num episodes completed:   {num_episodes_completed}')
        print(f'total rewards:            {all_episode_rewards}')
        if num_episodes_completed != 0:
            print(f'mean rewards per episode: {all_episode_rewards / num_episodes_completed:.2f}')
