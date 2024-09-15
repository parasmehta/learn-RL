# OK, I'll try to set up a function that can be used to get an agent. 

# Imports first:  (Maybe this should be moved to some higher structure.)

import gymnasium as gym
from matplotlib import pyplot as plt
#from plot_util import visualize_env
from ray.rllib.algorithms.dqn import DQNConfig


def get_agent(nr_trainings: int):
    """Provide CartPole agent trained via DQn for nr_trainings episodes.
    Create agent: 
        agent = get_agent(nr_trainings: int)
    Call for action a like this: 
        a = agent.compute_single_action(observation=state s, explore=False)

    Args:
        nr_trainings (int): Number of training episodes for the agent.

    Returns:
        agent instance, I guess..?
    """
    
    # 1.1 - Get the default config of DQN:
    config = DQNConfig()

    # 1.4 - Introduce the environment to the agent's config
    config.environment(env="CartPole-v1").framework(framework="tf2", 
                                                    eager_tracing=True
        ).rollouts(num_rollout_workers=4, num_envs_per_worker=2).evaluation(
            evaluation_config={"explore": False},
            evaluation_duration=10,
            evaluation_interval=1,
            evaluation_duration_unit="episodes",
        )
    
    # 1.5 - Build the agent from the config with .build()
    agent = config.build()
    
    # 3 - Run a loop for nr_trainings 
    #nr_trainings = nr_trainings  # pylint: disable=invalid-name
    #mean_rewards = []
    print(f"\nTraining the agent with {nr_trainings} trainings")
    for _ in range(nr_trainings):
        reports = agent.train()
        if _%5 == 0:
            print("training:",_, ", mean reward", reports["episode_reward_mean"])

    return agent


if __name__ == "__main__":
    print("Testing the agent construction: Create with 1 training")

    a = get_agent(1)
    print("agent a: ", a)
    print("done?")
