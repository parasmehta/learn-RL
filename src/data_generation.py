import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from save_dataset import save_dataset

# one round sequence_create

def sequence_create(env, nr_actions, nr_steps=2):
    state_i, _= env.reset() 
    print("Beginning")
    print(state_i)

    terminated = False
    truncated = False
    
    save_rows = [] 
    for i in range(nr_steps):
        if terminated or truncated:
             break
        else:
            # save_row = [s, a, s']
            # Save initial state s
            save_row = [state_i] 

            ## HERE GOES THE ACTION CHOSEN BY THE AGENT
            chosen_action = np.random.randint(nr_actions)
            # Make step
            state_i, _, terminated, truncated, _ = env.step(action=chosen_action)
            
            # Save the action a
            save_row.append(chosen_action)
            # Save new state s'
            save_row.append(state_i) 

            # save_rows = [[save_row_1], [save_row_2] .... nr_steps]
            save_rows.append(save_row) 
    return save_rows

# run episodes
def run_episodes(nr_episodes=100, nr_steps=10):
    # Create an environment

    whole_dataset = []
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    nr_actions = env.action_space.n
    for _ in tqdm(range(nr_episodes)):
        env.reset()
        episode_data = sequence_create(env, nr_actions, nr_steps)
        whole_dataset.extend(episode_data)
    return whole_dataset

table = run_episodes(nr_episodes=100, nr_steps=100)
save_dataset(datapoints=table, folder="data/random_dataset", name = "eps_100_runs_100")