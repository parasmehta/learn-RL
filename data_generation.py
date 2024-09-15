import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# one round sequence_create
def sequence_create(env, nr_actions, nr_steps=2):
    step_i = env.reset() # Take this line
    save_rows = [] # Take this line

    for i in range(nr_steps):
        if terminated or truncated:
             break
        else:
            save_row = [i, step_i[0]] # Take this
            chosen_action = np.random.randint(nr_actions)
            step_i, _, terminated, truncated, _ = env.step(action=chosen_action)
            save_row.append(step_i[0]) # Take this
            save_row.append(step_i[1]) # Take this
            save_rows.append(save_row) # Take this  
    return save_rows

# run episodes
def run_episodes(nr_episodes=1, nr_steps=2):
    # Create an environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    nr_actions = env.action_space.n
    for _ in tqdm(range(nr_episodes)):
        env.reset()
        saved_data= sequence_create(env, nr_actions, nr_steps)
    return saved_data

