import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from agent import get_agent

# one round sequence_create

def sequence_create(env, nr_actions, nr_steps=2, nr_trainings=5):
    state_i, _= env.reset() 
    print("Beginning")
    print(state_i)

    terminated = False
    truncated = False
    
    save_rows = [] 

    # create agent with nr_trainings=5 as defaulte
    agent = get_agent(nr_trainings)

    for i in range(nr_steps):
        if terminated or truncated:
             break
        else:
            # save_row = [i, s, a, s']
            # Save initial state s
            save_row = [i, state_i] 

            ## HERE GOES THE ACTION CHOSEN BY THE AGENT
            #chosen_action = np.random.randint(nr_actions)
            chosen_action = agent.compute_single_action(observation=state_i, explore=False)

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
def run_episodes(nr_episodes=1, nr_steps=2, nr_trainings=5):
    # Create an environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    nr_actions = env.action_space.n
    for _ in tqdm(range(nr_episodes)):
        env.reset()
        saved_data= sequence_create(env, nr_actions, nr_steps)
    return saved_data

#table = run_episodes(nr_episodes=1, nr_steps=3)

table = []
for nr_t in (1,3, 5):
    table.append(run_episodes(nr_episodes=1, nr_steps=3, nr_trainings=nr_t))

print(table)