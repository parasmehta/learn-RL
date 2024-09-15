import numpy as np
import gymnasium as gym
import os
import torch

def save_dataset(datapoints, folder, name):
    '''
    
    Args:
        list_tuples (List(Tuple)): list with (s, a, s') from the simulation

    External:
        saved_dataset pytorch format
    
    Returns:
        None
    '''

    all_dicts = {}
    for i in range(len(datapoints)):
        dpi = datapoints[i]
        s_initial, a, s_final = dpi[0], dpi[1], dpi[2]
        combined_data = combine_data(s_i=s_initial, a=a, s_f=s_final)
        torch_data = convert_to_pytorch(combined_data)
        all_dicts[i] = torch_data 
         
    save_pytorch_data(data = all_dicts, filedir=folder, filename=name)


def combine_data(s_i, a, s_f):
        """
        Returns:
            Dictionary with combined args.

        """

        vec_in = s_i.tolist()
        x_list = []

        for i in range(len(vec_in)):
             x_list.append(vec_in[i])

        x_list.append(float(a))  

        print(x_list) 
        combined = {
            's_initial_a': x_list,
            's_final': s_f
        }
        return combined
    
def convert_to_numpy(data):
    # Convert the combined data to numpy arrays
    return {k: np.asarray(v).astype(np.float32) for k, v in data.items()}

def convert_to_pytorch(data):
    # Convert the combined data to PyTorch tensors
    dict_np = convert_to_numpy(data)
    return {k: torch.from_numpy(v) for k, v in dict_np.items()}

def save_pytorch_data(data, filedir, filename):
    os.makedirs(filedir, exist_ok=True)
    filepath = filedir + '/' + filename + ".pt"

    # Save PyTorch data
    torch.save(data, filepath)

    print(f"- Saved pytorch combined data to path: {filepath} \n")
