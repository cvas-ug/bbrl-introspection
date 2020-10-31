import os
import json
import torch
import numpy as np
import logging

from torch.utils.data import Dataset

class ActivationsDataset(Dataset):

    def __init__(self, root):
        self.root = root
        self.activations, self.labels = self.load_dataset()

    def load_dataset(self):
        class_dict = {"approach": 0, "grasp": 1, "retract": 2}
        acts = []
        labels = []
        dataset = torch.load(self.root)

        # root is a .pt file holding the tensors collected
        for behaviour_key in dataset.keys():
            # Dict ep_number -> list of timesteps
            behaviour_episodes = dataset[behaviour_key]

            for episode_key in sorted(behaviour_episodes.keys()):
                
                # List of dicts layer_name -> tensor
                timestep_list = behaviour_episodes[episode_key]
                
                for timestep in timestep_list:
                    # Only feature extraction layer activations gathered
                    # will need to change in the future if more layer
                    # activations used.
                    acts_tensor = torch.cat((timestep["fc1"], timestep["fc2"]))
                    acts.append(acts_tensor)
                    labels.append(class_dict[behaviour_key])

        # Create numpy arrays for activation vectors and ground truth behaviour labels
        return torch.stack(acts), np.array(labels)

    def __getitem__(self, idx):
        return self.activations[idx], self.activations[idx], self.labels[idx]

    def __len__(self):
        return len(self.activations)