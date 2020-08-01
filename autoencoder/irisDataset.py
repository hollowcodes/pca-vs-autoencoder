
import numpy as np
import torchvision
import torch
import json


class IrisDataset(torch.utils.data.Dataset):
    def __init__(self, path: str=""):
        with open(path, "r") as f:
            self.dataset = json.load(f)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        data, label = sample[0], sample[1]

        return torch.tensor(data), torch.tensor(label)

    def __len__(self):
        return len(self.dataset)