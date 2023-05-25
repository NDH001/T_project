from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np


class ChurnData(Dataset):
    def __init__(self, csv):
        self.label = csv["Churn"].to_numpy()
        self.data = csv.drop(columns=["Churn"]).to_numpy()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]

        data = torch.tensor(data)
        label = torch.tensor(label)
        return data, label
