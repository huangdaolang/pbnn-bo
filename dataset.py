from torch.utils.data import Dataset
import torch


class pref_dataset(Dataset):
    def __init__(self, x_duels, pref):
        self.x_duels = x_duels
        self.pref = pref

    def __getitem__(self, index):
        entry = {"x1": self.x_duels[index][0].reshape(-1),
                 "x2": self.x_duels[index][1].reshape(-1),
                 "pref": self.pref[index]}
        return entry

    def __len__(self):
        return len(self.pref)


class utility_dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        entry = {"x": self.x[index],
                 "y": self.y[index]}
        return entry

    def __len__(self):
        return len(self.y)

