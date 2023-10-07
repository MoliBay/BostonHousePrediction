import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
'''
打包数据集
'''

class MyDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return torch.tensor(data=self.X[idx], dtype=torch.float32), \
               torch.tensor(data=self.y[idx], dtype=torch.float32)

    def __len__(self):
        return len(self.X)