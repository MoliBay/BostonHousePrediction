import torch
from torch import nn
'''
全连接网络
'''


class Model(nn.Module):

    def __init__(self, n_features=13):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(in_features=n_features, out_features=8)
        self.linear2 = nn.Linear(in_features=8, out_features=1)
        print("model __init__:")

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.squeeze(x)
        # print(f'x.shape:{x.shape}')
        return x
