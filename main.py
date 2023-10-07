"""
    1，搭建一个可用于波士顿房价预测的全连接网络模型（PyTorch）；
"""

import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from model import Model
from myDataset import MyDataset

if __name__ == '__main__':

    # 读取数据集
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    print(f'data.shape:{data.shape}, target.shape:{target.shape}')

    # 划分训练集，测试集
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1)
    print(f'X_train.shape:{X_train.shape}, y_train.shape:{y_train.shape}')
    print(f'X_test.shape:{X_test.shape}, y_test.shape:{y_test.shape}')

    # 打包训练集，打包测试集
    train_dataset = MyDataset(X=X_train, y=y_train)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=32)

    test_dataset = MyDataset(X=X_test, y=y_test)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=32)

    # 设备检测
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 加载模型
    model = Model()
    model.to(device=device)

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 定义损失函数
    loss_fn = nn.MSELoss(reduction='mean')

    loss_train = []
    loss_test = []
    def train(epochs=100):
        model.train()
        for epoch in range(epochs):
            batch_loss = []
            for batch_X, batch_y in train_dataloader:
                batch_X = batch_X.to(device=device)
                batch_y = batch_y.to(device=device)

                y_pred = model(batch_X)

                loss = loss_fn(y_pred, batch_y)
                # print(f'epoch:{epoch + 1} ,loss:{loss}')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.data.numpy())

            print(epoch+1, np.mean(batch_loss))
            loss_train.append(np.mean(batch_loss))
            batch_loss_test = eval()
            loss_test.append(batch_loss_test)

    # 测试集预测
    def eval(dataloader=test_dataloader):
        model.eval()
        batch_loss = []
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device=device)
                batch_y = batch_y.to(device=device)
                y_pred = model(batch_X)
                loss = loss_fn(y_pred, batch_y)
                batch_loss.append(loss)
        print(f'loss_test:{np.array(batch_loss).mean()}')
        return np.array(batch_loss).mean()


    train()
    x = np.arange(100)
    y1 = np.array(loss_train)
    y2 = np.array(loss_test)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show()


