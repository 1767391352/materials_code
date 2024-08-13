import random

import matplotlib.pyplot as plt
import torch
from torch_geometric.datasets import QM9
from torch_geometric.transforms import BaseTransform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = torch.load("D:\\桌面\\materials code\\cif2graph_data\\dataset\\MP_train_dataset_ABX3.pth")


# 初始化一些用于绘图的变量
class RandomAttributeNoise(BaseTransform):
    def __init__(self, std=0.1):
        super(RandomAttributeNoise, self).__init__()
        self.std = std

    def __call__(self, data):
        if data.x is not None:
            noise = torch.randn_like(data.x) * self.std
            data.x += noise
        if data.edge_attr is not None:
            noise = torch.randn_like(data.edge_attr) * self.std
            data.edge_attr += noise
        return data

    def __repr__(self):
        return '{}(std={})'.format(self.__class__.__name__, self.std)


add_noise = RandomAttributeNoise(std=0.1)
fig, ax = plt.subplots()  # 创建一个图形和一个坐标轴
losses_whole = []  # 用于存储每个epoch的损失
losses_period = []
if __name__ == '__main__':
    print(len(dataset))

    train_dataset = []

    #     print(data)
    num = 0
    for i in range(len(dataset)):
        train_dataset.append(dataset[i])
        for _ in range(9):
            train_dataset.append(add_noise(dataset[i]))

    print(len(train_dataset))
    torch.save(train_dataset, "D:\\桌面\\materials code\\cif2graph_data\\dataset\\MP_train_dataset_ABX3_noised9.pth")

    # print(numpy.array(data.y)[0])
