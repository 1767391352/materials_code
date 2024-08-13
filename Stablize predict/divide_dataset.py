import random

import matplotlib.pyplot as plt
import torch
from torch_geometric.datasets import QM9

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = torch.load("D:\\桌面\\materials code\\cif2graph_data\\dataset\\MP_dataset_whole_Ef.pth")

# train_dataset = CustomDataset(dataset[:8000])

# print("train_dataset.len():       ", train_dataset.__len__())
#
# model = model.Is_Stable_GNN(train_dataset.num_features, 2)


# train_dataset, test_dataset = train_test_split_edges(dataset.dataset, test_ratio=0.2)

# train_loader = DataLoader(dataset=train_dataset, batch_size=4000, shuffle=True)
# print(train_loader.__dict__)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# 初始化一些用于绘图的变量

fig, ax = plt.subplots()  # 创建一个图形和一个坐标轴
losses_whole = []  # 用于存储每个epoch的损失
losses_period = []
if __name__ == '__main__':
    print(len(dataset))
    select_data = {0: 0, 1: 0}
    train_dataset = []
    test_dataset = []
    valid_dataset = []
    # for data in train_loader:
    #     print(data)
    num = 0
    for i in range(len(dataset)):

        if i % 100 <= 10:
            valid_dataset.append(dataset[i])

        elif i % 100 <= 100:
            train_dataset.append(dataset[i])
        # else:
        #     valid_dataset.append(dataset[i])
    print(len(train_dataset), len(valid_dataset))

    torch.save(train_dataset, "D:\\桌面\\materials code\\cif2graph_data\\dataset\\MP_train_dataset_Ef.pth")

    torch.save(valid_dataset, "D:\\桌面\\materials code\\cif2graph_data\\dataset\\MP_valid_dataset_Ef.pth")
    # print(numpy.array(data.y)[0])
