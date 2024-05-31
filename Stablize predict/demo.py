import codecs
import matplotlib.pyplot as plt
import numpy
from tqdm import tqdm
import model
from utils import CustomDataset
from torch import optim, nn
import torch
from torch_geometric.loader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = torch.load("D:\\桌面\\materials code\\cif2graph_data\\dataset\\dataset_whole.pth")

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
    # for data in train_loader:
    #     print(data)
    train_dataset = [dataset[i] for i in range(len(dataset)) if i % 20 == 6 or i % 20 == 14]
    torch.save(train_dataset, "D:\\桌面\\materials code\\cif2graph_data\\dataset\\train_dataset_whole.pth")
    test_dataset = [dataset[i] for i in range(len(dataset)) if i % 20 != 6 and i % 20 != 14]
    torch.save(test_dataset, "D:\\桌面\\materials code\\cif2graph_data\\dataset\\test_dataset_whole.pth")
        # print(numpy.array(data.y)[0])
