import codecs
import matplotlib.pyplot as plt
from tqdm import tqdm
import model
from utils import CustomDataset
from torch import optim, nn
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = torch.load("D:\\桌面\\materials code\\cif2graph_data\\dataset\\train_dataset_whole.pth")

# sampled_dataset = [dataset[i] for i in range(len(dataset)) if i % 15 == 0]
train_dataset = CustomDataset(dataset)

print("train_dataset.len():       ", train_dataset.__len__())

model = model.Is_Stable_GNN(train_dataset.num_features, 2)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# train_dataset, test_dataset = train_test_split_edges(dataset.dataset, test_ratio=0.2)


# print(train_loader.__dict__)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
class_weights = torch.tensor([33730, 119488])  # 初始权重

# 归一化权重
normalized_weights = class_weights / class_weights.sum()

# 将权重转换为所需的设备（例如CPU或GPU）
weights_tensor = normalized_weights.to(device)
print(weights_tensor)
criterion = nn.CrossEntropyLoss(weights_tensor)

# sampler = WeightedRandomSampler(weights=weights_tensor, num_samples=4000, replacement=True)
# train_loader = DataLoader(dataset=train_dataset, batch_size=4000, sampler=sampler)
train_loader = DataLoader(dataset=train_dataset, batch_size=3000, shuffle=True)

num_epochs = 10000
# 初始化一些用于绘图的变量

fig, ax = plt.subplots()  # 创建一个图形和一个坐标轴
losses_whole = []  # 用于存储每个epoch的损失
losses_period = []
if __name__ == '__main__':
    # for data in train_loader:
    #     print(data)
    model = model.to(device)
    try:
        model.load_state_dict(torch.load("D:\\桌面\\materials code\\cif2graph_data\\model\\Is_Stable_GNN.pt"))
        print("load success!")
    except Exception as e:
        print(e)
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Training"):
        epoch_loss = 0

        for data in train_loader:
            # print(data)

            data.x = data.x.float()  # 确保输入数据是 float 类型
            data = data.to(device)

            out = model(data)  # 前向传播
            # print(out)
            # print(data.y)

            loss = criterion(out, data.y)  # 计算损失
            epoch_loss += loss.item()  # 累积损失值
            optimizer.zero_grad()
            loss.backward()  # 反向传播
            optimizer.step()

        if epoch % 100 == 0 and epoch != 0:
            losses_period.clear()
            torch.save(model.state_dict(), "D:\\桌面\\materials code\\cif2graph_data\\model\\Is_Stable_GNN.pt")
            # print("saved")
        average_loss = epoch_loss / len(train_loader)

        losses_whole.append(average_loss)
        losses_period.append(average_loss)
        # 绘制损失曲线图

        if epoch % 20 == 0 and epoch != 0:
            print("epoch_loss:    ", epoch_loss)
            print("epoch:     :", epoch)
            # plt.draw()

            ax.clear()  # 清除之前的图形
            ax.plot(losses_whole)  # 绘制损失曲线
            ax.set_xlabel('Epoch')  # 设置x轴标签
            ax.set_ylabel('Loss')  # 设置y轴标签
            ax.set_title('Loss with Epoch (whole time)')  # 设置图表标题
            plt.savefig("D:\\桌面\\materials code\\cif2graph_data\\result\\Is_Stable_GNN_whole.png")

            ax.clear()  # 清除之前的图形
            ax.plot(losses_period)  # 绘制损失曲线
            ax.set_xlabel('Epoch')  # 设置x轴标签
            ax.set_ylabel('Loss')  # 设置y轴标签
            ax.set_title('Loss with Epoch (period)')  # 设置图表标题
            plt.savefig("D:\\桌面\\materials code\\cif2graph_data\\result\\Is_Stable_GNN_period.png")
