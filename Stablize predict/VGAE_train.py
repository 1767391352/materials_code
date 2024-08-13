import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import VGAE, GATConv, GCNConv, BatchNorm, LayerNorm
from torch_geometric.utils import batched_negative_sampling
from tqdm import tqdm

from utils import CustomDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = torch.load("D:\\桌面\\materials code\\cif2graph_data\\dataset\\QM9_train_dataset.pth")
torch.manual_seed(3509)
# sampled_dataset = [dataset[i] for i in range(len(dataset)) if i % 15 == 0]
train_dataset = CustomDataset(dataset)
train_loader = DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)


# 定义使用GATConv的编码器
class Encoder(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels):
        super(Encoder, self).__init__()
        self.gcn = GCNConv(num_features, hidden_channels)
        self.mu = GCNConv(hidden_channels, out_channels)
        self.logvar = GCNConv(hidden_channels, out_channels)
        self.bn = LayerNorm(hidden_channels)

    def forward(self, x, edge_index):
        x = torch.relu(self.gcn(x, edge_index))
        # print(x.size())
        x = self.bn(x)
        return self.mu(x, edge_index), self.logvar(x, edge_index)


class Decoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_features, sigmoid=True):
        super(Decoder, self).__init__()
        self.gcn = GATConv(in_channels, hidden_channels, heads=4)
        self.conv = GATConv(hidden_channels * 4, num_features // 4, heads=4)

    def forward(self, z, edge_index, edge_attr, sigmoid: bool = True):
        z = torch.relu(self.gcn(z, edge_index, edge_attr))
        return torch.sigmoid(self.conv(z, edge_index, edge_attr))


# 创建 VGAE 模型


# 设置模型和优化器
out_channels = 32
num_features = train_dataset.num_features
encoder = Encoder(num_features, 64, out_channels)

model = VGAE(encoder=encoder)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 10000
# 初始化一些用于绘图的变量

fig, ax = plt.subplots()  # 创建一个图形和一个坐标轴
losses_whole = []  # 用于存储每个epoch的损失
losses_period = []
# 训练模型
if __name__ == '__main__':
    model = model.to(device)
    try:
        model.load_state_dict(torch.load("D:\\桌面\\materials code\\cif2graph_data\\model\\VGAE_Is_Stable_GNN.pt"))
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

            # out = model(data)  # 前向传播
            # # print(out)
            # # print(data.y)
            #
            # loss = criterion(out, data.y)  # 计算损失
            # 累积损失值
            z = model.encode(data.x, data.edge_index)

            # print(model.encode(data.x, data.edge_index, data.edge_attr))

            # z = model.reparametrize(mu, logstd)
            # loss = 0
            # print(data)
            # print(data.batch)
            # print(negative_sampling(data.edge_index))
            # print(batched_negative_sampling(data.edge_index, data.batch))
            loss = model.kl_loss()
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss.backward()  # 反向传播
            # optimizer.step()

        if epoch % 100 == 0 and epoch != 0:
            torch.save(model.state_dict(), "D:\\桌面\\materials code\\cif2graph_data\\model\\VGAE_Is_Stable_GNN.pt")
            # print("saved")
        average_loss = epoch_loss / len(train_loader)

        losses_whole.append(average_loss)
        losses_period.append(average_loss)
        # 绘制损失曲线图

        if epoch % 20 == 0 and epoch != 0:
            if epoch == 20:
                losses_whole.clear()
            print("epoch_loss:    ", epoch_loss)
            print("epoch:     :", epoch)
            # plt.draw()

            ax.clear()  # 清除之前的图形
            ax.plot(losses_whole)  # 绘制损失曲线
            ax.set_xlabel('Epoch')  # 设置x轴标签
            ax.set_ylabel('Loss')  # 设置y轴标签
            ax.set_title('Loss with Epoch (whole time)')  # 设置图表标题
            plt.savefig("D:\\桌面\\materials code\\cif2graph_data\\result\\VGAE_Is_Stable_GNN_whole.png")

            ax.clear()  # 清除之前的图形
            ax.plot(losses_period)  # 绘制损失曲线
            ax.set_xlabel('Epoch')  # 设置x轴标签
            ax.set_ylabel('Loss')  # 设置y轴标签
            ax.set_title('Loss with Epoch (period)')  # 设置图表标题
            plt.savefig("D:\\桌面\\materials code\\cif2graph_data\\result\\VGAE_Is_Stable_GNN_period.png")
            losses_period.clear()
