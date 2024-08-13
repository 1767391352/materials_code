import random

import matplotlib.pyplot as plt
import ray
import torch
from torch import optim, nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import VGAE
from tqdm import tqdm
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from model import Is_Stable_GNN
from model import VGAE_Encoder
from utils import CustomDataset, mae_metric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(3509)
dataset = torch.load("D:\\桌面\\materials code\\cif2graph_data\\dataset\\QM9_train_dataset.pth")
dataset_v = torch.load("D:\\桌面\\materials code\\cif2graph_data\\dataset\\QM9_valid_dataset.pth")
dataset_t = torch.load("D:\\桌面\\materials code\\cif2graph_data\\dataset\\QM9_test_dataset.pth")
# sampled_dataset = [dataset[i] for i in range(len(dataset)) if i % 15 == 0]
train_dataset = CustomDataset(dataset)
valid_dataset = CustomDataset(dataset_v)
test_dataset = CustomDataset(dataset_t)
# 获取节点的隐层表示
nodes_out_channels = 32
num_features = train_dataset.num_features
nodes_encoder = VGAE_Encoder(num_features, 64, nodes_out_channels)
nodes_encode_model = VGAE(encoder=nodes_encoder)
nodes_encode_model.load_state_dict(torch.load("D:\\桌面\\materials code\\cif2graph_data\\model\\VGAE_Is_Stable_GNN.pt"))
nodes_encode_model = nodes_encode_model.to(device)
# 获取边的隐层表示
# input_dim = train_dataset.num_edge_features
# hidden_dim = 64
# latent_dim = 16
# edges_encode_model = model.VAE(input_dim, hidden_dim, latent_dim)
# edges_encode_model.load_state_dict(torch.load("D:\\桌面\\materials code\\cif2graph_data\\model\\VAE_Is_Stable_GNN.pt"))
# edges_encode_model = edges_encode_model.to(device)


# model = model.Is_Stable_GNN(train_dataset.num_node_features, train_dataset.num_edge_features)


# train_dataset, test_dataset = train_test_split_edges(dataset.dataset, test_ratio=0.2)


# print(train_loader.__dict__)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
# class_weights = torch.tensor([33730, 119488])  # 初始权重
#
# # 归一化权重
# normalized_weights = class_weights / class_weights.sum()
#
# # 将权重转换为所需的设备（例如CPU或GPU）
# weights_tensor = normalized_weights.to(device)
# print(weights_tensor)
criterion = nn.MSELoss()

# print(train_dataset[5])
num_epochs = 10000
# 初始化一些用于绘图的变量

fig, ax = plt.subplots(2, 1, figsize=(8, 12))

losses_whole = []  # 用于存储每个epoch的损失
losses_period = []
valid_losses_whole = []  # 用于存储每个epoch的损失
true_labels = []
predicted_labels = []
r2_whole = []

config = {
    "learning_rate": tune.loguniform(1e-4, 1e-1),  # 学习率的对数均匀分布      # 批量大小的选择
    "hidden_layers": tune.qrandint(16, 64, 32),  # 隐藏层单元数的量化随机整数
    "layer_nums": tune.randint(1, 16)
}
scheduler = ASHAScheduler(
    max_t=10,  # 最大训练迭代次数
    grace_period=1,  # 最小训练迭代次数
    reduction_factor=2  # 减少因子
)
from ray.tune import CLIReporter

reporter = CLIReporter(max_progress_rows=10)
reporter.add_metric_column("loss")


def r2_score_gpu(y_true, y_pred):
    # 计算总平方和
    total_sum_squares = torch.sum((y_true - torch.mean(y_true)) ** 2)

    # 计算残差平方和
    residual_sum_squares = torch.sum((y_true - y_pred) ** 2)

    # 计算R-squared
    r2 = 1 - (residual_sum_squares.item() / total_sum_squares.item())
    return max(r2, 0.0)


def ray_train(config):
    model = Is_Stable_GNN(nodes_out_channels, train_dataset.num_edge_features)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    # -------------------------------train-------------------------------#
    epoch_loss = 0

    model.train()
    # for data in train_loader:
    #     # print(data)
    #
    #     data.x = data.x.float()  # 确保输入数据是 float 类型
    #     data = data.to(device)
    #
    #     data.x = nodes_encode_model.encode(data.x, data.edge_index)  # 节点VAE
    #     #
    #     # x_recon, mu, logvar = edges_encode_model(data.edge_attr)
    #     # data.edge_attr = edges_encode_model.reparameterize(mu, logvar)  # 边VAE
    #     out = model(data)  # 前向传播
    #     # print(data)
    #     # print(out.size())
    #     # print(data.y.size())
    #
    #     loss = criterion(out, data.y)  # 计算损失
    #
    #     epoch_loss += loss.item()  # 累积损失值
    #     optimizer.zero_grad()
    #     loss.backward()  # 反向传播
    #     optimizer.step()

    # average_loss = epoch_loss / len(train_loader)
    average_loss = 0

    # losses_whole.append(average_loss)
    return {"loss": average_loss}


def train(model, epoch):
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    # -------------------------------train-------------------------------#
    model.train()
    epoch_loss = 0

    for data in train_loader:
        # print(data)

        data.x = data.x.float()  # 确保输入数据是 float 类型
        data = data.to(device)

        data.x = nodes_encode_model.encode(data.x, data.edge_index)  # 节点VAE
        #
        # x_recon, mu, logvar = edges_encode_model(data.edge_attr)
        # data.edge_attr = edges_encode_model.reparameterize(mu, logvar)  # 边VAE
        out = model(data)  # 前向传播
        # print(data)
        # print(out)
        # print(data.y)

        loss = criterion(out, data.y)  # 计算损失
        mae = mae_metric(out.data.cpu(), data.y.cpu()).item()
        epoch_loss += mae  # 累积损失值
        # epoch_loss += loss.item()  # 累积损失值
        optimizer.zero_grad()
        loss.backward()  # 反向传播
        optimizer.step()

    if epoch % 100 == 0 and epoch != 0:
        torch.save(model.state_dict(), "D:\\桌面\\materials code\\cif2graph_data\\model\\Is_Stable_GNN.pt")
        # print("saved")
    average_loss = epoch_loss / len(train_loader)

    losses_whole.append(average_loss)
    return average_loss, losses_whole


def valid(model):
    model.eval()
    valid_epoch_loss = 0
    r2s = 0
    for valid_data in valid_loader:
        # print(data)
        with torch.no_grad():
            valid_data.x = valid_data.x.float()  # 确保输入数据是 float 类型
            valid_data = valid_data.to(device)
            valid_data.x = nodes_encode_model.encode(valid_data.x, valid_data.edge_index)  # 节点VAE
            #
            # x_recon, mu, logvar = edges_encode_model(valid_data.edge_attr)
            # valid_data.edge_attr = edges_encode_model.reparameterize(mu, logvar)  # 边VAE
            out = model(valid_data)  # 前向传播
            loss = criterion(out, valid_data.y)  # 计算损失
            r2 = r2_score_gpu(valid_data.y, out)
            r2s += r2
            mae = mae_metric(out.data.cpu(), valid_data.y.cpu()).item()

            valid_epoch_loss += mae  # 累积损失值
        # print(out)
        # print(data.y)
    r2_average = r2s / len(valid_loader)
    r2_whole.append(r2_average)
    valid_average_loss = valid_epoch_loss / len(valid_loader)

    valid_losses_whole.append(valid_average_loss)
    return valid_average_loss, r2_average, r2_whole


def test(model):
    model.eval()
    test_epoch_loss = 0
    r2s = 0
    for test_data in test_loader:
        # print(data)
        with torch.no_grad():
            test_data.x = test_data.x.float()  # 确保输入数据是 float 类型
            test_data = test_data.to(device)
            test_data.x = nodes_encode_model.encode(test_data.x, test_data.edge_index)  # 节点VAE
            #
            # x_recon, mu, logvar = edges_encode_model(valid_data.edge_attr)
            # valid_data.edge_attr = edges_encode_model.reparameterize(mu, logvar)  # 边VAE
            out = model(test_data)  # 前向传播
            loss = criterion(out, test_data.y)  # 计算损失
            r2 = r2_score_gpu(test_data.y, out)
            r2s += r2
            mae = mae_metric(out.data.cpu(), test_data.y.cpu()).item()

            test_epoch_loss += mae  # 累积损失值
        # print(out)
        # print(data.y)
    r2_average = r2s / len(test_loader)
    r2_whole.append(r2_average)
    test_average_loss = test_epoch_loss / len(test_loader)


    return test_average_loss, r2_average, r2_whole


if __name__ == '__main__':
    train_loader = DataLoader(dataset=train_dataset, batch_size=2500, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=2500, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=2500, shuffle=True)
    model = Is_Stable_GNN(nodes_out_channels, train_dataset.num_edge_features, layer_nums=12)
    model = model.to(device)
    try:
        model.load_state_dict(torch.load("D:\\桌面\\materials code\\cif2graph_data\\model\\Is_Stable_GNN12.pt"))
        print("load success!")
    except Exception as e:
        print(e)
    # ----------------------------训练----------------------------------------

    test_average_loss, r2_average, r2_whole = test(model)
    print("loss:  ", test_average_loss, "R^2:  ", r2_average)

    # # ----------------------------训练----------------------------------------
    # # for data in train_loader:
    # #     print(data)
    #
    # try:
    #     model.load_state_dict(torch.load("D:\\桌面\\materials code\\cif2graph_data\\model\\Is_Stable_GNN.pt"))
    #     print("load success!")
    # except Exception as e:
    #     print(e)
    # for epoch in tqdm(range(num_epochs), desc="Training"):
    #     average_loss, losses_whole = train(model, epoch)
    #     valid_average_loss, valid_losses_whole, r2_average, r2_whole = valid(model, epoch)
    #     # 绘制损失曲线图
    #
    #     # -------------------------------valid-------------------------------#
    #
    #     # model.train()
    #     if epoch % 20 == 0 and epoch != 0 and epoch != 20:
    #         print("     epoch_loss:", average_loss, "     valid_epoch_loss:", valid_average_loss,
    #               "     R^2:", r2_average)
    #         # plt.draw()
    #         # if epoch == 20:
    #         #     losses_whole.clear()
    #         #     valid_losses_whole.clear()
    #
    #         # ax[0].clear()  # 清除之前的图形
    #         ax[0].clear()  # 清除之前的图形
    #         ax[1].clear()  # 清除之前的图形
    #
    #         ax[0].plot(losses_whole, color='blue', label='loss')  # 绘制损失曲线
    #         ax[0].plot(valid_losses_whole, color='orange', label='valid_loss')  # 绘制F1曲线
    #         ax[1].plot(r2_whole, color='purple', label='R^2')  # 绘制F1曲线
    #
    #         # ax[0].set_xlabel('Epoch')  # 设置x轴标签
    #         ax[0].set_ylabel('Loss / valid_loss with Epoch')  # 设置y轴标签
    #
    #         # ax[1].set_xlabel('Epoch')  # 设置x轴标签
    #
    #         # ax[2].set_xlabel('Epoch')  # 设置x轴标签
    #         ax[1].set_ylabel('R^2 with Epoch')  # 设置y轴标签
    #
    #         ax[0].legend(loc="upper left")
    #         ax[1].legend(loc="upper left")
    #         # plt.legend(loc="upper right")
    #         plt.tight_layout()
    #
    #         plt.savefig("D:\\桌面\\materials code\\cif2graph_data\\result\\Is_Stable_GNN_whole.png")
