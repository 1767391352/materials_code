import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import VGAE, GATConv
from torch_geometric.utils import train_test_split_edges

# 加载数据
data = Planetoid("/tmp/Cora", "Cora")[0]
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)

# 定义使用GATConv的编码器
class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, 2 * out_channels, heads=1)
        self.conv_mu = GATConv(2 * out_channels, out_channels, heads=1)
        self.conv_logstd = GATConv(2 * out_channels, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# 设置模型和优化器
out_channels = 16  # 隐空间的维度
num_features = data.num_features
encoder = GATEncoder(num_features, out_channels)
model = VGAE(encoder)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
def train():
    model.train()
    optimizer.zero_grad()
    z, mu, logstd = model.encode(data.x, data.train_pos_edge_index)
    loss = model.recon_loss(z, data.train_pos_edge_index)
    loss += (1 / data.num_nodes) * model.kl_loss(mu, logstd)
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(200):
    loss = train()
    print(f'Epoch: {epoch+1}, Loss: {loss:.4f}')