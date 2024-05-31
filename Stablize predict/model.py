import torch
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool, SAGPooling, global_add_pool, TransformerConv, GCNConv, \
    BatchNorm, VGAE, GAE


class Encoder(torch.nn.Module):
    def __init__(self, num_features, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(num_features, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = torch.relu(self.conv1(x, edge_index, edge_attr))
        return self.conv_mu(x, edge_index, edge_attr), self.conv_logstd(x, edge_index, edge_attr)


class Is_Stable_GNN(nn.Module):

    def __init__(self, num_features, num_classes, heads=16):
        super(Is_Stable_GNN, self).__init__()  # "add" 是聚合邻居特征的方式

        self.gat1 = GATConv(num_features, 8, heads=heads,
                            add_self_loops=True, dropout=0.0)
        self.gat2 = GATConv(8 * heads, 8, heads=heads,
                            add_self_loops=True, dropout=0.0)
        self.gat3 = GATConv(8 * heads, 8, heads=heads,
                            add_self_loops=True, dropout=0.0)

        # self.gcn1 = GCNConv(num_features, 16,
        #                     add_self_loops=True)
        # self.transformer = TransformerConv(num_features, 16, heads=heads,
        #                                    dropout=0.0, edge_dim=23)
        self.silu = torch.nn.SiLU()
        self.relu = torch.nn.ReLU()
        self.bn1 = BatchNorm(8 * heads)
        self.bn2 = BatchNorm(8 * heads)
        self.bn3 = BatchNorm(8 * heads)
        self.bn4 = BatchNorm(num_classes)
        # self.gat2 = SuperGATConv(64 * heads, 32, is_undirected=True, heads=4, attention_type='SD',
        # add_self_loops=True)

        self.sag_pool = SAGPooling(16 * heads, 8 * heads, GNN=GATConv)

        self.linear = torch.nn.Linear(8 * heads * 2, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        x = self.gat1(x, edge_index, edge_attr)

        # print(x.size())
        x = self.bn1(x)
        # print(x.size())

        x = self.relu(x)

        x = self.gat2(x, edge_index, edge_attr)

        # print(x.size())
        x = self.bn2(x)
        # print(x.size())

        x = self.relu(x)


        x = self.gat3(x, edge_index, edge_attr)

        # print(x.size())
        x = self.bn3(x)
        # print(x.size())

        x = self.relu(x)

        # x, edge_index, _, batch, param, score = self.sag_pool(x, edge_index, edge_attr, batch)
        # print(x.size())

        x = torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=1)

        # print(x.size())

        x = self.linear(x)

        x = self.bn4(x)

        # x = torch.relu(x)

        return x
