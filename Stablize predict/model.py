import torch
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool, SAGPooling, global_add_pool, TransformerConv, GCNConv, \
    BatchNorm, VGAE, GAE, CGConv, LayerNorm



class VGAE_Encoder(torch.nn.Module):
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


class Is_Stable_GNN(nn.Module):

    def __init__(self, num_features, edge_features, num_classes, heads=16):
        super(Is_Stable_GNN, self).__init__()  # "add" 是聚合邻居特征的方式

        # self.gat1 = GATConv(num_features, 8, heads=heads,
        #                     add_self_loops=True, dropout=0.0)
        # self.gat2 = GATConv(8 * heads, 8, heads=heads,
        #                     add_self_loops=True, dropout=0.0)
        # self.gat3 = GATConv(8 * heads, 8, heads=heads,
        #                     add_self_loops=True, dropout=0.0)
        self.embedding = torch.nn.Linear(num_features, 8)
        self.cgcnn = CGConv(num_features, edge_features, batch_norm=True)
        # self.gcn1 = GCNConv(num_features, 16,
        #                     add_self_loops=True)
        # self.transformer = TransformerConv(num_features, 16, heads=heads,
        #                                    dropout=0.0, edge_dim=23)


        self.silu = torch.nn.SiLU()
        self.relu = torch.nn.ReLU()
        self.softplus = torch.nn.Softplus()
        self.bn1 = BatchNorm(8 * heads)
        self.ln = LayerNorm(num_features)
        self.bn3 = BatchNorm(8 * heads)
        self.ln4 = LayerNorm(num_classes)
        # self.gat2 = SuperGATConv(64 * heads, 32, is_undirected=True, heads=4, attention_type='SD',
        # add_self_loops=True)

        # self.sag_pool = SAGPooling(16 * heads, 8 * heads, GNN=GATConv)

        self.linear = torch.nn.Linear(num_features * 2, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        # x = self.gat1(x, edge_index, edge_attr)
        #
        # # print(x.size())
        # x = self.bn1(x)
        # # print(x.size())
        #
        # x = self.relu(x)
        #
        # x = self.gat2(x, edge_index, edge_attr)
        #
        # # print(x.size())
        # x = self.bn2(x)
        # # print(x.size())
        #
        # x = self.relu(x)
        #
        #
        # x = self.gat3(x, edge_index, edge_attr)
        #
        # # print(x.size())
        # x = self.bn3(x)
        # # print(x.size())
        #
        # x = self.relu(x)
        # x = self.embedding(x)
        x = self.cgcnn(x, edge_index, edge_attr)

        x = self.ln(x)
        #
        x = self.softplus(x)

        # x, edge_index, _, batch, param, score = self.sag_pool(x, edge_index, edge_attr, batch)

        x = torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=1)

        # print(x.size())

        x = self.linear(x)

        x = self.ln4(x)

        # x = torch.relu(x)

        return x
