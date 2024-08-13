from typing import Tuple, Union

import torch
import torch.nn.functional as F
from deepkan import DeepKAN
from torch import Tensor
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv, \
    LayerNorm, CGConv, global_max_pool, global_mean_pool, global_add_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops


class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, num_heads=1):
        super(MultiHeadAttention, self).__init__()
        assert in_channels % num_heads == 0, "The number of channels must be divisible by the number of heads."
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_proj = Linear(in_channels, 3 * in_channels)
        self.out_proj = Linear(in_channels, in_channels)

    def forward(self, x):
        qkv = self.qkv_proj(x).view(-1, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        attention = torch.matmul(q * self.scale, k.transpose(-1, -2))
        attention = attention.softmax(dim=-1)
        out = torch.matmul(attention, v)
        out = out.view(-1, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        return out


class CGConvWithAttention(MessagePassing):
    def __init__(self, channels: Union[int, Tuple[int, int]], dim=0, aggr='add', batch_norm=False, bias=True,
                 num_heads=1, **kwargs):
        super(CGConvWithAttention, self).__init__(aggr=aggr, **kwargs)
        self.channels = channels
        self.dim = dim
        self.batch_norm = batch_norm
        self.num_heads = num_heads

        if isinstance(channels, int):
            channels = (channels, channels)

        self.lin_f = Linear(sum(channels) + dim, channels[1], bias=bias)
        self.lin_s = Linear(sum(channels) + dim, channels[1], bias=bias)
        self.attn = MultiHeadAttention(channels[1], num_heads=num_heads)

        if batch_norm:
            self.bn = LayerNorm(channels[1])
        else:
            self.bn = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        # self.attn.reset_parameters()
        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        if isinstance(x, torch.Tensor):
            x = (x, x)

        # Propagate messages using the attention mechanism
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        out = out if self.bn is None else self.bn(out)
        out = out + x[1]
        return out

    def message(self, x_i, x_j, edge_attr, index):
        if edge_attr is None:
            z = torch.cat([x_i, x_j], dim=-1)
        else:
            z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        # Original message calculation
        msg = torch.nn.functional.gelu(self.lin_f(z)) * torch.nn.functional.gelu(self.lin_s(z))
        # Apply attention over the messages
        msg = self.attn(msg)
        return msg

    def __repr__(self):
        return '{}({}, dim={})'.format(self.__class__.__name__, self.channels, self.dim)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # 均值向量
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # 方差向量
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))

        mu = self.fc21(h)
        logvar = self.fc22(h)
        return mu, logvar


# 解码器网络
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc3(z))
        x_recon = torch.sigmoid(self.fc4(h))
        return x_recon


# VAE模型
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


class VGAE_Encoder(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels):
        super(VGAE_Encoder, self).__init__()
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
    def __init__(self, num_features, edge_features, atom_hidden_layers=64, edge_hidden_layers=128, heads=8,
                 layer_nums=4, l1_num=4, l2_num=4):
        super(Is_Stable_GNN, self).__init__()
        self.atom_encoder = nn.Linear(num_features, atom_hidden_layers)
        self.edge_encoder = nn.Linear(edge_features, edge_hidden_layers)
        # self.atom_encoder = DeepKAN(num_features, [atom_hidden_layers])
        # self.edge_encoder = DeepKAN(edge_features, [edge_hidden_layers])
        self.cgcnn = CGConv(atom_hidden_layers, edge_hidden_layers, batch_norm=True)
        self.cgcnna = CGConvWithAttention(channels=atom_hidden_layers, dim=edge_hidden_layers, batch_norm=True,
                                          num_heads=4)
        # self.kan = DeepKAN(hidden_layers, [hidden_layers])
        self.silu = torch.nn.SiLU()
        self.gelu = torch.nn.GELU()
        # self.relu = torch.nn.ReLU()
        self.softplus = torch.nn.Softplus()
        self.dropout = nn.Dropout(0.2)
        self.ln1 = LayerNorm(atom_hidden_layers)
        self.ln2 = LayerNorm(edge_hidden_layers)
        self.layer_nums = layer_nums

        self.trans_one = nn.Linear(atom_hidden_layers * (1 + layer_nums), atom_hidden_layers)
        # self.trans_one = DeepKAN(atom_hidden_layers * (1 + layer_nums), [atom_hidden_layers])

        self.l1 = nn.ModuleList([nn.Linear(atom_hidden_layers, atom_hidden_layers)
                                 for _ in range(l1_num)])
        self.l2 = nn.ModuleList([nn.Linear(atom_hidden_layers, atom_hidden_layers)
                                 for _ in range(l2_num)])
        # self.l1 = nn.ModuleList([DeepKAN(atom_hidden_layers, [atom_hidden_layers])
        #                          for _ in range(l1_num)])
        # self.l2 = nn.ModuleList([DeepKAN(atom_hidden_layers, [atom_hidden_layers])
        #                          for _ in range(l2_num)])

        self.linear = torch.nn.Linear(atom_hidden_layers, 1)
        # self.l1 = DeepKAN(hidden_layers, [hidden_layers for _ in range(l1_num)])
        #
        # self.linear = DeepKAN(hidden_layers, [1])

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        x = self.ln1(self.atom_encoder(x))
        # x = self.ln1(x)
        edge_attr = self.ln2(self.edge_encoder(edge_attr))
        # edge_attr = self.ln2(edge_attr)

        # info_edge_attr = edge_attr
        # print(info.size())
        for _ in range(self.layer_nums):
            # info_x = x
            # x = self.cgcnn(x, edge_index, edge_attr)
            x = self.cgcnna(x, edge_index, edge_attr)
            # x = self.silu(x)
            # x = info_x + x
        #
        # x = info_x
        # x = self.trans_one(x)
        for hidden in self.l1:
            x = self.gelu(hidden(x))
            # x = hidden(x)

        x = global_mean_pool(x, batch)

        for hidden in self.l2:
            x = self.gelu(hidden(x))
            # x = hidden(x)

        x = self.linear(x).view(-1)

        # x = self.dropout(x)

        return x
