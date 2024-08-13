from torch_geometric.datasets import QM9, Mat
dataset = QM9(root="D:\\桌面\\materials code\\cif2graph_data\\dataset\\qm9.pth")
import codecs
import os

import matplotlib.pyplot as plt
import torch
from monty.serialization import loadfn

if __name__ == '__main__':
    torch.save(dataset, "D:\\桌面\\materials code\\cif2graph_data\\dataset\\dataset_whole_QM9.pth")
    for data in dataset[:20]:
        print(data)
        print(data.x)
        print(data.edge_attr)
        print(data.edge_index)
        print(data.y)
        # print(data.idx)
        # print(data.z)