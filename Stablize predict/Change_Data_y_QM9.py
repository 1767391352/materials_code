
import torch
from torch_geometric.datasets import QM9
dataset = QM9(root="D:\\桌面\\materials code\\cif2graph_data\\dataset\\qm9.pth")

if __name__ == '__main__':
    new_dataset = []
    for data in dataset:

        data.y = data.y[0][4].clone().detach().float().unsqueeze(0)
        new_dataset.append(data)
    print(new_dataset[5].y)
    torch.save(new_dataset, "D:\\桌面\\materials code\\cif2graph_data\\dataset\\QM9_dataset_whole.pth")
