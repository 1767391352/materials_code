from torch_geometric.datasets import QM9
dataset = QM9(root="D:\\桌面\\materials code\\cif2graph_data\\dataset\\qm9.pth")


if __name__ == '__main__':
    for data in dataset:
        print(data)
        # print(data.x)
        # print(data.edge_attr)
        # print(data.edge_index)
        # print(data.y)
        # print(data.idx)
        # print(data.z)