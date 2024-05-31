import codecs
import matplotlib.pyplot as plt
from tqdm import tqdm
import model
from utils import CustomDataset
from torch import optim, nn
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = torch.load("D:\\桌面\\materials code\\cif2graph_data\\dataset\\train_dataset_whole.pth")


test_dataset = CustomDataset(dataset)

print("train_dataset.len():       ", test_dataset.__len__())

model = model.Is_Stable_GNN(test_dataset.num_features, 2)

optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-1)

# train_dataset, test_dataset = train_test_split_edges(dataset.dataset, test_ratio=0.2)

test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
# print(train_loader.__dict__)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)



if __name__ == '__main__':
    # for data in train_loader:
    #     print(data)
    model = model.to(device)
    try:
        model.load_state_dict(torch.load("D:\\桌面\\materials code\\cif2graph_data\\model\\Is_Stable_GNN.pt"))
        print("load success!")
    except Exception as e:
        print(e)
    model.eval()

    true_labels = []
    predicted_labels = []
    for data in test_loader:
        # print(data)
        with torch.no_grad():
        # data.x = data.x.float()  # 确保输入数据是 float 类型
            data = data.to(device)

            out = model(data)  # 前向传播
        # print(out)
        # print(data.y)
        _, predicted_indices = torch.max(out, dim=1)  # dim=1 表示跨类别维度

        predicted_labels.extend(predicted_indices.cpu().numpy())
        # 累积真实的类别索引
        true_labels.extend(data.y.cpu().numpy())
    metric = classification_report(true_labels, predicted_labels)
    print(metric)








