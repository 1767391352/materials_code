import codecs
import os

import matplotlib.pyplot as plt
import torch
from monty.serialization import loadfn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = torch.load("D:\\桌面\\materials code\\cif2graph_data\\dataset\\dataset_whole.pth")


# train_dataset = CustomDataset(dataset[:8000])

# print("train_dataset.len():       ", train_dataset.__len__())
#
# model = model.Is_Stable_GNN(train_dataset.num_features, 2)


# train_dataset, test_dataset = train_test_split_edges(dataset.dataset, test_ratio=0.2)

# train_loader = DataLoader(dataset=train_dataset, batch_size=4000, shuffle=True)
# print(train_loader.__dict__)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# 初始化一些用于绘图的变量
def build_dict():
    write_file = codecs.open("D:\\桌面\\materials code\\cif2graph_data\\label\\result_dict.txt", "w", "utf_8")
    listdir_json = os.listdir("D:\\桌面\\materials code\\cif2graph_data\\json\\")
    for filename in listdir_json:
        print(filename)
        whole_json = loadfn("D:\\桌面\\materials code\\cif2graph_data\\json\\" + filename)

        for json in whole_json:
            if json["symmetry"]["symbol"] == "Pm-3m":

                # print(json["symmetry"]["symbol"])
            # for item in json:
            #     print(item + ":" + str(json[item]))
            # print("----------------------------------------------------------------------")
            # is_stable_dic[json["material_id"]] = json["is_stable"]
                write_file.write(json["material_id"] + ":" + str(json["energy_per_atom"]) + ":" + str(
                    json["formation_energy_per_atom"]) + ":" + str(json["energy_above_hull"]) + "\n")

    write_file.close()


fig, ax = plt.subplots()  # 创建一个图形和一个坐标轴
losses_whole = []  # 用于存储每个epoch的损失
losses_period = []
if __name__ == '__main__':
    build_dict()
    lable_dic = {}
    read_file = codecs.open("D:\\桌面\\materials code\\cif2graph_data\\label\\result_dict.txt", "r", "utf_8")
    for line in read_file.readlines():
        dict = line.strip().split(":")
        if dict[1] == "None" or dict[2] == "None" or dict[3] == "None":
            continue

        dict_list = []
        for i in range(len(dict)):
            if i == 2:
                dict_list.append(float(dict[i]))

        lable_dic[dict[0] + ".cif"] = dict_list
    # print(lable_dic)
    new_dataset = []
    for data in dataset:
        if data.cif in lable_dic:
            data.y = torch.tensor(lable_dic[data.cif], dtype=torch.float)
            new_dataset.append(data)
            # print(data.y[2])
            # print(data.y, data.cif)

    torch.save(new_dataset, "D:\\桌面\\materials code\\cif2graph_data\\dataset\\dataset_whole_y3.pth")
# print(numpy.array(data.y)[0])
