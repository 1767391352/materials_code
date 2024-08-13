import codecs
import concurrent.futures
import os
import warnings

import numpy
import numpy as np
import torch
from monty.serialization import loadfn
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from torch_geometric.data import Data, InMemoryDataset

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
band_structure_order = ['1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p', '6s', '4f', '5d', '6p', '7s']


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='weights/checkpoint.pth.tar', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class CustomDataset(InMemoryDataset):
    def __init__(self, dataset):
        super(CustomDataset, self).__init__(dataset)
        # List of raw files, in your case point cloud
        self.dataset = dataset

    def _download(self):
        pass

    def processed_file_names(self):
        """ return list of files should be in processed dir, if found - skip processing."""
        pass

    def download(self):
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        # assert dmin <= dmax
        # assert dmax - dmin > step

        self.flag = 1
        if dmax == dmin:
            self.flag = 0
        if self.flag == 1:
            self.filter = numpy.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        if self.flag == 1:
            return numpy.exp(-(distances[..., numpy.newaxis] - self.filter) ** 2 /
                             self.var ** 2)
        else:
            return numpy.exp((0 * distances[..., numpy.newaxis]))


def mae_metric(prediction, target):
    mae = torch.mean(torch.abs(prediction - target))
    return mae


def build_dict():
    write_file = codecs.open("D:\\桌面\\materials code\\cif2graph_data\\label\\is_stable.txt", "w", "utf_8")
    listdir_json = os.listdir("D:\\桌面\\materials code\\cif2graph_data\\json\\")
    for filename in listdir_json:
        print(filename)
        whole_json = loadfn("D:\\桌面\\materials code\\cif2graph_data\\json\\" + filename)

        for json in whole_json:
            # if json["e_ionic"] is not None:
            #     print(json["e_ionic"])
                # for item in json:
                #     print(item + ":" + str(json[item]))
                # print("----------------------------------------------------------------------")
                # is_stable_dic[json["material_id"]] = json["is_stable"]
                write_file.write(json["material_id"] + ":" + str(json["formation_energy_per_atom"]) + "\n")

    write_file.close()


def muti_process(listdir_cif, cnn, filename, lable_dic):
    # for filename in listdir_cif:

    # if num % 10000 == 0:
    #     break
    parser = CifParser("D:\\桌面\\materials code\\cif2graph_data\\cif\\" + filename)
    structure = parser.parse_structures(primitive=False)[0]
    analyzer = SpacegroupAnalyzer(structure)
    structure = analyzer.get_conventional_standard_structure()
    # print(structure)
    a = structure.lattice.a
    b = structure.lattice.b
    c = structure.lattice.c
    # abc_max = max(a, max(b, c))

    whole_information = [a, b, c,
                         structure.lattice.alpha, structure.lattice.beta,
                         structure.lattice.gamma, structure.lattice.volume]

    # 将 whole_information 转换为张量
    tensor_whole_info = torch.tensor(whole_information).unsqueeze(0)  # 增加一维，使其成为 (1, 7) 形状

    # x 是一个包含原子质量的列表
    x = []

    for site in structure.sites:

        # a = Element(site.specie.symbol)
        # print(a.x, a.ion_energies, a.nvalence_electrons, a.period)

        nano_x = [site.specie.number, site.specie.atomic_mass, site.x / a, site.y / b, site.z / c,
                  site.specie.atomic_radius,
                  site.specie.average_anionic_radius,
                  site.specie.average_cationic_radius,
                  site.specie.electron_affinity,
                  site.specie.group,
                  site.specie.row]

        for band in band_structure_order:
            try:
                if band in site.specie.atomic_orbitals_eV:
                    nano_x.append(site.specie.atomic_orbitals_eV[band])
                else:
                    nano_x.append(0.0)
            except Exception as e:
                # print(e)
                # print(site.specie.number)
                nano_x.append(0.0)
        # print(nano_x)
        x.append(nano_x)
        # print(x)

    # x = [int(site.specie.atomic_mass) for site in structure.sites]

    # 初始化一个空列表来存储新的张量
    new_tensors = []

    # 对于 x 中的每个原子质量，创建一个新的张量并将其与 whole_information 拼接
    for xmass in x:
        # 将单个原子质量转换为张量，并增加一维，使其与 whole_information 的维度匹配
        tensor_xmass = torch.tensor(xmass, dtype=torch.float).unsqueeze(0)

        # 拼接 whole_information 和单个原子质量的张量
        # dim=1 表示沿着列方向拼接
        new_tensor = torch.cat((tensor_xmass, tensor_whole_info), dim=1)
        new_tensors.append(new_tensor)

    # 将新的张量列表转换为一个整体的张量
    x = torch.cat(new_tensors, dim=0)
    # print(x)
    try:
        graph = cnn.get_bonded_structure(structure)
    except Exception as e:
        print(e)

    edge_index = []
    edge_attr = []
    distances = []

    for i, site in enumerate(structure):
        neighbors = graph.get_connected_sites(i)
        for neighbor in neighbors:
            j = neighbor.index
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_vector_x = (structure[i].x - structure[j].x) / a
            edge_vector_y = (structure[i].y - structure[j].y) / b
            edge_vector_z = (structure[i].z - structure[j].z) / c
            # 添加键长
            distance = structure.get_distance(i, j)

            distances.append(distance)
            distances.append(distance)
            # 计算键角，假设三个连续的原子i, j, k形成的角
            # angles = []
            # for k_site in graph.get_connected_sites(j):
            #     k = k_site.index
            #     if k != i:  # 避免重复计算同一对原子
            #         angle = get_angle(structure[i].coords - structure[j].coords,
            #                           structure[k].coords - structure[j].coords)
            #         angles.append(angle)
            # # 使用平均角度作为特征
            # mean_angle = numpy.mean(angles) if angles else 0
            # 计算键价
            edge_attr.append([-edge_vector_x, -edge_vector_y, -edge_vector_z])
            edge_attr.append([edge_vector_x, edge_vector_y, edge_vector_z])
    gauss = GaussianDistance(dmin=0, dmax=8, step=0.2)

    for i in range(len(distances)):
        edge_attr[i].extend(gauss.expand(distances[i]))

        # print(edge_attr[i])
    # edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    # edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # flag = lable_dic[filename]
    y = [lable_dic[filename]]
    # for i in range(len(structure.sites)):
    #     y.append(flag)
    # y = torch.tensor(y, dtype=torch.long)

    data = Data(x=x,
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float), y=torch.tensor(y, dtype=torch.float), cif=filename)

    # 打印 PyG 数据集
    # print("\nPyG Data:")

    # # print(data.x)
    # print(data.edge_index)
    # print(data.edge_attr)
    return data


def build_dataset():
    num = 1

    lable_dic = {}
    read_file = codecs.open("D:\\桌面\\materials code\\cif2graph_data\\label\\is_stable.txt", "r", "utf_8")
    listdir_json = os.listdir("D:\\桌面\\materials code\\cif2graph_data\\json\\")
    for line in read_file.readlines():
        dict = line.strip().split(":")
        if dict[1] != 'None':

            lable_dic[dict[0] + ".cif"] = float(dict[1])

    print(lable_dic)

    listdir_cif = os.listdir("D:\\桌面\\materials code\\cif2graph_data\\cif\\")
    output_cifs = []
    for filename in listdir_cif:
        if filename in lable_dic.keys():

            output_cifs.append(filename)
    print(len(output_cifs))
    cnn = CrystalNN()
    dataset = []
    file_names = []
    num = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        # 使用线程池执行任务
        futures = {executor.submit(muti_process, listdir_cif, cnn, filename, lable_dic) for filename in
                   output_cifs}
        for future in concurrent.futures.as_completed(futures):
            try:
                data = future.result()

                if data is not None:

                    dataset.append(data)
                    num += 1
                    if num % 100 == 0:
                        print(num / len(lable_dic) * 100, "%")
                    # if num == 100:
                    #     torch.save(dataset, "D:\\桌面\\materials code\\cif2graph_data\\dataset\\dataset_whole.pth")
                    #     exit(0)
                    # print(num)
            except Exception as e:
                print(f"Error processing123 {e}")

    # muti_process(listdir_cif, num, cnn, lable_dic)

    # for filename in listdir_cif:
    #     file_names.append(filename)
    #     num += 1
    #     if num % 100 == 0:
    #         print(num / 153235 * 100, "%")
    #     # if num % 10000 == 0:
    #     #     break
    #     parser = CifParser("D:\\桌面\\materials code\\cif2graph_data\\cif\\" + filename)
    #     structure = parser.parse_structures()[0]
    #     # print(structure)
    #     a = structure.lattice.a
    #     b = structure.lattice.b
    #     c = structure.lattice.c
    #     abc_max = max(a, max(b, c))
    #
    #     whole_information = [a, b, c,
    #                          structure.lattice.alpha, structure.lattice.beta,
    #                          structure.lattice.gamma, structure.lattice.volume]
    #
    #     # 将 whole_information 转换为张量
    #     tensor_whole_info = torch.tensor(whole_information).unsqueeze(0)  # 增加一维，使其成为 (1, 7) 形状
    #
    #     # x 是一个包含原子质量的列表
    #     x = []
    #
    #     for site in structure.sites:
    #         x.append([site.specie.atomic_mass, site.x / a, site.y / b, site.z / c])
    #     # x = [int(site.specie.atomic_mass) for site in structure.sites]
    #
    #     # 初始化一个空列表来存储新的张量
    #     new_tensors = []
    #
    #     # 对于 x 中的每个原子质量，创建一个新的张量并将其与 whole_information 拼接
    #     for xmass in x:
    #         # 将单个原子质量转换为张量，并增加一维，使其与 whole_information 的维度匹配
    #         tensor_xmass = torch.tensor(xmass).unsqueeze(0)
    #
    #         # 拼接 whole_information 和单个原子质量的张量
    #         # dim=1 表示沿着列方向拼接
    #         new_tensor = torch.cat((tensor_xmass, tensor_whole_info), dim=1)
    #         new_tensors.append(new_tensor)
    #
    #     # 将新的张量列表转换为一个整体的张量
    #     x = torch.cat(new_tensors, dim=0)
    #     # print(x)
    #     try:
    #         graph = cnn.get_bonded_structure(structure)
    #     except Exception as e:
    #         print(e)
    #         continue
    #     edge_index = []
    #     edge_attr = []
    #     distances = []
    #
    #     for i, site in enumerate(structure):
    #         neighbors = graph.get_connected_sites(i)
    #         for neighbor in neighbors:
    #             j = neighbor.index
    #             edge_index.append([i, j])
    #             edge_index.append([j, i])
    #             edge_vector_x = structure[i].x - structure[j].x
    #             edge_vector_y = structure[i].y - structure[j].y
    #             edge_vector_z = structure[i].z - structure[j].z
    #             # 添加键长
    #             distance = structure.get_distance(i, j)
    #
    #             distances.append(distance)
    #             distances.append(distance)
    #             # 计算键角，假设三个连续的原子i, j, k形成的角
    #             # angles = []
    #             # for k_site in graph.get_connected_sites(j):
    #             #     k = k_site.index
    #             #     if k != i:  # 避免重复计算同一对原子
    #             #         angle = get_angle(structure[i].coords - structure[j].coords,
    #             #                           structure[k].coords - structure[j].coords)
    #             #         angles.append(angle)
    #             # # 使用平均角度作为特征
    #             # mean_angle = numpy.mean(angles) if angles else 0
    #             # 计算键价
    #             valence = neighbor.site.specie.oxi_state if hasattr(neighbor.site.specie, 'oxi_state') else 0
    #             edge_attr.append([distance, -edge_vector_x, -edge_vector_y, -edge_vector_z, valence])
    #             edge_attr.append([distance, edge_vector_x, edge_vector_y, edge_vector_z, valence])
    #     gauss = GaussianDistance(dmin=0, dmax=abc_max, step=abc_max / 19)
    #
    #     # for i, site in enumerate(structure):
    #     #     print(i)
    #     # indices = [i for i, site in enumerate(structure)]
    #     # reference_indices = [i for i, site in enumerate(structure)]
    #     # print(indices)
    #     # r = rdf.RadialDistributionFunction(structures=structure, rmax=abc_max, indices=indices, reference_indices=reference_indices)
    #     # r, rdff = r.rdf, r.dr
    #     # print(r)
    #     # print(rdff)
    #     # print(edge_attr)
    #
    #     for i in range(len(distances)):
    #         edge_attr[i].extend(gauss.expand(distances[i]))
    #         # print(edge_attr[i])
    #     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    #     edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    #
    #     # flag = lable_dic[filename]
    #     y = [lable_dic[filename]]
    #     # for i in range(len(structure.sites)):
    #     #     y.append(flag)
    #     y = torch.tensor(y, dtype=torch.long)
    #
    #     data = Data(x=x,
    #                 edge_index=edge_index,
    #                 edge_attr=edge_attr, y=y)
    #
    #     # 打印 PyG 数据集
    #     # print("\nPyG Data:")
    #     # print(data)
    #     # # print(data.x)
    #     # print(data.edge_index)
    #     # print(data.edge_attr)
    #     dataset.append(data)
    # dataset = CustomDataset(dataset)
    print(dataset.__len__())
    torch.save(dataset, "D:\\桌面\\materials code\\cif2graph_data\\dataset\\MP_dataset_whole_Ef.pth")
    # dataset = torch.load("D:\\桌面\\materials code\\cif2graph_data\\dataset\\dataset_whole.pth")
    # dataset = CustomDataset(dataset)
    # print(dataset.get(3))
    # print(dataset.len())
    # dataset = torch.load("D:\\桌面\\materials code\\cif2graph_data\\dataset\\dataset_f10000.pth")
    # print(dataset.len())


if __name__ == '__main__':
    # build_dict()
    build_dataset()
    # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
