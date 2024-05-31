import csv
import os

import torch_geometric.utils
from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter

from monty.serialization import dumpfn, loadfn


# from torch_geometric.nn import MessagePassing
# import torch
def get_cif(docs):
    # 初始化 MPRester 对象
    listdir = os.listdir("./cif/")
    for doc in docs:
        if doc.material_id + ".cif" not in listdir:
            structure = mpr.get_structure_by_material_id(doc.material_id)
            print("downloading  to ./cif/" + doc.material_id + ".cif")
            try:
                cif_writer = CifWriter(structure)

                cif_writer.write_file("./cif/" + doc.material_id + ".cif")
            except Exception as e:
                print(e)
        else:
            print("downloaded")


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def get_json(docs):
    i = 1
    for chunk in chunker(docs, 1000):
        dumpfn(chunk, "materials_summary" + str(i) + ".json")
        i += 1


if __name__ == '__main__':
    # 替换为您的 API 密钥
    API_KEY = "123456"
    with MPRester(API_KEY) as mpr:
        # 使用 summary.search 方法搜索所有材料ID
        # 通过指定 fields 参数为 ["material_id"] 来只获取材料ID
        docs = mpr.materials.summary.search()

        print("getting docs done.")

        # 读取所有数据信息
    get_json(docs)
    # materials_summary = loadfn("materials_summary.json")

    # get_cif(docs)
