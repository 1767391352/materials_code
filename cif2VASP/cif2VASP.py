
from pymatgen.io.vasp.inputs import Potcar
import pymatgen.io.vasp.sets as sets
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

if __name__ == '__main__':

    # 从 CIF 文件读取结构
    parser = CifParser("D:\\桌面\\materials_code\\cif2graph_data\\cif\\" + "mp-510624.cif")
    structure = parser.parse_structures()[0]
    analyzer = SpacegroupAnalyzer(structure)
    standardized_structure = analyzer.get_conventional_standard_structure()
    # 创建一个 MPRelaxSet 对象
    # vasp_input_set = sets.MPRelaxSet(standardized_structure)
    vasp_input_set =sets.MPNonSCFSet(standardized_structure)
    # print(vasp_input_set.as_dict())
    # 将 VASP 输入文件写入到一个目录中
    vasp_input_set.write_input("./inputs")