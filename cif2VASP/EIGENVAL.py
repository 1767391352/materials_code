import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('Agg')
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.plotter import BSDOSPlotter, \
    BSPlotter, BSPlotterProjected, DosPlotter
import configparser


def read_ini_file(file_path):
    # 创建一个ConfigParser对象
    config = configparser.ConfigParser()

    # 从指定的文件中读取配置
    config.read(file_path)

    # 获取所有的section
    sections = config.sections()

    # 从'settings' section中获取所有选项
    return int(config["settings"].get("vb_energy_range")), int(config["settings"].get("cb_energy_range"))


# read vasprun.xml，get band and dos information
bs_vasprun = Vasprun("./vasprun.xml", parse_projected_eigen=True)
bs_data = bs_vasprun.get_band_structure(line_mode=True, kpoints_filename="./KPOINTS")
print("energy:", bs_data.get_band_gap()["energy"], "ev")
dos_vasprun = Vasprun("./vasprun.xml")
dos_data = dos_vasprun.complete_dos

vb_energy_range, cb_energy_range = read_ini_file("./config.ini")
# set figure parameters, draw figure
banddos_fig = BSDOSPlotter(bs_projection='elements', dos_projection='elements',
                           vb_energy_range=vb_energy_range, cb_energy_range=cb_energy_range)
banddos_fig.get_plot(bs=bs_data, dos=dos_data)
plt.savefig('banddos_fig.png')
