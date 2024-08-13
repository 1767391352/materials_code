import matplotlib as mpl
import matplotlib.pyplot as plt
from pymatgen.electronic_structure.plotter import BSDOSPlotter
from pymatgen.io.vasp.outputs import Vasprun

mpl.use('Agg')
# read vasprun.xml，get band and dos information
bs_vasprun = Vasprun("./vasprun.xml", parse_projected_eigen=True)
bs_data = bs_vasprun.get_band_structure(line_mode=True)

dos_vasprun = Vasprun("./vasprun.xml")
dos_data = dos_vasprun.complete_dos

vb_energy_range, cb_energy_range = 5, 9
# set figure parameters, draw figure
banddos_fig = BSDOSPlotter(bs_projection='elements', dos_projection='elements',
                           vb_energy_range=vb_energy_range, cb_energy_range=cb_energy_range)
banddos_fig.get_plot(bs=bs_data, dos=dos_data)
plt.savefig('./band&dos_fig.png')
