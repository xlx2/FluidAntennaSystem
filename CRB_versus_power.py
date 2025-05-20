import numpy as np
import matplotlib.pyplot as plt
from utils.math import dBm2pow
from LA_FAS.consts import Parameters
fas_parameters = Parameters.copy()
from LA_MIMO.consts import Parameters
mimo_parameters = Parameters.copy()
from LA_FAS.system import Fluid_Antenna_System
from LA_MIMO.system import MIMO_System
import time


objval_fas = []
objval_mimo = []

for P_dBm in range(30, 42, 2):
    t_start = time.time()
    print(f'Current P: {P_dBm}(dBm)')
    Parameters['power'] = dBm2pow(P_dBm)
    _, crb_fas = Fluid_Antenna_System(Parameters).run(verbose=False)
    objval_fas.append(crb_fas)
    crb_mimo = MIMO_System(Parameters).run(verbose=False)
    objval_mimo.append(crb_mimo)
    t_end = time.time()
    print(f'Time elapsed: {t_end - t_start:.3f} seconds, fas crb: {crb_fas:.3f}, mimo crb: {crb_mimo:.3f}')

# 绘图
plt.figure()
x_vals = np.arange(30, 42, 2)  # x 轴对应的 P_dBm 值
plt.plot(x_vals, objval_fas, 'b-s', linewidth=1.5, label='Proposed')
plt.plot(x_vals, objval_mimo, 'r-o', linewidth=1.5, label='Traditional')
plt.grid(True)
plt.xlabel('Power Budget (dBm)')
plt.ylabel('Root CRB of DoA estimation (°)')
plt.legend(loc='best')

# 保存 PDF
plt.savefig('Fig/CRB_Power.pdf', format='pdf')
plt.show()
