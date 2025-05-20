import numpy as np
import matplotlib.pyplot as plt
from utils.utils import dB2pow
from LA_FAS.consts import Parameters
from system import FAS_System, MIMO_System
import time


objval_fas = []
objval_mimo = []

for SINR in range(8, 13, 1):
    t_start = time.time()
    print(f'Current SINR: {SINR}(dB)')
    Parameters['qos_threshold'] = dB2pow(SINR)
    crb_fas = FAS_System(Parameters).run(verbose=False)
    objval_fas.append(crb_fas)
    crb_mimo = MIMO_System(Parameters).run(verbose=False)
    objval_mimo.append(crb_mimo)
    t_end = time.time()
    print(f'Time elapsed: {t_end - t_start:.3f} seconds, fas crb: {crb_fas:.3f}, mimo crb: {crb_mimo:.3f}')

# 绘图
plt.figure()
x_vals = np.arange(8, 13, 1) 
plt.plot(x_vals, objval_fas, 'b-s', linewidth=1.5, label='Proposed')
plt.plot(x_vals, objval_mimo, 'r-o', linewidth=1.5, label='Traditional')
plt.grid(True)
plt.xlabel('Power Budget (dBm)')
plt.ylabel('Root CRB of DoA estimation (°)')
plt.legend(loc='best')

# 保存 PDF
plt.savefig('Fig/CRB_Power.pdf', format='pdf')
plt.show()
