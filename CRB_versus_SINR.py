import numpy as np
import matplotlib.pyplot as plt
from utils.math import dB2pow
from LA_FAS.consts import Parameters
fas_parameters = Parameters.copy()
from LA_MIMO.consts import Parameters
mimo_parameters = Parameters.copy()
from LA_FAS.system import Fluid_Antenna_System
from LA_MIMO.system import MIMO_System
import time
# np.random.seed(10)  # fix the random parameters


objval_fas = []
objval_mimo = []
sinr_range = range(8, 20, 2)
for SINR in sinr_range:
    t_start = time.time()
    print(f'Current SINR: {SINR}(dB)')
    fas_parameters['Gamma'] = dB2pow(SINR)
    mimo_parameters['Gamma'] = dB2pow(SINR)
    _, crb_fas = Fluid_Antenna_System(**fas_parameters).run()
    objval_fas.append(crb_fas)
    _, crb_mimo = MIMO_System(**mimo_parameters).run()
    objval_mimo.append(crb_mimo)
    t_end = time.time()
    print(f'Time elapsed: {t_end - t_start:.3f} seconds, fas crb: {crb_fas:.3f}, mimo crb: {crb_mimo:.3f}')

# 绘图
plt.figure()
plt.plot(sinr_range, objval_fas, 'b-s', linewidth=1.5, label='Fluid Antenna')
plt.plot(sinr_range, objval_mimo, 'r-o', linewidth=1.5, label='Traditional MIMO')
plt.grid(True)
plt.xlabel('Qos Threshold (dB)')
plt.ylabel(r'Root CRB of DoA estimation $(\theta°)$')
plt.legend(loc='best')

# 保存 图片
plt.savefig('figs/CRB_SINR.png', format='png', dpi=300)
plt.savefig('figs/CRB_SINR.eps', format='eps')
plt.show()
