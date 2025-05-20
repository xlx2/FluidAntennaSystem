from LA_MIMO.consts import Parameters
from LA_MIMO.system import MIMO_System
Parameters_mimo = Parameters.copy()
from LA_FAS.consts import Parameters
from LA_FAS.system import Fluid_Antenna_System
Parameters_fas = Parameters.copy()
mimo = MIMO_System(**Parameters_mimo)
fas = Fluid_Antenna_System(**Parameters_fas)
for i in range(10):
    crb = mimo.run()
    print(crb)

# fas.run()