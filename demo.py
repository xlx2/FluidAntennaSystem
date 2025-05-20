from LA_MIMO.consts import Parameters
from LA_MIMO.system import MIMO_system
Parameters_mimo = Parameters.copy()
from LA_FAS.consts import Parameters
from LA_FAS.system import FluidAntennaSystem
Parameters_fas = Parameters.copy()
mimo = MIMO_system(**Parameters_mimo)
fas = FluidAntennaSystem(**Parameters_fas)
# print(mimo.run(isPlot=True))
fas.run(isPlot=True)