from LA_FAS.consts import Parameters
from LA_FAS.system import Fluid_Antenna_System
from Channel.fas import FAS_Channel

fas_parameters = Parameters.copy()
for i in range(10):
    fas_parameters['H'] = FAS_Channel(Ny=fas_parameters['N'], N_user=fas_parameters['K']).get_channel()
    mimo = Fluid_Antenna_System(**fas_parameters)
    _, crb = mimo.run()
    print(crb)