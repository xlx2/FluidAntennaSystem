from LA_MIMO.consts import Parameters
from LA_MIMO.system import MIMO_System
from Channel.mimo import MIMO_Channel

mimo_parameters = Parameters.copy()
for i in range(10):
    mimo_parameters['H'] = MIMO_Channel(N=mimo_parameters['N'], N_user=mimo_parameters['K']).get_channel()
    mimo = MIMO_System(**mimo_parameters)
    _, crb = mimo.run()
    print(crb)