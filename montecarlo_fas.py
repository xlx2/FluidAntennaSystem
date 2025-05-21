from LA_FAS.consts import Parameters
from LA_FAS.system import Fluid_Antenna_System
from Channel.fas import FAS_Channel

fas_parameters = Parameters.copy()
n_monte_carlo = 1
crb_results = []

for i in range(n_monte_carlo):
    try:
        print(f"[{i+1}/{n_monte_carlo}]")
        fas_parameters['H'] = FAS_Channel(Ny=fas_parameters['N'], N_user=fas_parameters['K']).get_channel()
        system = Fluid_Antenna_System(**fas_parameters)
        _, crb = system.run(isPlot=True)
        crb_results.append(crb)
    except Exception as e:
        print(f"[Warning] Iteration {i+1} failed: {e}")
        continue

print(f"\nSuccess: {len(crb_results)} / {n_monte_carlo}")
if crb_results:
    print(f"Mean CRB: {sum(crb_results) / len(crb_results):.4f}")
