from LA_MIMO.consts import Parameters
from LA_MIMO.system import MIMO_System
from Channel.mimo import MIMO_Channel

mimo_parameters = Parameters.copy()
n_monte_carlo = 200
crb_results = []

for i in range(n_monte_carlo):
    try:
        print(f"[{i+1}/{n_monte_carlo}]")
        mimo_parameters['H'] = MIMO_Channel(N=mimo_parameters['N'], N_user=mimo_parameters['K']).get_channel()
        system = MIMO_System(**mimo_parameters)
        _, crb = system.run()
        crb_results.append(crb)
    except Exception as e:
        print(f"[Warning] Iteration {i+1} failed: {e}")
        continue

print(f"\nSuccess: {len(crb_results)} / {n_monte_carlo}")
if crb_results:
    print(f"Mean CRB: {sum(crb_results) / len(crb_results):.4f}")