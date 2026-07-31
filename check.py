from sim_core.scenario import load_scenario
from sim_core.engine import Simulation

config = load_scenario('scenarios/pullaway_phantom.yaml')
sim = Simulation(config)
steps = int(config.duration / config.timestep)
checkpoints = {2.0, 4.0, 6.0, 8.0, 8.1, 8.5, 9.0, 10.0, 11.0, 12.0}
for _ in range(steps):
    sim.step(config.timestep)
    t = round(sim.time - config.timestep, 2)
    if t in checkpoints:
        lead = sim.vehicles["lead"]
        print(f"t={t}  lead.s={lead.s:.6f}  lead.speed={lead.speed:.6f}")