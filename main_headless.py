"""Runs a scenario with no visual layer and prints what happened, tick by
tick, for one vehicle. This exists so the core engine can be tested
before any pygame code exists.

By default, prints the FIRST vehicle listed in the scenario's YAML (which
happens to be the attacked one in every current example scenario). Pass a
second argument to print a different vehicle by name.
"""
import sys
from sim_core.scenario import load_scenario
from sim_core.engine import Simulation
from sim_core.units import ms_to_mph

def main(scenario_path: str, vehicle_id: str | None = None) -> None:
    config = load_scenario(scenario_path)
    sim = Simulation(config)
    log = sim.run()

    if vehicle_id is None:
        vehicle_id = config.vehicles[0].vehicle_id  # default: first vehicle in the scenario
        if vehicle_id not in {v.vehicle_id for v in config.vehicles}:
            raise ValueError(f"no such vehicle {vehicle_id!r} in this scenario")

    print(f"Scenario: {config.name}")
    print(f"Showing vehicle: {vehicle_id!r} (pass a second argument to choose a different one)")
    print(f"{'t':>5} | {'vehicle':>9} | {'speed(mph)':>10} | {'belief_dist':>11} | {'obstacle?':>9} | source")
    for entry in log:
        if entry.vehicle_id != vehicle_id:
            continue
        belief = entry.fused_belief
        dist_str = f"{belief.distance_to_obstacle:.1f}" if belief.distance_to_obstacle is not None else "-"
        speed_mph = ms_to_mph(entry.speed)
        print(f"{entry.time:5.1f} | {entry.vehicle_id:>9} | {speed_mph:10.2f} | "
              f"{dist_str:>11} | {str(belief.obstacle_present):>9} | {belief.source}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "scenarios/phantom_brake.yaml"
    vehicle = sys.argv[2] if len(sys.argv) > 2 else None
    main(path, vehicle)