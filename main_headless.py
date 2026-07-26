import sys
from sim_core.scenario import load_scenario
from sim_core.engine import Simulation


def main(scenario_path: str) -> None:
    config = load_scenario(scenario_path)
    sim = Simulation(config)
    log = sim.run()

    print(f"Scenario: {config.name}")
    print(f"{'t':>5} | {'vehicle':>9} | {'speed':>6} | {'belief_dist':>11} | {'obstacle?':>9} | source")
    for entry in log:
        if entry.vehicle_id != "follower":
            continue  # only print attacked vehicle's tick-by-tick belief
        belief = entry.fused_belief
        dist_str = f"{belief.distance_to_obstacle:.1f}" if belief.distance_to_obstacle is not None else "-"
        print(f"{entry.time:5.1f} | {entry.vehicle_id:>9} | {entry.speed:6.2f} | "
              f"{dist_str:>11} | {str(belief.obstacle_present):>9} | {belief.source}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "scenarios/phantom_brake.yaml"
    main(path)
