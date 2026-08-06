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
    print(f"{'t':>5} | {'vehicle':>9} | {'speed(mph)':>10} | {'lane':>4} | {'radar':>9} | {'camera':>9} | {'lidar':>9} | {'belief_dist':>11} | {'obstacle?':>9} | {'source':>9} | {'reacting_to':>22} | collision | roundabout")
    printed_any = False
    for entry in log:
        if entry.vehicle_id != vehicle_id:
            continue
        printed_any = True
        belief = entry.fused_belief
        dist_str = f"{belief.distance_to_obstacle:.1f}" if belief.distance_to_obstacle is not None else "-"
        speed_mph = ms_to_mph(entry.speed)
        collision_str = f"COLLISION({entry.severity})" if entry.collision else ""

        def sensor_str(reading):
            # a raw sensor reading, shown as its distance with a trailing
            # '*' if this reading was attacker-controlled - so an
            # attack that a fusion policy successfully ignores (e.g.
            # radar_spoof against camera_priority) is still visible here,
            # even though belief_dist never reflects it
            if reading is None or reading.detected_distance is None:
                return "-"
            marker = "*" if reading.is_attacked else ""
            return f"{reading.detected_distance:.1f}{marker}"

        radar_str = sensor_str(entry.radar_reading)
        camera_str = sensor_str(entry.camera_reading)
        lidar_str = sensor_str(entry.lidar_reading)

        if entry.roundabout_confused:
            confused_str = "MISSED EXIT"
        elif entry.roundabout_excursion_remaining > 0:
            confused_str = f"(looping, {entry.roundabout_excursion_remaining:.0f}m left)"
        else:
            confused_str = ""
        reacting_to = entry.ground_truth_kind or "-"
        print(f"{entry.time:5.1f} | {entry.vehicle_id:>9} | {speed_mph:10.2f} | {entry.lane:>4} | "
              f"{radar_str:>9} | {camera_str:>9} | {lidar_str:>9} | {dist_str:>11} | {str(belief.obstacle_present):>9} | {belief.source:>9} | {reacting_to:>22} | {collision_str:>9} | {confused_str}")

    if not printed_any:
        available = sorted({entry.vehicle_id for entry in log})
        print(f"\nNo data for {vehicle_id!r}. Vehicles actually present in this run: {available}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "scenarios/phantom_brake.yaml"
    vehicle = sys.argv[2] if len(sys.argv) > 2 else None
    main(path, vehicle)