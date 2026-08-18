"""Runs a scenario with no visual layer and prints what happened, tick by
tick, for one vehicle. This exists so the core engine can be tested
before any pygame code exists.

By default, prints the FIRST vehicle listed in the scenario's YAML (which
happens to be the attacked one in every current example scenario). Pass a
second argument to print a different vehicle by name.
"""
import csv
import os
import sys
from datetime import datetime

from sim_core.scenario import load_scenario
from sim_core.engine import Simulation
from sim_core.units import ms_to_mph

class Tee:
    """Writes everything to both the real stdout and a file at once, so
    existing print() calls don't need to change at all to also produce a
    saved record"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

CSV_FIELDS = ["time", "vehicle_id", "speed_mph", "lane", "radar_dist", "radar_attacked",
              "camera_dist", "camera_attacked", "lidar_dist", "lidar_attacked",
              "belief_dist", "obstacle_present", "belief_source", "reacting_to",
              "collision", "severity", "roundabout_confused", "roundabout_excursion_remaining"]

def main(scenario_path: str, vehicle_id: str | None = None, csv_path: str | None = None) -> None:
    config = load_scenario(scenario_path)
    sim = Simulation(config)
    log = sim.run()

    if vehicle_id is None:
        vehicle_id = config.vehicles[0].vehicle_id  # default: first vehicle in the scenario
        if vehicle_id not in {v.vehicle_id for v in config.vehicles}:
            raise ValueError(f"no such vehicle {vehicle_id!r} in this scenario")
    show_all = vehicle_id == "all"

    csv_writer = None
    csv_file = None
    if csv_path is not None:
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()

    print(f"Scenario: {config.name}")
    if show_all:
        print("Showing all vehicles, one after another")
    else:
        print(f"Showing vehicle: {vehicle_id!r} (pass a second argument to choose a different one (by name), or 'all' for every vehicle)")
    print(f"{'t':>5} | {'vehicle':>9} | {'speed(mph)':>10} | {'lane':>4} | {'radar':>9} | {'camera':>9} | {'lidar':>9} | {'belief_dist':>11} | {'obstacle?':>9} | {'source':>9} | {'reacting_to':>22} | collision | roundabout")
    printed_any = False

    if show_all:
        declared_order = [v.vehicle_id for v in config.vehicles]
        seen_order = list(dict.fromkeys(e.vehicle_id for e in log))
        vehicle_order = declared_order + [v for v in seen_order if v not in declared_order]
        by_vehicle = {vid: [e for e in log if e.vehicle_id == vid] for vid in vehicle_order}
        ordered_log = [e for vid in vehicle_order for e in by_vehicle[vid]]
    else:
        ordered_log = log

    current_vehicle = None
    for entry in ordered_log:
        if not show_all and entry.vehicle_id != vehicle_id:
            continue
        if show_all and entry.vehicle_id != current_vehicle:
            current_vehicle = entry.vehicle_id
            print(f"--- {current_vehicle} ---")
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

        if csv_writer is not None:
            def raw_dist(reading):
                return reading.detected_distance if reading is not None else None
            def raw_attacked(reading):
                return reading.is_attacked if reading is not None else False
            csv_writer.writerow({
                "time": entry.time, "vehicle_id": entry.vehicle_id, "speed_mph": speed_mph, "lane": entry.lane,
                "radar_dist": raw_dist(entry.radar_reading), "radar_attacked": raw_attacked(entry.radar_reading),
                "camera_dist": raw_dist(entry.camera_reading), "camera_attacked": raw_attacked(entry.camera_reading),
                "lidar_dist": raw_dist(entry.lidar_reading), "lidar_attacked": raw_attacked(entry.lidar_reading),
                "belief_dist": belief.distance_to_obstacle, "obstacle_present": belief.obstacle_present,
                "belief_source": belief.source, "reacting_to": entry.ground_truth_kind,
                "collision": entry.collision, "severity": entry.severity,
                "roundabout_confused": entry.roundabout_confused,
                "roundabout_excursion_remaining": entry.roundabout_excursion_remaining,
            })

    if csv_file is not None:
        csv_file.close()

    if not printed_any:
        available = sorted({entry.vehicle_id for entry in log})
        print(f"\nNo data for {vehicle_id!r}. Vehicles actually present in this run: {available}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "scenarios/phantom_brake.yaml"
    vehicle = sys.argv[2] if len(sys.argv) > 2 else None

    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/csv", exist_ok=True)
    scenario_stem = os.path.splitext(os.path.basename(path))[0]
    vehicle_stem = vehicle or "default"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"results/logs/{scenario_stem}_{vehicle_stem}_{timestamp}.txt"
    csv_path = f"results/csv/{scenario_stem}_{vehicle_stem}_{timestamp}.csv"

    real_stdout = sys.stdout
    with open(log_path, "w") as log_file:
        sys.stdout = Tee(real_stdout, log_file)
        try:
            main(path, vehicle, csv_path)
        finally:
            sys.stdout = real_stdout
    print(f"\n(full output also saved to {log_path})")
    print(f"(per-tick CSV also saved to {csv_path})")