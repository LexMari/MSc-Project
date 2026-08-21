"""Batch experiment runner

A batch config references a base scenario YAML and sweeps one or more
fields across a set of values, running the full cross product and
writing one row per (run, vehicle) to CSV.

Batch config format:

  name: "..."
  base_scenario: "scenarios/some_scenario.yaml"
  output_csv: "results/some_sweep.csv"
  sweep:
    random_seed: {count: 30}
    solo.fusion_policy: [...]             # dotted "vehicle_id.field" targets a specific vehicle
    visibility: [1.0, 0.5, 0.2]           # no dot targets a ScenarioConfig field

Every combination of the swept values is run once. A field with a dot
in its name is applied to the named vehicle's VehicleConfig, everything
else is applied to ScenarioConfig.
"""

import csv
import itertools
import os
import yaml

from .scenario import load_scenario
from .engine import Simulation
from .units import ms_to_mph

def _resolve_sweep_values(spec):
    """Expand a sweep spec into the list of values to run"""
    if isinstance(spec, dict) and "count" in spec:
        return list(range(spec["count"]))
    return spec

def _apply_override(config, field, value):
    """apply one swept value to a loaded ScenarioConfig

    a dotted vehicle_id.field targets that vehicle's VehicleConfig
    anything else is applied to ScenarioConfig"""
    if "." in field:
        vehicle_id, attr = field.split(".", 1)
        target = next((v for v in config.vehicles if v.vehicle_id == vehicle_id), None)
        if target is None:
            raise ValueError(
                f"no vehicle {vehicle_id!r} in base scenario to apply sweep field {field!r} to"
            )
    else:
        target, attr = config, field

    if not hasattr(target, attr):
        raise ValueError(
            f"{attr!r} is not a field of {type(target).__name__} "
            f"(sweep field {field!r})"
        )

    setattr(target, attr, value)

def run_batch(batch_config_path: str, quiet: bool = False) -> list[dict]:
    """Runs every combination in a batch config, writes the results to its output_csv"""
    with open(batch_config_path) as f:
        raw = yaml.safe_load(f)

    base_scenario_path = raw["base_scenario"]
    output_csv = raw["output_csv"]
    sweep = raw.get("sweep", {})

    sweep_fields = list(sweep.keys())
    sweep_values = [_resolve_sweep_values(sweep[f]) for f in sweep_fields]
    combos = list(itertools.product(*sweep_values))

    rows = []
    for run_id, combo in enumerate(combos):
        combo_dict = dict(zip(sweep_fields, combo))
        config = load_scenario(base_scenario_path)
        for field, value in combo_dict.items():
            _apply_override(config, field, value)

        sim = Simulation(config)
        log = sim.run()

        by_vehicle: dict[str, list] = {}
        for entry in log:
            by_vehicle.setdefault(entry.vehicle_id, []).append(entry)

        for vehicle_id, entries in sorted(by_vehicle.items()):
            collided = any(e.collision for e in entries)
            severity = next((e.severity for e in entries if e.severity is not None), None)
            collision_time = next((e.time for e in entries if e.collision), None)
            min_speed_mph = round(ms_to_mph(min(e.speed for e in entries)), 2)
            final_speed_mph = round(ms_to_mph(entries[-1].speed), 2)

            row = {"run_id": run_id, **combo_dict, "vehicle_id": vehicle_id,
                   "collided": collided, "severity": severity or "",
                   "collision_time": collision_time if collision_time is not None else "",
                   "min_speed_mph": min_speed_mph, "final_speed_mph": final_speed_mph}
            rows.append(row)

        if not quiet:
            print(f"run {run_id + 1}/{len(combos)}: {combo_dict}")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    fieldnames = ["run_id"] + sweep_fields + ["vehicle_id", "collided", "severity",
                                                "collision_time", "min_speed_mph", "final_speed_mph"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows
