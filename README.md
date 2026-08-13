# Analysing the Impact of Cyber Attacks on Autonomous Vehicles Through Simulation

This is a 2D simulation framework for testing how different sensor fusion policies affect a vehicle's resilience to sensor spoofing attacks (radar, LiDAR, camera and GPS) plus a GUI for building custom scenarios, and a batch tool for running a scenario across multiple parameters.

## Requirements

Python 3.12. Third-party dependencies: `pygame`, `matplotlib`, `pyyaml`

```
pip install pygame matplotlib pyyaml
```

Everything else (`sim_core`, the headless testing, batch tooling) has no dependency beyond the standard library, so headless runs and tests work without `pygame`/`matplotlib` installed if the GUI is not needed.

## Running a scenario
**GUI** - opens the scenario editor - pass a path to jump straight into
playback of that scenario instead:

```
python main_gui.py
python main_gui.py scenarios/radar_spoof_masking.yaml
```

**Headless** - runs one scenario tick-by-tick to the console, and
writes a full log CSV to `results/logs/` and `results/csv/`. Defaults to `scenarios/phantom_brake.yaml` if no path is given and an optional second argument filters the printed output to one named vehicle:

```
python main_headless.py scenarios/radar_spoof_masking.yaml
python main_headless.py scenarios/radar_spoof_masking.yaml solo
```

**Batch** - runs every combination in a batch config (see `batches/`
for examples) and writes the results to the `output_csv` path named in
that config:

```
python main_batch.py batches/radar_spoof_masking_by_policy.yaml
```

**Graph** - turns a batch's output CSV into collision-rate and
minimum-speed bar charts, saved to `results/graphs/`. The swept field
name is optional if the CSV only has one:

```
python main_graph.py results/radar_spoof_masking_by_policy.csv solo.fusion_policy
```

## Tests

```
pytest
```

`tests/test_engine.py` covers the simulation core (vehicle physics,
sensors, fusion policies, attacks, hazards, junctions/roundabouts,
collision detection). `tests/test_batch.py` covers the batch sweep
mechanism.

## Project structure

```
sim_core/           the simulation engine - no dependencies
 - attacks.py           radar_spoof, lidar_spoof, camera_phantom, gps_spoof, jam
 - batch.py             runs a scenario across a swept parameter, used by main_batch.py
 - braking.py           reaction time and stopping-distance model
 - engine.py            ties everything into a per-tick simulation - the only module that knows of all others
 - fusion.py            the four fusion policies under test
 - hazards.py           pedestrian crossings, obstacles in road
 - junction.py          traffic light logic
 - navigation.py        GPS policies, roundabout confusion
 - scenario.py          turns YAML scenarios into a runnable simulation
 - vehicle.py           vehicle state and kinematics
 - sensors.py           radar/camera/LiDAR/GPS models
 - severity.py          collision severity classification 
 - track.py             track geometry, features (junctions, roundabouts)
 - units.py             mph/m/s conversion helpers (all user inputted speed relies on mph, the system runs in m/s - batches require m/s input)

scenarios/              scenario YAML files
batches/                batch sweep configs
results/                batch CSV output, per-scenario default-run CSVs/logs, graphs
tests/                  pytest suite

main_gui.py             scenario editor + playback GUI
main_headless.py        single-scenario console runner
main_batch.py           runs a batch config
main_graph.py           creates a batch's graph from its output CSV
editable_scenario.py    scenario representation used by the GUI
editor_widgets.py       pygame widgets (text field, button, dropdown, slider, scroll panel) built for the editor
```
