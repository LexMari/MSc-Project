"""Scenario configuration loading - turns a YAML file into the objects the
engine needs, so experiments can be defined without touching code.

Scenario files specify vehicle speed in mph, the engine itself works
entirely in m/s internally. Conversion happens here once at load time.

Track layout (straight length, radius, and named features like junctions
and roundabouts) is optional per scenario - if omitted, Track's own
sensible defaults are used. This allows a scenario to define
multiple junctions, move the roundabout, etc, without every
existing scenario needing to specify a track layout it doesn't care about
"""
from dataclasses import dataclass, field
import yaml

from .units import mph_to_ms

@dataclass
class VehicleConfig:
    vehicle_id: str
    start_distance: float          # metres along the track (see Track in track.py) - e.g. track.junction_at - 150
    start_speed: float             # m/s - converted from start_speed_mph at load time, see load_scenario()
    fusion_policy: str = "camera_priority"
    lane: int = 0                  # 0 = own/correct-side lane, 1 = opposite lane (see vehicle.py)
    direction: int = 1             # 1 = normal traffic, -1 = oncoming traffic (see vehicle.py)
    gps_policy: str = "naive"      # "naive" | "plausibility_checked" - see navigation.py

    # Driver diversity: reaction_time/max_deceleration explicitly set
    # here always win. If left unset and braking_variation > 0, a value
    # is instead randomly drawn within +/- braking_variation (a fraction,
    # e.g. 0.15 = 15%) of the Highway Code-derived defaults in
    # braking.py, seeded by ScenarioConfig.random_seed for
    # reproducibility - see Simulation.__init__ in engine.py. Left at
    # its default (0.0)
    reaction_time: float | None = None
    max_deceleration: float | None = None
    braking_variation: float = 0.0

@dataclass
class TrackConfig:
    straight_length: float = 200.0
    radius: float = 60.0
    features: list[dict] = field(default_factory=list)  # each dict: feature_id, feature_type, position

@dataclass
class SpawnerConfig:
    """Background traffic spawner: a new vehicle enters the track at a
    named junction feature once per traffic-light cycle, with a randomly
    chosen lane/direction (see Simulation._maybe_spawn_vehicle in
    engine.py) - lane 0 joins normal traffic flow, lane 1 joins as
    oncoming traffic, both from the same physical entry point. A spawned
    vehicle despawns once it has completed exactly one full lap. There is
    no other despawn condition (age, random chance, etc.), and if the
    scenario ends first, that's fine - it never gets to finish its lap.
    Absent from a scenario entirely (the default), no spawning happens at
    all - every existing scenario is unaffected."""

    feature_id: str = "junction_1"
    max_concurrent: int = 5
    speed_mph: float = 30.0
    fusion_policy: str = "camera_priority"
    braking_variation: float = 0.0

@dataclass
class ScenarioConfig:
    name: str
    vehicles: list[VehicleConfig]
    attacks: list[dict] = field(default_factory=list)
    hazards: list[dict] = field(default_factory=list)
    duration: float = 30.0
    timestep: float = 0.1
    track: TrackConfig = field(default_factory=TrackConfig)
    random_seed: int | None = None   # seeds driver-diversity randomisation (see VehicleConfig.braking_variation). None gives a fresh, non-reproducible seed each run
    visibility: float = 1.0   # multiplier on MAX_SENSOR_RANGE/LIDAR_MAX_SENSOR_RANGE (engine.py), e.g. 0.5 = fog/rain halving effective sensor range. 1.0 = clear, unaffected - every existing scenario is unaffected by default.
    spawner: SpawnerConfig | None = None   # background traffic spawner - see SpawnerConfig above. None (default) means no spawning, every existing scenario unaffected.

def load_scenario(path: str) -> ScenarioConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    vehicles = []
    for v in raw["vehicles"]:
        v = dict(v)
        if "start_speed_mph" in v:
            v["start_speed"] = mph_to_ms(v.pop("start_speed_mph"))
        vehicles.append(VehicleConfig(**v))

    track_raw = raw.get("track", {})
    track = TrackConfig(
        straight_length=track_raw.get("straight_length", 200.0),
        radius=track_raw.get("radius", 60.0),
        features=track_raw.get("features", []),
    )

    return ScenarioConfig(
        name=raw.get("name", "unnamed"),
        vehicles=vehicles,
        attacks=raw.get("attacks", []),
        hazards=raw.get("hazards", []),
        duration=raw.get("duration", 30.0),
        timestep=raw.get("timestep", 0.1),
        track=track,
        random_seed=raw.get("random_seed"),
        visibility=raw.get("visibility", 1.0),
        spawner=SpawnerConfig(**raw["spawner"]) if "spawner" in raw else None,
    )