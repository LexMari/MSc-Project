"""Scenario configuration loading - turns a YAML file into the objects the
engine needs

Scenario files specify vehicle speed in mph, the engine itself works
entirely in m/s internally. Conversion happens here once at load time.

Track layout (straight length, radius, and named features like junctions
and roundabouts) is optional per scenario - if omitted, Track's own
defaults are used. This allows a scenario to define
multiple junctions, move the roundabout, without every
existing scenario needing to specify a track layout
"""
from dataclasses import dataclass, field
import yaml

from .units import mph_to_ms

@dataclass
class VehicleConfig:
    vehicle_id: str
    start_distance: float          # metres along the track - e.g. track.junction_at - 150
    start_speed: float             # m/s - converted from start_speed_mph in load_scenario()
    fusion_policy: str = "camera_priority"
    lane: int = 0                  # 0 = own/correct-side lane, 1 = opposite lane (see vehicle.py)
    direction: int = 1             # 1 = normal traffic, -1 = oncoming traffic (see vehicle.py)
    gps_policy: str = "naive"      # "naive" | "plausibility_checked" - see navigation.py

    # Driver diversity. A defined reaction_time/max_deceleration always wins.
    # If unset and braking_variation > 0, a value is drawn within
    # +/- braking_variation (a fraction: 0.15 = 15%) of the Highway Code defaults,
    # seeded by ScenarioConfig.random_seed - see Simulation.__init__.
    reaction_time: float | None = None
    max_deceleration: float | None = None
    braking_variation: float = 0.0

@dataclass
class TrackConfig:
    straight_length: float = 200.0
    radius: float = 60.0
    features: list[dict] = field(default_factory=list)
    # Each dict: feature_id, feature_type, position, and radius for roundabouts.
    # Left as dicts rather than TrackFeature objects because Track builds
    # them itself - see Track.__post_init__.

@dataclass
class SpawnerConfig:
    """Background traffic entering at a named junction, one per light cycle.

    Lane and direction are chosen randomly per spawn (lane 0 joins normal flow,
    lane 1 joins as oncoming) from the same physical entry point - see
    Simulation._maybe_spawn_vehicle. A spawned vehicle despawns after
    one full lap. By default, this is absent from a scenario so nothing spawns.
    """

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
    random_seed: int | None = None   # seeds driver-diversity randomisation (see VehicleConfig.braking_variation). None gives a non-reproducible seed each run
    visibility: float = 1.0   # multiplier on MAX_SENSOR_RANGE/LIDAR_MAX_SENSOR_RANGE (engine.py), e.g. 0.5 = halving effective sensor range. 1.0 = clear, unaffected
    spawner: SpawnerConfig | None = None   # background traffic spawner - None (default) means no spawning

def load_scenario(path: str) -> ScenarioConfig:
    """Load and validate a scenario YAML file into a ScenarioConfig

    Mistakes made in configurations are also raised here since the scenario
    cannot run with errors."""
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