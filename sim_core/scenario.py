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

@dataclass
class TrackConfig:
    straight_length: float = 200.0
    radius: float = 60.0
    features: list[dict] = field(default_factory=list)  # each dict: feature_id, feature_type, position

@dataclass
class ScenarioConfig:
    name: str
    vehicles: list[VehicleConfig]
    attacks: list[dict] = field(default_factory=list)
    hazards: list[dict] = field(default_factory=list)
    duration: float = 30.0
    timestep: float = 0.1
    track: TrackConfig = field(default_factory=TrackConfig)

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
    )