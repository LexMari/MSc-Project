"""scenario configuration loading - turns YAML files into objects the
engine needs"""
from dataclasses import dataclass, field
import yaml


@dataclass
class VehicleConfig:
    vehicle_id: str
    start_distance: float          # metres along the track (see Track in track.py) - e.g. track.junction_at - 150
    start_speed: float
    fusion_policy: str = "camera_priority"


@dataclass
class ScenarioConfig:
    name: str
    vehicles: list[VehicleConfig]
    attacks: list[dict] = field(default_factory=list)
    duration: float = 30.0
    timestep: float = 0.1


def load_scenario(path: str) -> ScenarioConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    vehicles = [VehicleConfig(**v) for v in raw["vehicles"]]
    return ScenarioConfig(
        name=raw.get("name", "unnamed"),
        vehicles=vehicles,
        attacks=raw.get("attacks", []),
        duration=raw.get("duration", 30.0),
        timestep=raw.get("timestep", 0.1),
    )