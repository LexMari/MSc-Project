"""Editable scenario representation for the GUI
editor - mirrors the YAML schema load_scenario()/ScenarioConfig

Only converted to the real schema at save time or read from an existing
file at load time
"""
import yaml

FUSION_POLICIES = ["camera_priority", "majority_vote", "confidence_weighted", "plausibility_filtered"]
GPS_POLICIES = ["naive", "plausibility_checked"]
ATTACK_TYPES = ["radar_spoof", "lidar_spoof", "camera_phantom", "gps_spoof", "radar_jam", "camera_jam", "lidar_jam"]
HAZARD_TYPES = ["pedestrian_crossing", "obstacle_in_road"]
PEDESTRIAN_TYPES = ["adult", "child", "elderly"]
FEATURE_TYPES = ["junction", "roundabout"]


class EditableScenario:
    def __init__(self):
        self.name = "New scenario"
        self.duration = 20.0
        self.timestep = 0.1
        self.random_seed = None
        self.visibility = 1.0
        self.track_straight_length = 200.0
        self.track_radius = 60.0
        self.track_features = []
        self.vehicles = []
        self.attacks = []
        self.hazards = []
        self.spawner = None

    def available_feature_ids(self) -> list[str]:
        if self.track_features:
            return [f.get("feature_id") for f in self.track_features if f.get("feature_id")]
        return ["junction_1", "roundabout_1"]

    def feature_position(self, feature_id: str) -> float | None:
        import math
        if self.track_features:
            for f in self.track_features:
                if f.get("feature_id") == feature_id:
                    return f.get("position")
            return None
        if feature_id == "junction_1":
            return 1.5 * self.track_straight_length + math.pi * self.track_radius
        if feature_id == "roundabout_1":
            return 0.75 * self.track_straight_length
        return None

    def track_total_length(self) -> float:
        import math
        return 2 * self.track_straight_length + 2 * math.pi * self.track_radius

    def to_yaml_dict(self) -> dict:
        d = {"name": self.name, "duration": self.duration, "timestep": self.timestep, "vehicles": self.vehicles}
        if self.random_seed is not None:
            d["random_seed"] = self.random_seed
        if self.visibility != 1.0:
            d["visibility"] = self.visibility
        if self.track_straight_length != 200.0 or self.track_radius != 60.0 or self.track_features:
            d["track"] = {"straight_length": self.track_straight_length, "radius": self.track_radius, "features": self.track_features}
        if self.attacks:
            d["attacks"] = self.attacks
        if self.hazards:
            d["hazards"] = self.hazards
        if self.spawner:
            d["spawner"] = self.spawner
        return d

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_yaml_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml_path(cls, path: str) -> "EditableScenario":
        with open(path) as f:
            raw = yaml.safe_load(f)
        es = cls()
        es.name = raw.get("name", "Untitled")
        es.duration = raw.get("duration", 20.0)
        es.timestep = raw.get("timestep", 0.1)
        es.random_seed = raw.get("random_seed")
        es.visibility = raw.get("visibility", 1.0)
        track = raw.get("track", {})
        es.track_straight_length = track.get("straight_length", 200.0)
        es.track_radius = track.get("radius", 60.0)
        es.track_features = track.get("features", [])
        es.vehicles = raw.get("vehicles", [])
        es.attacks = raw.get("attacks", [])
        es.hazards = raw.get("hazards", [])
        es.spawner = raw.get("spawner")
        return es

    def validate(self) -> list[str]:
        problems = []
        vehicle_ids = [v.get("vehicle_id") for v in self.vehicles]
        if not self.vehicles:
            problems.append("needs at least one vehicle")
        if len(vehicle_ids) != len(set(vehicle_ids)):
            problems.append("vehicle names must be unique")
        for v in self.vehicles:
            if not v.get("vehicle_id"):
                problems.append("every vehicle needs a name")

        feature_ids = set(self.available_feature_ids())
        for a in self.attacks:
            target = a.get("target_vehicle")
            if target not in vehicle_ids:
                problems.append(f"attack targets unknown vehicle {target!r}")
            for key in ("trigger_before_feature", "trigger_after_feature"):
                if key in a and a[key] not in feature_ids:
                    problems.append(f"attack's {key} refers to unknown feature {a[key]!r}")
        for h in self.hazards:
            if "feature_id" in h and h["feature_id"] not in feature_ids:
                problems.append(f"hazard refers to unknown feature {h['feature_id']!r}")
        if self.duration <= 0:
            problems.append("duration must be positive")
        if self.timestep <= 0:
            problems.append("timestep must be positive")

        custom_feature_ids = [f.get("feature_id") for f in self.track_features]
        if len(custom_feature_ids) != len(set(custom_feature_ids)):
            problems.append("track feature IDs must be unique")
        for f in self.track_features:
            if not f.get("feature_id"):
                problems.append("every track feature needs an ID")

        if self.spawner:
            if self.spawner.get("feature_id") not in feature_ids:
                problems.append(f"spawner refers to unknown feature {self.spawner.get('feature_id')!r}")

        return problems
