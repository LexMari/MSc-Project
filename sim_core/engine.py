"""Simulation engine - puts vehicle physics, track geometry, sensor
generation, attack injection, and fusion together into a scenario.
the only module that knows about all the others - everything else stays separate
"""
from dataclasses import dataclass

from .vehicle import VehicleState
from .track import Track, TrackFeature
from .junction import TrafficLight, STOP_LINE_OFFSET
from .hazards import HAZARD_TYPES
from .sensors import SensorReading, SensorType
from .attacks import Attack, ATTACK_TYPES
from .fusion import FusionPolicy, FusedBelief, FUSION_POLICIES
from .braking import required_deceleration
from .scenario import ScenarioConfig

RESUME_ACCELERATION = 2.0      # m/s^2 - comfortable acceleration back toward cruise_speed once no obstacle is believed present
MAX_SENSOR_RANGE = 150.0       # metres - candidates beyond this are treated as "nothing detected," not "a very distant obstacle"

# TODO: roundabout give-way logic at Track features_of_type("roundabout")

@dataclass
class TickResult:
    """One simulation tick's worth of data, per vehicle - this is what
    later gets fed into the outcome/severity model"""
    time: float
    vehicle_id: str
    position: tuple[float, float]
    speed: float
    fused_belief: FusedBelief

class Simulation:
    """Runs a single scenario for its full configured duration."""

    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.time = 0.0

        features = [TrackFeature(**f) for f in config.track.features]
        self.track = Track(straight_length=config.track.straight_length,
                            radius=config.track.radius, features=features)

        # every junction feature gets its own independent traffic light
        self.traffic_lights: dict[str, TrafficLight] = {
            f.feature_id: TrafficLight() for f in self.track.features_of_type("junction")
        }

        self.vehicles: dict[str, VehicleState] = {
            vc.vehicle_id: VehicleState(
                vehicle_id=vc.vehicle_id,
                s=vc.start_distance,
                speed=vc.start_speed,
                cruise_speed=vc.start_speed,
            )
            for vc in config.vehicles
        }

        self.fusion_policies: dict[str, FusionPolicy] = {
            vc.vehicle_id: FUSION_POLICIES[vc.fusion_policy]()
            for vc in config.vehicles
        }

        # attacks are stored as (target_vehicle_id, Attack instance) pairs
        self.attacks: list[tuple[str, Attack]] = []
        for a in config.attacks:
            attack_cls = ATTACK_TYPES[a["type"]]
            kwargs = {k: v for k, v in a.items() if k not in ("type", "target_vehicle")}
            self.attacks.append((a["target_vehicle"], attack_cls(**kwargs)))

        # hazards are ground truth, not attached to any specific vehicle
        self.hazards = []
        for h in config.hazards:
            hazard_cls = HAZARD_TYPES[h["type"]]
            kwargs = {k: v for k, v in h.items() if k != "type"}
            self.hazards.append(hazard_cls(**kwargs))

    def _ground_truth_readings(self, vehicle_id: str) -> dict[SensorType, SensorReading]:
        """builds clean sensor readings for one vehicle, based
        on nearest vehicle ahead of it on the loop. Deliberately
        simple - radar and camera both report the same ground-truth
        distance/velocity here, since the point of this skeleton is the
        attack/fusion logic, not sensor physics"""
        vehicle = self.vehicles[vehicle_id]
        x, y, _heading = self.track.position_at(vehicle.s)
        # RADAR and CAMERA are always present, even with nothing detected
        # (detected_distance=None) -- this matters because a phantom
        # attack needs to be able to fabricate a detection where ground
        # truth had genuinely nothing, which it can't do if there's no
        # reading object for it to act on in the first place.
        readings: dict[SensorType, SensorReading] = {
            SensorType.GPS: SensorReading(SensorType.GPS, detected_position=(x, y)),
            SensorType.RADAR: SensorReading(SensorType.RADAR),
            SensorType.CAMERA: SensorReading(SensorType.CAMERA),
        }

        candidates: list[tuple[float, float]] = []  # (distance_ahead, closing_speed)

        others = [v for v in self.vehicles.values() if v.vehicle_id != vehicle_id]
        if others:
            nearest = min(others, key=lambda v: self.track.distance_ahead(vehicle.s, v.s))
            distance = self.track.distance_ahead(vehicle.s, nearest.s)
            candidates.append((distance, vehicle.speed - nearest.speed))

        for hazard in self.hazards:
            if hazard.is_present(self.time):
                feature = self.track.feature(hazard.feature_id)
                distance = self.track.distance_ahead(vehicle.s, feature.position)
                candidates.append((distance, vehicle.speed))  # stationary hazard -> closing speed = vehicle's own speed

        for feature_id, light in self.traffic_lights.items():
            if light.state_at(self.time) != "green":
                feature = self.track.feature(feature_id)
                stop_line = feature.position - STOP_LINE_OFFSET
                distance = self.track.distance_ahead(vehicle.s, stop_line)
                candidates.append((distance, vehicle.speed))  # stop line doesn't move -> closing speed = vehicle's own speed

        if candidates:
            candidates = [c for c in candidates if c[0] <= MAX_SENSOR_RANGE]

        if candidates:
            nearest_distance, nearest_closing_speed = min(candidates, key=lambda c: c[0])
            readings[SensorType.RADAR] = SensorReading(SensorType.RADAR, detected_distance=nearest_distance, detected_velocity=nearest_closing_speed)
            readings[SensorType.CAMERA] = SensorReading(SensorType.CAMERA, detected_distance=nearest_distance)

        return readings

    def _apply_attacks(self, vehicle_id: str, readings: dict[SensorType, SensorReading]) -> dict[SensorType, SensorReading]:
        vehicle = self.vehicles[vehicle_id]
        for target_id, attack in self.attacks:
            if target_id != vehicle_id:
                continue
            attack.check_trigger(self.time, vehicle.s, self.track)
            sensor = attack.target_sensor
            if sensor in readings:
                readings[sensor] = attack.apply(readings[sensor], self.time)
        return readings

    def step(self, dt: float) -> list[TickResult]:
        results = []
        for vehicle_id, vehicle in self.vehicles.items():
            readings = self._ground_truth_readings(vehicle_id)
            readings = self._apply_attacks(vehicle_id, readings)
            belief = self.fusion_policies[vehicle_id].fuse(readings)

            if belief.obstacle_present and belief.distance_to_obstacle is not None:
                decel = required_deceleration(vehicle.speed, belief.distance_to_obstacle)
            else:
                decel = 0.0

            if decel > 0:
                vehicle.acceleration = -decel
            elif vehicle.speed < vehicle.cruise_speed:
                # nothing currently requires braking (either no obstacle
                # was detected, or one was detected but is far enough away
                # to need zero deceleration right now) - resume toward
                # cruise speed, capped so this tick can't overshoot it
                vehicle.acceleration = min(RESUME_ACCELERATION, (vehicle.cruise_speed - vehicle.speed) / dt)
            else:
                vehicle.acceleration = 0.0

            x, y, _heading = self.track.position_at(vehicle.s)
            results.append(TickResult(self.time, vehicle_id, (x, y), vehicle.speed, belief))

        for vehicle in self.vehicles.values():
            vehicle.step(dt)
        self.time += dt
        return results

    def run(self) -> list[TickResult]:
        """runs full scenario and returns every tick's results"""
        log: list[TickResult] = []
        steps = int(self.config.duration / self.config.timestep)
        for _ in range(steps):
            log.extend(self.step(self.config.timestep))
        return log