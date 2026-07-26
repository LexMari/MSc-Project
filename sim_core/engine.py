"""simulation engine - puts vehicle physics, track geometry, sensor
generation, attack injection, and fusion together into a scenario.
the only module that knows about all the others - everything else stays separate
"""
from dataclasses import dataclass

from .vehicle import VehicleState
from .track import Track
from .junction import TrafficLight
from .sensors import SensorReading, SensorType
from .attacks import Attack, ATTACK_TYPES
from .fusion import FusionPolicy, FusedBelief, FUSION_POLICIES
from .scenario import ScenarioConfig

BRAKE_TRIGGER_DISTANCE = 10.0  # metres
BRAKE_DECELERATION = -6.0      # m/s^2
CRUISE_ACCELERATION = 0.0      # m/s^2

# TODO: wire TrafficLight state into vehicle braking
# decisions - currently junction/traffic light exist geometrically
# (see Track.junction_at, TrafficLight.state_at) but are
# consulted by any vehicle controller - same for roundabout give-way
# logic at Track.roundabout_at

@dataclass
class TickResult:
    """one simulation ticks worth of data per vehicle - this is what
    later gets logged, plotted, or fed into the outcome/severity model."""
    time: float
    vehicle_id: str
    position: tuple[float, float]
    speed: float
    fused_belief: FusedBelief

class Simulation:
    """runs a single scenario for its full configured duration"""

    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.time = 0.0
        self.track = Track()
        self.traffic_light = TrafficLight()

        self.vehicles: dict[str, VehicleState] = {
            vc.vehicle_id: VehicleState(
                vehicle_id=vc.vehicle_id,
                s=vc.start_distance,
                speed=vc.start_speed,
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

    def _ground_truth_readings(self, vehicle_id: str) -> dict[SensorType, SensorReading]:
        """builds clean sensor readings for one vehicle, based
        on nearest vehicle ahead of it on the loop. Deliberately
        simple - radar and camera both report the same ground-truth
        distance/velocity here, since the point of this skeleton is the
        attack/fusion logic, not sensor physics"""
        vehicle = self.vehicles[vehicle_id]
        x, y, _heading = self.track.position_at(vehicle.s)
        readings: dict[SensorType, SensorReading] = {
            SensorType.GPS: SensorReading(SensorType.GPS, detected_position=(x, y)),
        }

        others = [v for v in self.vehicles.values() if v.vehicle_id != vehicle_id]
        if others:
            nearest = min(others, key=lambda v: self.track.distance_ahead(vehicle.s, v.s))
            distance = self.track.distance_ahead(vehicle.s, nearest.s)
            closing_speed = vehicle.speed - nearest.speed
            readings[SensorType.RADAR] = SensorReading(SensorType.RADAR, detected_distance=distance, detected_velocity=closing_speed)
            readings[SensorType.CAMERA] = SensorReading(SensorType.CAMERA, detected_distance=distance)
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

            if belief.obstacle_present and belief.distance_to_obstacle is not None \
                    and belief.distance_to_obstacle < BRAKE_TRIGGER_DISTANCE:
                vehicle.acceleration = BRAKE_DECELERATION
            else:
                vehicle.acceleration = CRUISE_ACCELERATION

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