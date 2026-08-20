"""Simulation engine, this is the only module aware of all of the others

Tick order (see step): swerve decision -> sense -> inject attacks -> fuse ->
GPS belief -> control law -> collision detection -> integrate -> spawn/despawn.

Sensing acts on last tick's positions, and control on this tick's
belief, so a vehicle's response always lags the world by one timestep
"""
from dataclasses import dataclass
import random

from .vehicle import VehicleState
from .track import Track, TrackFeature
from .junction import TrafficLight, STOP_LINE_OFFSET
from .hazards import HAZARD_TYPES
from .sensors import SensorReading, SensorType
from .attacks import Attack, ATTACK_TYPES, Jam
from .fusion import FusionPolicy, FusedBelief, FUSION_POLICIES
from .navigation import GPSPolicy, GPS_POLICIES
from .braking import required_deceleration, can_stop_safely, REACTION_TIME, MAX_DECELERATION
from .scenario import ScenarioConfig
from .units import mph_to_ms
from . import severity

SEVERITY_RANK = {"slight": 0, "serious": 1, "fatal": 2}

RESUME_ACCELERATION = 2.0      # m/s^2 - acceleration back toward cruise_speed once no obstacle is believed present
MAX_SENSOR_RANGE = 150.0       # metres - candidates beyond this are treated as "nothing detected," not "a very distant obstacle" (radar/camera)
LIDAR_MAX_SENSOR_RANGE = 80.0  # metres - LiDAR's own, shorter effective range compared to radar/camera above
COLLISION_DISTANCE = 1.4       # in metres, must sit below the smallest gap of a correctly-stopped vehicle (~1.6m at 20mph) and above the tightest legitimate swerve pass (~1.17m)
PEDESTRIAN_COLLISION_DISTANCE = 1.0   # metres
VEHICLE_LENGTH = 4.5           # metres - used only for sensing (see _ahead_distance_to_vehicle), not for collision detection which stays based on reference-point distance and is separately calibrated via COLLISION_DISTANCE above
SWERVE_RETURN_BUFFER = 5.0     # metres of extra travel past the obstacle before swerving back into lane 0
SWERVE_TIME_SAFETY_MARGIN = 1.5  # a vehicle only swerves if the oncoming lane looks clear for at least this multiple of the estimated swerve duration
MIN_SWERVE_SPEED = 3.0         # m/s - below this, a vehicle just stops rather than ever swerving (see _maybe_swerve)
SWERVE_REEVALUATION_JUMP_THRESHOLD = 4.0   # metres - a belief distance jumping by more than this from one swerve evaluation to the next is treated as a different hazard requiring a fresh assessment, not a continuous approach to the same one (see _maybe_swerve)
ROUNDABOUT_TRIGGER_WINDOW = 2.0    # metres - how close counts as "arriving" at a roundabout, for the confusion check
ROUNDABOUT_RESET_DISTANCE = 50.0   # metres - once further than this past a roundabout, the confusion check re-arms for the vehicle's next lap
GPS_CONFUSION_MAGNITUDE = 15.0     # metres - how far an accepted belief must actually be from true position to cause a wrong exit
ROUNDABOUT_OCCUPANCY_RADIUS = 15.0   # metres either side of a roundabout's marked position that counts as "on" it

@dataclass
class TickResult:
    """One simulation tick's worth of data, per vehicle - this is what
    later gets fed into the outcome/severity model"""
    time: float
    vehicle_id: str
    position: tuple[float, float]
    speed: float
    fused_belief: FusedBelief
    lane: int = 0
    collision: bool = False   # True if this vehicle was in a collision this tick
    severity: str | None = None   # "slight" | "serious" | "fatal" - only set on the tick collision first becomes True, see severity.py for the sourcing behind this classification
    roundabout_confused: bool = False   # True on the tick this vehicle was judged to have taken an extra lap at a roundabout due to a corrupted GPS belief
    roundabout_excursion_remaining: float = 0.0   # metres of an extra roundabout lap still left to drive this tick, if any - see VehicleState.step()
    radar_reading: SensorReading | None = None   # the raw radar reading this tick, post-attack - may disagree with fused_belief if the active fusion policy doesn't trust radar (or trusts it and gets fooled)
    camera_reading: SensorReading | None = None   # the raw camera reading this tick, post-attack - same idea as radar_reading
    lidar_reading: SensorReading | None = None   # the raw LiDAR reading this tick
    ground_truth_kind: str | None = None   # what the nearest ground-truth candidate actually was this tick ("vehicle", "hazard:pedestrian_crossing", "traffic_light", "roundabout_giveway", ...), or None - see _ground_truth_readings

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

        # Driver diversity, hazard timing, spawner and
        # severity sampling are seeded here
        self.rng = random.Random(config.random_seed)
        self.vehicles: dict[str, VehicleState] = {}
        for vc in config.vehicles:
            reaction_time = vc.reaction_time
            if reaction_time is None:
                reaction_time = REACTION_TIME * (1 + self.rng.uniform(-vc.braking_variation, vc.braking_variation))
            max_deceleration = vc.max_deceleration
            if max_deceleration is None:
                max_deceleration = MAX_DECELERATION * (1 + self.rng.uniform(-vc.braking_variation, vc.braking_variation))
            self.vehicles[vc.vehicle_id] = VehicleState(
                vehicle_id=vc.vehicle_id,
                s=vc.start_distance,
                speed=vc.start_speed,
                cruise_speed=vc.start_speed,
                lane=vc.lane,
                direction=vc.direction,
                reaction_time=reaction_time,
                max_deceleration=max_deceleration,
            )

        self.fusion_policies: dict[str, FusionPolicy] = {
            vc.vehicle_id: FUSION_POLICIES[vc.fusion_policy]()
            for vc in config.vehicles
        }

        # Keep own lane and oncoming policies separate so that they do not interfere
        self.oncoming_fusion_policies: dict[str, FusionPolicy] = {
            vc.vehicle_id: FUSION_POLICIES[vc.fusion_policy]()
            for vc in config.vehicles
        }

        self.gps_policies: dict[str, GPSPolicy] = {
            vc.vehicle_id: GPS_POLICIES[vc.gps_policy]()
            for vc in config.vehicles
        }

        self._last_belief: dict[str, FusedBelief] = {}

        self._last_belief_speed: dict[str, float] = {}

        self._last_oncoming_belief: dict[str, FusedBelief] = {}

        self._last_ground_truth_kind: dict[str, str | None] = {}

        # attacks are stored as (target_vehicle_id, Attack instance) pairs
        self.attacks: list[tuple[str, Attack]] = []
        for a in config.attacks:
            attack_cls = ATTACK_TYPES[a["type"]]
            kwargs = {k: v for k, v in a.items() if k not in ("type", "target_vehicle")}
            self.attacks.append((a["target_vehicle"], attack_cls(**kwargs)))

        # hazards are ground truth, not attached to any specific vehicle.
        # A hazard's start_time left as None (see PedestrianCrossing in
        # hazards.py) is resolved to a random value here, using
        # the same seeded RNG as driver-diversity braking above - a
        # fixed random_seed reproduces both a scenario's driver diversity
        # and its random pedestrian timing together
        self.hazards = []
        for h in config.hazards:
            hazard_cls = HAZARD_TYPES[h["type"]]
            kwargs = {k: v for k, v in h.items() if k != "type"}
            hazard = hazard_cls(**kwargs)
            if getattr(hazard, "start_time", "not applicable") is None:
                latest_start = max(0.0, config.duration - hazard.duration)
                hazard.start_time = self.rng.uniform(0.0, latest_start)
            self.hazards.append(hazard)

        # Background traffic spawner - see SpawnerConfig in scenario.py.
        # None (the default) means no spawning at all
        self.spawner_config = config.spawner
        self._spawn_counter = 0
        self._spawned_vehicle_ids: set[str] = set()
        self._spawn_origin_s: dict[str, float] = {}
        if self.spawner_config is not None:
            light = self.traffic_lights[self.spawner_config.feature_id]
            self._next_spawn_time = light.cycle_length

    def _ahead_distance(self, vehicle: VehicleState, target_s: float) -> float:
        """Forward distance from vehicle to target_s, in the vehicle's own
        direction of travel.

        Normal traffic (direction=1) travels toward increasing s,
        so this is just Track.distance_ahead().
        Oncoming is swapped (travels towards decreasing s)"""
        if vehicle.direction == 1:
            return self.track.distance_ahead(vehicle.s, target_s)
        return self.track.distance_ahead(target_s, vehicle.s)

    def _ahead_distance_to_vehicle(self, vehicle: VehicleState, other: VehicleState) -> float | None:
        """Forward distance to another vehicle, or None if it isn't ahead.

        Same-direction traffic uses plain forward distance, less VEHICLE_LENGTH so
        the reported range is to the lead vehicle's rear.

        Opposite-direction traffic uses shortest-path proximity instead: two
        vehicles closing head-on don't stop being a hazard when their s-coordinates
        cross - this crossing is the collision point. A forward-only distance
        would flip them from "just ahead" to "almost a full lap ahead" at
        the wrong moment.
        """
        if other.direction == vehicle.direction:
            return max(0.0, self._ahead_distance(vehicle, other.s) - VEHICLE_LENGTH)

        return abs(self.track.signed_gap(vehicle.s, other.s))

    def _ground_truth_readings(self, vehicle_id: str) -> tuple[dict[SensorType, SensorReading], str | None]:
        """builds the clean readings for one vehicle, from the nearest thing ahead in its lane.

        Candidates are other vehicles in the same lane and hazards occupying it
        oncoming traffic is only ever an obstacle to avoid.

        Radar and camera report identical ground truth
        LiDAR sees the same candidate but only within its shorter range
        Both ranges scale by config.visibility

        Returns (readings, nearest_kind) - nearest_kind names what was detected,
        for observability - belief.source only ever names the sensor
        """
        vehicle = self.vehicles[vehicle_id]
        x, y, _heading = self.track.lane_position_at(vehicle.s, vehicle.lane)
        # Radar/camera/LiDAR readings always exist, even detecting nothing: a phantom
        # attack fabricates a detection by modifying a reading, so it needs an object
        # to act on where ground truth had none.
        readings: dict[SensorType, SensorReading] = {
            SensorType.GPS: SensorReading(SensorType.GPS, detected_position=(x, y)),
            SensorType.RADAR: SensorReading(SensorType.RADAR),
            SensorType.CAMERA: SensorReading(SensorType.CAMERA),
            SensorType.LIDAR: SensorReading(SensorType.LIDAR),
        }

        candidates: list[tuple[float, float, str]] = []  # (distance_ahead, closing_speed, kind)

        others = [v for v in self.vehicles.values()
                  if v.vehicle_id != vehicle_id and v.lane == vehicle.lane]
        vehicle_candidates: list[tuple[float, float, str]] = []
        for other in others:
            distance = self._ahead_distance_to_vehicle(vehicle, other)
            if distance is None:
                continue
            # closing speed: both vehicles speed is a magnitude, so their
            # closing rate accounts for whether each is moving toward or
            # away from the other along the shared s-axis
            own_rate = vehicle.speed * vehicle.direction
            other_rate = other.speed * other.direction
            vehicle_candidates.append((distance, own_rate - other_rate, "vehicle"))
        if vehicle_candidates:
            candidates.append(min(vehicle_candidates, key=lambda c: c[0]))

        for hazard in self.hazards:
            if hazard.lane_at(self.time) == vehicle.lane:
                feature = self.track.feature(hazard.feature_id)
                distance = self._ahead_distance(vehicle, feature.position)
                kind = f"hazard:{type(hazard).__name__.lower()}" if type(hazard).__name__ != "ObstacleInRoad" else "hazard:obstacle_in_road"
                kind = "hazard:pedestrian_crossing" if type(hazard).__name__ == "PedestrianCrossing" else kind
                candidates.append((distance, vehicle.speed, kind))  # stationary hazard -> closing speed = vehicle's own speed

        if vehicle.direction == 1:
            for feature_id, light in self.traffic_lights.items():
                if light.state_at(self.time) != "green":
                    feature = self.track.feature(feature_id)
                    stop_line = feature.position - STOP_LINE_OFFSET
                    distance = self._ahead_distance(vehicle, stop_line)
                    candidates.append((distance, vehicle.speed, "traffic_light"))  # stop line doesn't move -> closing speed = vehicle's own speed

            for feature in self.track.features_of_type("roundabout"):
                if self._vehicle_on_roundabout(vehicle, feature):
                    continue
                occupied = any(
                    other.vehicle_id != vehicle_id and self._vehicle_on_roundabout(other, feature)
                    for other in self.vehicles.values()
                )
                if occupied:
                    entry_point = feature.position - ROUNDABOUT_OCCUPANCY_RADIUS
                    distance = self._ahead_distance(vehicle, entry_point)
                    candidates.append((distance, vehicle.speed, "roundabout_giveway"))  # entry point doesn't move -> closing speed = vehicle's own speed

        effective_max_range = MAX_SENSOR_RANGE * self.config.visibility
        effective_lidar_range = LIDAR_MAX_SENSOR_RANGE * self.config.visibility

        if candidates:
            candidates = [c for c in candidates if c[0] <= effective_max_range]

        nearest_kind = None
        if candidates:
            nearest_distance, nearest_closing_speed, nearest_kind = min(candidates, key=lambda c: c[0])
            readings[SensorType.RADAR] = SensorReading(SensorType.RADAR, detected_distance=nearest_distance, detected_velocity=nearest_closing_speed)
            readings[SensorType.CAMERA] = SensorReading(SensorType.CAMERA, detected_distance=nearest_distance, detected_velocity=nearest_closing_speed)
            # LiDAR shares the same ground-truth nearest
            # candidate value as radar/camera but only within
            # its own range - beyond that it stays
            # at its baseline "nothing detected"
            if nearest_distance <= effective_lidar_range:
                readings[SensorType.LIDAR] = SensorReading(SensorType.LIDAR, detected_distance=nearest_distance, detected_velocity=nearest_closing_speed)

        return readings, nearest_kind

    def _oncoming_lane_readings(self, vehicle_id: str) -> dict[SensorType, SensorReading]:
        """Sensor readings for the lane OPPOSITE vehicle's current one,
        used specifically by _maybe_swerve to judge whether it's safe to
        swerve into it.
        """
        vehicle = self.vehicles[vehicle_id]
        opposite_lane = 1 - vehicle.lane

        readings: dict[SensorType, SensorReading] = {
            SensorType.RADAR: SensorReading(SensorType.RADAR),
            SensorType.CAMERA: SensorReading(SensorType.CAMERA),
            SensorType.LIDAR: SensorReading(SensorType.LIDAR),
            SensorType.GPS: SensorReading(SensorType.GPS),
        }

        others = [v for v in self.vehicles.values()
                  if v.vehicle_id != vehicle_id and v.lane == opposite_lane]
        candidates: list[tuple[float, float]] = []  # (distance, closing_speed)
        for other in others:
            distance = self._ahead_distance_to_vehicle(vehicle, other)
            if distance is None:
                continue
            own_rate = vehicle.speed * vehicle.direction
            other_rate = other.speed * other.direction
            candidates.append((distance, own_rate - other_rate))

        effective_max_range = MAX_SENSOR_RANGE * self.config.visibility
        effective_lidar_range = LIDAR_MAX_SENSOR_RANGE * self.config.visibility
        candidates = [c for c in candidates if c[0] <= effective_max_range]

        if candidates:
            nearest_distance, nearest_closing_speed = min(candidates, key=lambda c: c[0])
            readings[SensorType.RADAR] = SensorReading(SensorType.RADAR, detected_distance=nearest_distance, detected_velocity=nearest_closing_speed)
            readings[SensorType.CAMERA] = SensorReading(SensorType.CAMERA, detected_distance=nearest_distance, detected_velocity=nearest_closing_speed)
            if nearest_distance <= effective_lidar_range:
                readings[SensorType.LIDAR] = SensorReading(SensorType.LIDAR, detected_distance=nearest_distance, detected_velocity=nearest_closing_speed)

        return readings

    def _check_roundabout_confusion(self, vehicle: VehicleState, gps_reading: SensorReading, believed_xy: tuple[float, float]) -> bool:
        """Checks whether vehicle is arriving at any roundabout this tick,
        and if so, whether its GPS belief was corrupted

        If corrupted, the vehicle is judged to have taken
        the wrong exit: it must drive one extra lap of the roundabout
        before continuing on the main loop"""
        confused = False
        for feature in self.track.features_of_type("roundabout"):
            distance = self._ahead_distance(vehicle, feature.position)
            if feature.feature_id not in vehicle.roundabouts_resolved:
                if distance <= ROUNDABOUT_TRIGGER_WINDOW:
                    vehicle.roundabouts_resolved.add(feature.feature_id)
                    accepted_attacked_reading = gps_reading.is_attacked and believed_xy == gps_reading.detected_position
                    if accepted_attacked_reading:
                        true_x, true_y, _ = self.track.lane_position_at(vehicle.s, vehicle.lane)
                        bx, by = believed_xy
                        error = ((bx - true_x) ** 2 + (by - true_y) ** 2) ** 0.5
                        if error > GPS_CONFUSION_MAGNITUDE:
                            vehicle.roundabout_excursion_remaining += self.track.feature_circumference(feature.feature_id)
                            confused = True
            elif distance > ROUNDABOUT_RESET_DISTANCE:
                vehicle.roundabouts_resolved.discard(feature.feature_id)
        return confused

    def _vehicle_on_roundabout(self, vehicle: VehicleState, feature: TrackFeature) -> bool:
        return abs(self.track.signed_gap(vehicle.s, feature.position)) <= ROUNDABOUT_OCCUPANCY_RADIUS

    def _pedestrian_caution_multiplier(self, vehicle: VehicleState) -> float:
        """The highest PEDESTRIAN_CAUTION_MULTIPLIER among currently
        present pedestrian_crossing hazards within sensor range, ahead of
        vehicle in its own lane, or 1.0 (no extra caution) if none apply.
        Deliberately checked against ground truth, not the vehicle's
        fused belief - pedestrian crossings are never attacked in this
        project"""
        best = 1.0
        for hazard in self.hazards:
            if type(hazard).__name__ != "PedestrianCrossing" or hazard.lane_at(self.time) != vehicle.lane:
                continue
            feature = self.track.feature(hazard.feature_id)
            distance = self._ahead_distance(vehicle, feature.position)
            if distance <= MAX_SENSOR_RANGE * self.config.visibility:
                best = max(best, hazard.caution_multiplier)
        return best

    def _apply_attacks(self, vehicle_id: str, readings: dict[SensorType, SensorReading], only_general: bool = False) -> dict[SensorType, SensorReading]:
        """Applies every attack targeting vehicle_id to the given
        readings.

        only_general=True restricts this to Jam-type attacks
        only (RadarJam/CameraJam/LidarJam) - used for
        _oncoming_lane_readings, since a jammed sensor is
        unavailable in every direction.

        A spoof or phantom attack
        (RadarSpoof/LidarSpoof/CameraPhantom) fabricates a
        value representing something in the
        vehicle's own lane"""
        vehicle = self.vehicles[vehicle_id]
        for target_id, attack in self.attacks:
            if target_id != vehicle_id:
                continue
            if only_general and not isinstance(attack, Jam):
                continue
            attack.check_trigger(self.time, vehicle.s, self.track)
            sensor = attack.target_sensor
            if sensor in readings:
                readings[sensor] = attack.apply(readings[sensor], self.time)
        return readings

    def _maybe_swerve(self, vehicle: VehicleState) -> None:
        """Checks whether vehicle should begin or end a swerve manoeuvre
        this tick. A vehicle swerves into lane 1 when it is in lane 0,
        if it cannot stop safely in time for an obstacle in lane 0
        (braking alone - see can_stop_safely()), and lane 1 isn't itself
        occupied by an oncoming vehicle at roughly that position. It
        swerves back once it has travelled far enough past the obstacle's
        position
        """
        if vehicle.swerve_active:
            if vehicle.swerve_progress >= vehicle.swerve_return_progress:
                vehicle.lane = 0
                vehicle.swerve_active = False
                vehicle.swerve_progress = 0.0
                vehicle.swerve_return_progress = 0.0
            return

        if vehicle.lane != 0:
            return

        # safe_stopping_distance is small but not zero at low speed, so a vehicle
        # finishing an otherwise-successful stop can look "unable to stop safely".
        # Anything this slow can stop without swerving.
        if vehicle.speed < MIN_SWERVE_SPEED:
            return

        belief = self._last_belief.get(vehicle.vehicle_id)
        if belief is None or not belief.obstacle_present or belief.distance_to_obstacle is None:
            # not currently perceiving anything - clear so a future,
            # different hazard gets evaluated when it appears
            vehicle.swerve_evaluated = False
            vehicle.last_seen_obstacle_distance = None
            return

        ground_truth_kind = self._last_ground_truth_kind.get(vehicle.vehicle_id)
        if ground_truth_kind in ("traffic_light", "roundabout_giveway"):
            vehicle.swerve_evaluated = False
            vehicle.last_seen_obstacle_distance = None
            return

        obstacle_distance = belief.distance_to_obstacle

        if (vehicle.last_seen_obstacle_distance is not None
                and abs(obstacle_distance - vehicle.last_seen_obstacle_distance) > SWERVE_REEVALUATION_JUMP_THRESHOLD):
            vehicle.swerve_evaluated = False

        vehicle.last_seen_obstacle_distance = obstacle_distance

        if vehicle.swerve_evaluated:
            return

        snapshot_speed = self._last_belief_speed.get(vehicle.vehicle_id, vehicle.speed)
        if can_stop_safely(snapshot_speed, obstacle_distance, vehicle.reaction_time, vehicle.max_deceleration, self._pedestrian_caution_multiplier(vehicle)):
            vehicle.swerve_evaluated = True
            return  # braking alone is enough - no need to swerve

        # "Clear" is judged on time, not range, at 150m sensor range almost anything
        # in the oncoming lane would reject doing a swerve as it sees a hazard
        estimated_swerve_time = (obstacle_distance + SWERVE_RETURN_BUFFER) / vehicle.speed
        oncoming_belief = self._last_oncoming_belief.get(vehicle.vehicle_id)
        if oncoming_belief is None:
            return
        if oncoming_belief.obstacle_present and oncoming_belief.distance_to_obstacle is not None:
            closing_speed = oncoming_belief.detected_velocity
            if closing_speed is not None and closing_speed > 0:
                time_to_close = oncoming_belief.distance_to_obstacle / closing_speed
                if time_to_close < estimated_swerve_time * SWERVE_TIME_SAFETY_MARGIN:
                    return  # not enough time before the believed oncoming vehicle would reach this point

        vehicle.lane = 1
        vehicle.swerve_active = True
        vehicle.swerve_progress = 0.0
        vehicle.swerve_return_progress = obstacle_distance + SWERVE_RETURN_BUFFER

    def _detect_collisions(self) -> dict[str, str]:
        """Returns a dict mapping each vehicle_id involved in a collision
        this tick to its severity category ("slight"|"serious"|"fatal"),
        classified via severity.py using whichever impact/closing speed
        is appropriate for the kind of collision it was"""
        involved: dict[str, str] = {}
        vehicles = [v for v in self.vehicles.values() if not v.crashed]

        def _record(vehicle_id: str, sev: str) -> None:
            if vehicle_id not in involved or SEVERITY_RANK[sev] > SEVERITY_RANK[involved[vehicle_id]]:
                involved[vehicle_id] = sev

        all_vehicles = list(self.vehicles.values())
        for i in range(len(all_vehicles)):
            for j in range(i + 1, len(all_vehicles)):
                a, b = all_vehicles[i], all_vehicles[j]
                if a.crashed and b.crashed:
                    continue
                if a.lane != b.lane:
                    continue
                ax, ay, _ = self.track.lane_position_at(a.s, a.lane)
                bx, by, _ = self.track.lane_position_at(b.s, b.lane)
                if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= COLLISION_DISTANCE:
                    closing_speed = abs(a.speed * a.direction - b.speed * b.direction)
                    sev = severity.classify_severity(severity.vehicle_fatality_risk(closing_speed), self.rng)
                    if not a.crashed:
                        _record(a.vehicle_id, sev)
                    if not b.crashed:
                        _record(b.vehicle_id, sev)

        def _occluded(v: VehicleState, target_s: float) -> bool:
            distance_to_target = (target_s - v.s) * v.direction
            if distance_to_target < 0:
                return False
            for other in self.vehicles.values():
                if other.vehicle_id == v.vehicle_id or other.lane != v.lane or other.direction != v.direction:
                    continue
                other_rear = max(0.0, (other.s - v.s) * v.direction - VEHICLE_LENGTH)
                if other_rear < distance_to_target:
                    return True
            return False

        for hazard in self.hazards:
            if type(hazard).__name__ != "ObstacleInRoad" or not hazard.is_present(self.time):
                continue
            feature = self.track.feature(hazard.feature_id)
            ox, oy, _ = self.track.lane_position_at(feature.position, hazard.lane)
            for v in vehicles:
                if v.lane != hazard.lane:
                    continue
                if _occluded(v, feature.position):
                    continue
                vx, vy, _ = self.track.lane_position_at(v.s, v.lane)
                if ((vx - ox) ** 2 + (vy - oy) ** 2) ** 0.5 <= COLLISION_DISTANCE:
                    sev = severity.classify_severity(severity.vehicle_fatality_risk(v.speed), self.rng)
                    _record(v.vehicle_id, sev)

        for hazard in self.hazards:
            if type(hazard).__name__ != "PedestrianCrossing":
                continue
            feature = self.track.feature(hazard.feature_id)
            for v in vehicles:
                if hazard.lane_at(self.time) != v.lane:
                    continue
                if _occluded(v, feature.position):
                    continue
                vx, vy, _ = self.track.lane_position_at(v.s, v.lane)
                px, py, _ = self.track.lane_position_at(feature.position, v.lane)
                if ((vx - px) ** 2 + (vy - py) ** 2) ** 0.5 <= PEDESTRIAN_COLLISION_DISTANCE:
                    sev = severity.classify_severity(severity.pedestrian_fatality_risk(v.speed), self.rng)
                    hazard.struck = True
                    _record(v.vehicle_id, sev)

        return involved

    def _maybe_spawn_vehicle(self) -> None:
        """Spawns a new background vehicle at the spawner's junction each
        time the traffic light there completes a full cycle, up to
        SpawnerConfig.max_concurrent - if already at capacity, this
        cycle's spawn is skipped"""
        cfg = self.spawner_config
        if self.time < self._next_spawn_time:
            return
        self._next_spawn_time += self.traffic_lights[cfg.feature_id].cycle_length

        if len(self._spawned_vehicle_ids) >= cfg.max_concurrent:
            return

        feature = self.track.feature(cfg.feature_id)
        lane = 0 if self.rng.random() < 0.5 else 1
        direction = 1 if lane == 0 else -1

        reaction_time = REACTION_TIME * (1 + self.rng.uniform(-cfg.braking_variation, cfg.braking_variation))
        max_deceleration = MAX_DECELERATION * (1 + self.rng.uniform(-cfg.braking_variation, cfg.braking_variation))

        self._spawn_counter += 1
        vehicle_id = f"spawned_{self._spawn_counter}"
        speed = mph_to_ms(cfg.speed_mph)

        self.vehicles[vehicle_id] = VehicleState(
            vehicle_id=vehicle_id, s=feature.position, speed=speed, cruise_speed=speed,
            lane=lane, direction=direction, reaction_time=reaction_time, max_deceleration=max_deceleration,
        )
        self.fusion_policies[vehicle_id] = FUSION_POLICIES[cfg.fusion_policy]()
        self.oncoming_fusion_policies[vehicle_id] = FUSION_POLICIES[cfg.fusion_policy]()
        self.gps_policies[vehicle_id] = GPS_POLICIES["naive"]()
        self._spawned_vehicle_ids.add(vehicle_id)
        self._spawn_origin_s[vehicle_id] = feature.position

    def _despawn_completed_laps(self) -> None:
        """Removes any spawned vehicle that has travelled one full lap
        since it entered"""
        for vehicle_id in list(self._spawned_vehicle_ids):
            vehicle = self.vehicles[vehicle_id]
            origin = self._spawn_origin_s[vehicle_id]
            if abs(vehicle.s - origin) >= self.track.total_length:
                del self.vehicles[vehicle_id]
                del self.fusion_policies[vehicle_id]
                del self.oncoming_fusion_policies[vehicle_id]
                del self.gps_policies[vehicle_id]
                del self._spawn_origin_s[vehicle_id]
                self._last_belief.pop(vehicle_id, None)
                self._last_belief_speed.pop(vehicle_id, None)
                self._last_oncoming_belief.pop(vehicle_id, None)
                self._last_ground_truth_kind.pop(vehicle_id, None)
                self._spawned_vehicle_ids.discard(vehicle_id)

    def step(self, dt: float) -> list[TickResult]:
        results = []
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle.crashed:
                # Crashed vehicles are disabled: no sensing, fusion or control. Speed
                # already went to zero at the moment of impact
                vehicle.acceleration = 0.0
                x, y, _heading = self.track.lane_position_at(vehicle.s, vehicle.lane)
                belief = FusedBelief(None, False, "crashed")
                results.append(TickResult(self.time, vehicle_id, (x, y), vehicle.speed, belief, lane=vehicle.lane, collision=True, severity=vehicle.severity))
                continue

            self._maybe_swerve(vehicle)

            readings, ground_truth_kind = self._ground_truth_readings(vehicle_id)
            readings = self._apply_attacks(vehicle_id, readings)
            belief = self.fusion_policies[vehicle_id].fuse(readings)
            self._last_belief[vehicle_id] = belief
            self._last_belief_speed[vehicle_id] = vehicle.speed
            self._last_ground_truth_kind[vehicle_id] = ground_truth_kind

            oncoming_readings = self._oncoming_lane_readings(vehicle_id)
            oncoming_readings = self._apply_attacks(vehicle_id, oncoming_readings, only_general=True)
            oncoming_belief = self.oncoming_fusion_policies[vehicle_id].fuse(oncoming_readings)
            self._last_oncoming_belief[vehicle_id] = oncoming_belief

            gps_reading = readings[SensorType.GPS]
            believed_xy = self.gps_policies[vehicle_id].resolve(gps_reading, vehicle, dt)
            roundabout_confused = self._check_roundabout_confusion(vehicle, gps_reading, believed_xy)

            if belief.obstacle_present and belief.distance_to_obstacle is not None:
                caution = self._pedestrian_caution_multiplier(vehicle)
                decel = required_deceleration(vehicle.speed, belief.distance_to_obstacle, vehicle.reaction_time, vehicle.max_deceleration, caution)
            else:
                caution = 1.0
                decel = 0.0

            safe_to_resume = (
                not belief.obstacle_present
                or can_stop_safely(vehicle.cruise_speed, belief.distance_to_obstacle, vehicle.reaction_time, vehicle.max_deceleration, caution)
            )

            if decel > 0:
                vehicle.acceleration = -decel
            elif safe_to_resume and vehicle.speed < vehicle.cruise_speed:
                # Resuming is gated on whether stopping would still be safe once back at cruise speed
                vehicle.acceleration = min(RESUME_ACCELERATION, (vehicle.cruise_speed - vehicle.speed) / dt)
            elif not safe_to_resume and vehicle.speed > 0:
                # Not required to brake, not safe to resume. Without this branch a vehicle
                # coasts forever at whatever residual speed it had
                vehicle.acceleration = -min(RESUME_ACCELERATION, vehicle.speed / dt)
            else:
                vehicle.acceleration = 0.0

            x, y, _heading = self.track.lane_position_at(vehicle.s, vehicle.lane)
            results.append(TickResult(self.time, vehicle_id, (x, y), vehicle.speed, belief, lane=vehicle.lane, roundabout_confused=roundabout_confused, roundabout_excursion_remaining=vehicle.roundabout_excursion_remaining, radar_reading=readings[SensorType.RADAR], camera_reading=readings[SensorType.CAMERA], lidar_reading=readings[SensorType.LIDAR], ground_truth_kind=ground_truth_kind))

        collided = self._detect_collisions()
        for result in results:
            if result.vehicle_id in collided:
                result.collision = True
                result.severity = collided[result.vehicle_id]
                vehicle = self.vehicles[result.vehicle_id]
                if not vehicle.crashed:
                    # first tick of this collision: the impact itself
                    # stops the vehicle immediately, not gradually
                    vehicle.crashed = True
                    vehicle.severity = collided[result.vehicle_id]
                    vehicle.speed = 0.0
                    vehicle.acceleration = 0.0

        for vehicle in self.vehicles.values():
            vehicle.step(dt)
        self.time += dt

        if self.spawner_config is not None:
            self._maybe_spawn_vehicle()
            self._despawn_completed_laps()

        return results

    def run(self) -> list[TickResult]:
        """runs full scenario and returns every tick's results"""
        log: list[TickResult] = []
        steps = int(self.config.duration / self.config.timestep)
        for _ in range(steps):
            log.extend(self.step(self.config.timestep))
        return log