"""Attack models.

Each attack takes a clean SensorReading (ground truth, as the sensor would
report with no interference) and returns a manipulated SensorReading -
what the victim vehicle's sensor actually reports once attacked.

An attack needs a trigger - Three kinds are supported:
  - start_time: fires at a fixed point on the simulation clock.
  - trigger_before_feature + trigger_distance: fires once the target
    vehicle comes within trigger_distance metres of the named track
    feature (e.g. "junction_1" - see Track in track.py).
  - trigger_after_feature + trigger_distance: fires once the target
    vehicle has travelled trigger_distance metres *past* the named
    feature - e.g. "attack the lead car 50m after it clears the
    junction," to catch a vehicle just as it's pulling away.

trigger_after_feature does NOT use Track.distance_ahead
(the same helper trigger_before_feature uses), even
though it looks like the natural choice. distance_ahead wraps around
the whole loop, which means a vehicle that *hasn't reached the feature yet*
would appear to be some large distance "past" it (the remainder of the lap),
incorrectly arming the attack before the vehicle ever gets there.
Since VehicleState.s accumulates monotonically and is never itself wrapped
(only Track wraps it, for geometry purposes), subtraction is both simpler and
correct for a single-lap scenario: negative before the feature, growing
positive only after it's been passed.

Once a trigger condition is met, the attack "arms" (records the time it
fired) and then runs for 'duration' seconds from that point. It only
arms once per attack instance.
"""
from dataclasses import replace
from .sensors import SensorReading, SensorType

class Attack:
    """base class for all attacks - subclasses override 'apply'"""

    target_sensor: SensorType

    def __init__(self, duration: float, start_time: float | None = None,
                 trigger_before_feature: str | None = None,
                 trigger_after_feature: str | None = None,
                 trigger_distance: float | None = None):
        trigger_count = sum(x is not None for x in (start_time, trigger_before_feature, trigger_after_feature))
        if trigger_count == 0:
            raise ValueError(
                "an attack needs a trigger: start_time, trigger_before_feature, or trigger_after_feature"
            )
        if trigger_count > 1:
            raise ValueError("an attack should have one trigger, not several")
        if (trigger_before_feature is not None or trigger_after_feature is not None) and trigger_distance is None:
            raise ValueError("trigger_before_feature/trigger_after_feature requires trigger_distance to also be set")

        self.duration = duration
        self.start_time = start_time
        self.trigger_before_feature = trigger_before_feature
        self.trigger_after_feature = trigger_after_feature
        self.trigger_distance = trigger_distance
        self._armed_time: float | None = None

    def check_trigger(self, t: float, vehicle_s: float | None, track) -> None:
        """called once per tick by engine, before apply(). arms the
        attack (records trigger time) the first time its condition is
        met. a no-op after that - an attack only fires once"""
        if self._armed_time is not None:
            return

        if self.start_time is not None:
            if t >= self.start_time:
                self._armed_time = self.start_time
            return

        if vehicle_s is None or track is None:
            return

        if self.trigger_before_feature is not None:
            feature = track.feature(self.trigger_before_feature)
            distance_to_feature = track.distance_ahead(vehicle_s, feature.position)
            if distance_to_feature <= self.trigger_distance:
                self._armed_time = t
        elif self.trigger_after_feature is not None:
            feature = track.feature(self.trigger_after_feature)
            distance_since_feature = vehicle_s - feature.position  # deliberately unwrapped - see module docstring
            if distance_since_feature >= self.trigger_distance:
                self._armed_time = t

    def is_active(self, t: float) -> bool:
        if self._armed_time is None:
            return False
        return self._armed_time <= t < self._armed_time + self.duration

    def apply(self, reading: SensorReading, t: float) -> SensorReading:
        raise NotImplementedError

class RadarSpoof(Attack):
    """radar spoof: fabricates a phantom distance/velocity pair"""

    target_sensor = SensorType.RADAR

    def __init__(self, spoofed_distance: float, spoofed_velocity: float, **kwargs):
        super().__init__(**kwargs)
        self.spoofed_distance = spoofed_distance
        self.spoofed_velocity = spoofed_velocity

    def apply(self, reading: SensorReading, t: float) -> SensorReading:
        if not self.is_active(t):
            return reading
        return replace(
            reading,
            detected_distance=self.spoofed_distance,
            detected_velocity=self.spoofed_velocity,
            confidence=1.0,
            is_attacked=True,
        )

class LidarSpoof(Attack):
    """LiDAR laser relay/spoofing attack: fabricates a phantom
    distance/velocity pair, structurally identical to RadarSpoof, but
    grounded in the LiDAR-specific relay/spoofing literature discussed in
    the literature review rather than the RF-domain radar spoofing of
    Komissarov & Wool. Kept as a separate class (rather than reusing
    RadarSpoof against a different target_sensor) so the attack's
    identity in logs/scenarios reflects which sensor and which real-world
    attack technique it represents."""

    target_sensor = SensorType.LIDAR

    def __init__(self, spoofed_distance: float, spoofed_velocity: float, **kwargs):
        super().__init__(**kwargs)
        self.spoofed_distance = spoofed_distance
        self.spoofed_velocity = spoofed_velocity

    def apply(self, reading: SensorReading, t: float) -> SensorReading:
        if not self.is_active(t):
            return reading
        return replace(
            reading,
            detected_distance=self.spoofed_distance,
            detected_velocity=self.spoofed_velocity,
            confidence=1.0,
            is_attacked=True,
        )

class CameraPhantom(Attack):
    """split-second phantom object attack injects a
    depthless false obstacle for a short duration"""

    target_sensor = SensorType.CAMERA

    def __init__(self, phantom_distance: float, **kwargs):
        super().__init__(**kwargs)
        self.phantom_distance = phantom_distance

    def apply(self, reading: SensorReading, t: float) -> SensorReading:
        if not self.is_active(t):
            return reading
        return replace(
            reading,
            detected_distance=self.phantom_distance,
            confidence=1.0,
            is_attacked=True,
        )

class GPSSpoof(Attack):
    """shifts reported GPS position by a fixed offset for attacks
    duration - to mislead a vehicles route-following logic"""

    target_sensor = SensorType.GPS

    def __init__(self, offset: tuple[float, float], **kwargs):
        super().__init__(**kwargs)
        self.offset = offset

    def apply(self, reading: SensorReading, t: float) -> SensorReading:
        if not self.is_active(t) or reading.detected_position is None:
            return reading
        x, y = reading.detected_position
        dx, dy = self.offset
        return replace(
            reading,
            detected_position=(x + dx, y + dy),
            confidence=1.0,
            is_attacked=True,
        )

class Jam(Attack):
    """Base class for sensor jamming attacks: forces the target sensor to
    report nothing detected for the attack's duration, rather than
    fabricating a plausible-but-wrong value (see RadarSpoof/CameraPhantom/
    LidarSpoof/GPSSpoof for that). This models the distinction drawn in
    the literature review (Li et al.): jamming is a dropout - easy to
    detect, since a sensor visibly stops responding - unlike spoofing,
    whose entire danger is looking plausible. Concrete subclasses below
    just set target_sensor. The jamming behaviour itself is identical
    regardless of which sensor is jammed."""

    def apply(self, reading: SensorReading, t: float) -> SensorReading:
        if not self.is_active(t):
            return reading
        return replace(
            reading,
            detected_distance=None,
            detected_velocity=None,
            confidence=0.0,
            is_attacked=True,
        )

class RadarJam(Jam):
    target_sensor = SensorType.RADAR

class CameraJam(Jam):
    target_sensor = SensorType.CAMERA

class LidarJam(Jam):
    target_sensor = SensorType.LIDAR

ATTACK_TYPES = {
    "radar_spoof": RadarSpoof,
    "lidar_spoof": LidarSpoof,
    "camera_phantom": CameraPhantom,
    "gps_spoof": GPSSpoof,
    "radar_jam": RadarJam,
    "camera_jam": CameraJam,
    "lidar_jam": LidarJam,
}