"""Attack models: corrupt what a sensor reports

Each attack takes a clean SensorReading - ground truth, as the sensor would
report with no interference - and returns the manipulated one the victim
sees.

Every attack needs one trigger:
  start_time                                  fires at a fixed simulation time
  trigger_before_feature + trigger_distance   fires within N metres of a feature
  trigger_after_feature  + trigger_distance   fires N metres past a feature

Once triggered, an attack arms and runs for 'duration' seconds. It arms once
per instance and never re-fires
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
        # Only one trigger: several would make the arming time ambiguous, none
        # would mean the attack never happens
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
        """Arm the attack if its condition is met. Called once per tick before apply()"""
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
            distance_since_feature = vehicle_s - feature.position
            if distance_since_feature >= self.trigger_distance:
                self._armed_time = t

    def is_active(self, t: float) -> bool:
        if self._armed_time is None:
            return False
        return self._armed_time <= t < self._armed_time + self.duration

    def apply(self, reading: SensorReading, t: float) -> SensorReading:
        raise NotImplementedError

class RadarSpoof(Attack):
    """Fabricates a false distance/velocity pair on radar.

    RF-domain spoofing per Komissarov & Wool. Overwrites confidence to 1.0 - a
    successful spoof presents as a perfectly healthy reading, which is what makes
    confidence_weighted fusion vulnerable to it"""

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
    """Fabricates a false distance/velocity pair on LiDAR.

    Identical to RadarSpoof but kept separate so that they can be
    identified separately
    """

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
    """A split-second camera phantom attack injects
    an obstacle in range of a sensor for a short duration"""

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
    """Forces the target sensor to report nothing for the attack's duration.

    Models the jamming/spoofing distinction from Li et al.: a dropout is easy to
    detect because the sensor stops responding

    Confidence drops to 0.0 accordingly
    Subclasses only set target_sensor as behaviour is identical across sensors
    """

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