"""attack models

each attack takes a clean SensorReading (ground truth - as the sensor would
report with no interference) and returns a manipulated SensorReading -
what the victim vehicle's sensor actually reports once attacked.

an attack needs a trigger - two kinds are supported:
  - start_time: fires at a fixed point on the simulation clock
  - trigger_before_junction / trigger_before_roundabout: fires once the
    target vehicle comes within that many metres of given track
    feature, regardless of what time that happens to be.

once a trigger condition is met, attack "arms" (records time it
fired) and then runs for 'duration' seconds from that point.
it only arms once per attack instance
"""
from dataclasses import replace
from .sensors import SensorReading, SensorType

class Attack:
    """base class for all attacks - subclasses override 'apply'"""

    target_sensor: SensorType

    def __init__(self, duration: float, start_time: float | None = None,
                 trigger_before_junction: float | None = None,
                 trigger_before_roundabout: float | None = None):
        if start_time is None and trigger_before_junction is None and trigger_before_roundabout is None:
            raise ValueError(
                "an attack needs a trigger: start_time, trigger_before_junction, "
                "or trigger_before_roundabout"
            )
        self.duration = duration
        self.start_time = start_time
        self.trigger_before_junction = trigger_before_junction
        self.trigger_before_roundabout = trigger_before_roundabout
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

        if self.trigger_before_junction is not None:
            distance_to_feature = track.distance_ahead(vehicle_s, track.junction_at)
            if distance_to_feature <= self.trigger_before_junction:
                self._armed_time = t
        elif self.trigger_before_roundabout is not None:
            distance_to_feature = track.distance_ahead(vehicle_s, track.roundabout_at)
            if distance_to_feature <= self.trigger_before_roundabout:
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

ATTACK_TYPES = {
    "radar_spoof": RadarSpoof,
    "camera_phantom": CameraPhantom,
    "gps_spoof": GPSSpoof,
}