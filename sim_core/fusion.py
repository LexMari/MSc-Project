"""Fusion policies

Given a set of (possibly attacked) sensor readings, fusion policy decides
the vehicle's single arbitrated belief about the world
"""
from dataclasses import dataclass
from .sensors import SensorReading, SensorType

@dataclass
class FusedBelief:
    """A vehicle's arbitrated understanding of what is ahead."""

    distance_to_obstacle: float | None
    obstacle_present: bool
    source: str  # which sensor or logic produced this
    detected_velocity: float | None = None   # fused closing speed, m/s

class FusionPolicy:
    """base class for fusion policies. subclasses override 'fuse'"""

    name = "base"

    def fuse(self, readings: dict[SensorType, SensorReading]) -> FusedBelief:
        raise NotImplementedError

class CameraPriorityFusion(FusionPolicy):
    """Trusts the camera alone - other sensors are never consulted

    This is deliberately vulnerable, but is also used by Tesla (see dissertation literature review)"""

    name = "camera_priority"

    def fuse(self, readings: dict[SensorType, SensorReading]) -> FusedBelief:
        camera = readings.get(SensorType.CAMERA)
        if camera and camera.detected_distance is not None:
            return FusedBelief(camera.detected_distance, True, "camera", detected_velocity=camera.detected_velocity)
        return FusedBelief(None, False, "none")

class MajorityVoteFusion(FusionPolicy):
    """only trusts an obstacle report if at least two sensors report a
    similar distance within 'tolerance' metres of each other"""

    name = "majority_vote"

    def __init__(self, tolerance: float = 3.0):
        self.tolerance = tolerance

    def fuse(self, readings: dict[SensorType, SensorReading]) -> FusedBelief:
        # Sorted by distance so agreement is checked between adjacent readings only
        valid = sorted(
            (r for r in readings.values() if r.detected_distance is not None),
            key=lambda r: r.detected_distance,
        )
        if len(valid) < 2:
            return FusedBelief(None, False, "insufficient_sensors")
        for a, b in zip(valid, valid[1:]):
            if abs(a.detected_distance - b.detected_distance) <= self.tolerance:
                agreed_distance = (a.detected_distance + b.detected_distance) / 2
                agreed_velocity = None
                if a.detected_velocity is not None and b.detected_velocity is not None:
                    agreed_velocity = (a.detected_velocity + b.detected_velocity) / 2
                return FusedBelief(agreed_distance, True, "majority", detected_velocity=agreed_velocity)
        return FusedBelief(None, False, "no_agreement")

class ConfidenceWeightedFusion(FusionPolicy):
    """blends every sensor's reported distance, weighted by its confidence

    There is no rejection here so a spoofed reading is included in the average which
    always shifts the belief."""

    name = "confidence_weighted"

    def fuse(self, readings: dict[SensorType, SensorReading]) -> FusedBelief:
        weighted_sum, weight_total = 0.0, 0.0
        velocity_weighted_sum, velocity_weight_total = 0.0, 0.0
        for r in readings.values():
            if r.detected_distance is not None:
                weighted_sum += r.detected_distance * r.confidence
                weight_total += r.confidence
                if r.detected_velocity is not None:
                    velocity_weighted_sum += r.detected_velocity * r.confidence
                    velocity_weight_total += r.confidence
        if weight_total == 0:
            return FusedBelief(None, False, "no_data")
        fused_velocity = velocity_weighted_sum / velocity_weight_total if velocity_weight_total else None
        return FusedBelief(weighted_sum / weight_total, True, "weighted", detected_velocity=fused_velocity)

class PlausibilityFilteredFusion(FusionPolicy):
    """Filters out sensor readings that jump implausibly compared with
    what that same sensor most recently reported, then hands whatever
    survives to MajorityVoteFusion's agreement logic

    Every other policy trusts a reading's magnitude (agreement,
    confidence) but none check whether a reading is physically plausible
    given what the vehicle just saw - which is why an attack reporting
    an implausible sudden jump (radar_spoof_masking jumping 58m -> 200m
    in one tick) can succeed against them. Same principle already used
    for GPS in navigation.py's PlausibilityCheckedGPSPolicy, applied
    here to sensor fusion rather than as an attack-specific patch.

    Design decisions:
    - Plausibility is bounded by a fixed, ceiling
      (MAX_PLAUSIBLE_CLOSING_SPEED). An attacker who fabricates a
      distance could equally fabricate a "justifying" velocity to match
      it, so trusting a reading's own explanation for its own suspicious
      jump would defeat the point of the check.
    - "Last known good" only updates when a reading is accepted, never
      when rejected - mirrors the GPS policy's "hold last known good
      position" pattern. A sustained attack stays flagged for its entire
      duration rather than just its first tick, since comparing against
      the raw previous reading instead would let a spoofed value that
      holds steady after its initial jump stop being flagged after one
      tick.
    - A sudden absence (a sensor that had a real reading, now reports
      None) is treated as consistent with jamming and excluded, with no
      "last known good" value left to compare against.

    Known limitations:
    - A reading's first appearance (no previous reading to compare
      against) is always accepted. A camera_phantom-style attack, which
      fabricates a brand-new close detection rather than corrupting an
      existing tracked one, is not caught by this policy.
    - A small spoof within sensor noise tolerance is indistinguishable
      from ordinary sensor imprecision.
    - A gradually-ramped attack (moving the spoofed value a small,
      plausible amount each tick) would evade this check, since each
      tick-to-tick change stays within tolerance even though the
      cumulative drift does not.
    - Ground truth only reports the single nearest candidate each tick
      with no persistent object identity across ticks (see
      _ground_truth_readings in engine.py). A legitimate change in which
      real object is nearest could be misread as an implausible jump.
    """

    name = "plausibility_filtered"

    # Fixed ceiling on how far one candidate's distance can plausibly change per tick
    MAX_PLAUSIBLE_CLOSING_SPEED = 45.0   # m/s, ~100mph

    # Per-sensor slack on top of the physical ceiling (LiDAR tightest, radar
    # noisiest), how much leeway is accepted normally
    NOISE_TOLERANCE = {
        SensorType.LIDAR: 1.0,
        SensorType.CAMERA: 2.0,
        SensorType.RADAR: 3.0,
    }

    def __init__(self, timestep: float = 0.1, tolerance: float = 3.0):
        self.timestep = timestep
        self.tolerance = tolerance
        self._last_trusted: dict[SensorType, float | None] = {}
        self._ever_corroborated: dict[SensorType, bool] = {}
        self._majority = MajorityVoteFusion(tolerance=tolerance)

    def fuse(self, readings: dict[SensorType, SensorReading]) -> FusedBelief:
        max_delta = self.MAX_PLAUSIBLE_CLOSING_SPEED * self.timestep
        filtered: dict[SensorType, SensorReading] = {}

        for sensor_type in (SensorType.RADAR, SensorType.CAMERA, SensorType.LIDAR):
            reading = readings.get(sensor_type)
            if reading is None:
                continue
            current = reading.detected_distance
            last = self._last_trusted.get(sensor_type)

            if current is None:
                # sudden absence - consistent with jamming, exclude
                filtered[sensor_type] = SensorReading(sensor_type)
                self._last_trusted[sensor_type] = None
                continue

            if last is None:
                # first appearance - accepted
                filtered[sensor_type] = reading
                self._last_trusted[sensor_type] = current
                continue

            tolerance = self.NOISE_TOLERANCE.get(sensor_type, 3.0)
            if abs(current - last) <= max_delta + tolerance:
                filtered[sensor_type] = reading
                self._last_trusted[sensor_type] = current
            else:
                # implausible jump - excluded, and last_trusted is
                # deliberately not updated, so the last-known value
                # keeps being compared against for as long as the
                # attack persists
                filtered[sensor_type] = SensorReading(sensor_type)

        # any two currently-accepted readings that
        # agree with each other flags both sensors as having a
        # track record of being corroborated
        accepted = {st: r for st, r in filtered.items() if r.detected_distance is not None}
        for st_a, r_a in accepted.items():
            for st_b, r_b in accepted.items():
                if st_a != st_b and abs(r_a.detected_distance - r_b.detected_distance) <= self.tolerance:
                    self._ever_corroborated[st_a] = True
                    self._ever_corroborated[st_b] = True

        established_survivors = {
            st: r for st, r in accepted.items()
            if self._ever_corroborated.get(st, False)
        }
        if len(accepted) == 1 and len(established_survivors) == 1:
            # if one sensor survived filtering and has a track
            # record of corroboration - trust it rather than
            # handing it to MajorityVoteFusion, which would discard it
            # for lack of a second agreeing sensor this tick
            sole = next(iter(established_survivors.values()))
            return FusedBelief(sole.detected_distance, True, "plausibility:single_survivor", detected_velocity=sole.detected_velocity)

        belief = self._majority.fuse(filtered)
        return FusedBelief(belief.distance_to_obstacle, belief.obstacle_present, f"plausibility:{belief.source}", detected_velocity=belief.detected_velocity)

FUSION_POLICIES = {
    "camera_priority": CameraPriorityFusion,
    "majority_vote": MajorityVoteFusion,
    "confidence_weighted": ConfidenceWeightedFusion,
    "plausibility_filtered": PlausibilityFilteredFusion,
}
