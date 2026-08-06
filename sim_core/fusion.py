"""Fusion policies

Given a set of (possibly attacked) sensor readings, fusion policy decides
the vehicle's single arbitrated belief about the world
"""
from dataclasses import dataclass
from .sensors import SensorReading, SensorType

@dataclass
class FusedBelief:
    """The vehicle's arbitrated understanding of world after fusion"""

    distance_to_obstacle: float | None
    obstacle_present: bool
    source: str  # which sensor(s)/logic the fused belief came from - useful for later analysis/plots

class FusionPolicy:
    """base class for fusion policies. subclasses override 'fuse'"""

    name = "base"

    def fuse(self, readings: dict[SensorType, SensorReading]) -> FusedBelief:
        raise NotImplementedError

class CameraPriorityFusion(FusionPolicy):
    """trusts camera over other sensors whenever they disagree"""

    name = "camera_priority"

    def fuse(self, readings: dict[SensorType, SensorReading]) -> FusedBelief:
        camera = readings.get(SensorType.CAMERA)
        if camera and camera.detected_distance is not None:
            return FusedBelief(camera.detected_distance, True, "camera")
        return FusedBelief(None, False, "none")

class MajorityVoteFusion(FusionPolicy):
    """only trusts an obstacle report if at least two sensors report a
    similar distance within 'tolerance' metres of each other"""

    name = "majority_vote"

    def __init__(self, tolerance: float = 3.0):
        self.tolerance = tolerance

    def fuse(self, readings: dict[SensorType, SensorReading]) -> FusedBelief:
        distances = sorted(
            r.detected_distance for r in readings.values()
            if r.detected_distance is not None
        )
        if len(distances) < 2:
            return FusedBelief(None, False, "insufficient_sensors")
        for i in range(len(distances) - 1):
            if abs(distances[i] - distances[i + 1]) <= self.tolerance:
                agreed = (distances[i] + distances[i + 1]) / 2
                return FusedBelief(agreed, True, "majority")
        return FusedBelief(None, False, "no_agreement")

class ConfidenceWeightedFusion(FusionPolicy):
    """blends every sensor's reported distance, weighted by its confidence"""

    name = "confidence_weighted"

    def fuse(self, readings: dict[SensorType, SensorReading]) -> FusedBelief:
        weighted_sum, weight_total = 0.0, 0.0
        for r in readings.values():
            if r.detected_distance is not None:
                weighted_sum += r.detected_distance * r.confidence
                weight_total += r.confidence
        if weight_total == 0:
            return FusedBelief(None, False, "no_data")
        return FusedBelief(weighted_sum / weight_total, True, "weighted")

class PlausibilityFilteredFusion(FusionPolicy):
    """Filters out sensor readings that jump implausibly compared with
    what that same sensor most recently reported, then hands whatever
    survives to MajorityVoteFusion's agreement logic.

    Every other policy trusts a reading's magnitude (agreement,
    confidence) but none check whether a reading is physically plausible
    given what the vehicle just saw - which is why an attack reporting
    an implausible sudden jump (radar_spoof_masking jumping 58m -> 200m
    in one tick) can succeed against them. Same principle already used
    for GPS in navigation.py's PlausibilityCheckedGPSPolicy, applied
    here to sensor fusion generally rather than as an attack-specific
    patch.

    Design decisions:

    - Plausibility is bounded by a fixed, attacker-independent ceiling
      (MAX_PLAUSIBLE_CLOSING_SPEED), not by a reading's own
      self-reported detected_velocity. An attacker who fabricates a
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
      existing tracked one, is not caught by this policy. Rejecting
      first appearances that aren't near the sensor's own max range
      would also flag legitimate scenario design (a hazard hidden until
      a vehicle is close to a bend), so this is left unaddressed.
    - A small spoof within sensor noise tolerance is indistinguishable
      from ordinary sensor imprecision - a general limitation of
      plausibility-based defences, not specific to this implementation.
    - A gradually-ramped attack (moving the spoofed value a small,
      plausible amount each tick) would evade this check, since each
      tick-to-tick change stays within tolerance even though the
      cumulative drift does not.
    - Ground truth only reports the single nearest candidate each tick
      with no persistent object identity across ticks (see
      _ground_truth_readings in engine.py). A legitimate change in which
      real object is nearest could be misread as an implausible jump,
      though this is believed to be rare given how despawning/exiting
      range are currently modelled.
    """

    name = "plausibility_filtered"

    # Fixed, attacker-independent ceiling on how much any single real
    # candidate's distance could plausibly change in one tick - roughly
    # the fastest closing speed two vehicles could produce in this
    # simulator (two fast vehicles closing head-on, per the
    # obstacle_swerve speed sweep). Not derived from a reading's own
    # self-reported detected_velocity - see class docstring.
    MAX_PLAUSIBLE_CLOSING_SPEED = 45.0   # m/s, ~100mph

    # Extra per-sensor tolerance on top of the physical ceiling above,
    # reflecting that real sensors differ in precision (LiDAR tightest,
    # radar noisiest). This project's own estimates, not drawn from a
    # cited source, unlike severity.py's figures.
    NOISE_TOLERANCE = {
        SensorType.LIDAR: 1.0,
        SensorType.CAMERA: 2.0,
        SensorType.RADAR: 3.0,
    }

    def __init__(self, timestep: float = 0.1, tolerance: float = 3.0):
        # timestep defaults to 0.1 since every scenario in this project
        # uses that value. Fusion policies are constructed with no
        # arguments (see FUSION_POLICIES usage in engine.py), so there
        # is no path yet for a scenario with a different timestep to
        # override this.
        self.timestep = timestep
        self.tolerance = tolerance
        self._last_trusted: dict[SensorType, float | None] = {}
        # Sticky: once True for a sensor, stays True. Tracks whether
        # that sensor has ever had its reading corroborated by another
        # sensor agreeing with it, not merely whether it has ever
        # reported something - see fuse() for why this distinction
        # matters.
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
                # sudden absence - consistent with jamming, exclude,
                # nothing plausible left to hold onto
                filtered[sensor_type] = SensorReading(sensor_type)
                self._last_trusted[sensor_type] = None
                continue

            if last is None:
                # first appearance - accepted for combination purposes
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

        # Corroboration bookkeeping: any two currently-accepted readings
        # that agree with each other mark both sensors as having a
        # track record of being corroborated - sticky, so it persists
        # even on a later tick where a peer briefly goes silent or gets
        # filtered out. This is what distinguishes a coordinated
        # radar+LiDAR spoof (camera was corroborated by both for many
        # prior ticks before the attack started) from a
        # camera_phantom-style fabrication (camera is never corroborated
        # by anything, since the phantom is the only thing "there").
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
            # exactly one sensor survived filtering and has a track
            # record of corroboration: trust it directly rather than
            # handing it to MajorityVoteFusion, which would discard it
            # for lack of a second agreeing sensor this tick. A lone
            # sensor with no corroboration history (e.g. a phantom
            # nothing else has ever agreed with) still falls through to
            # ordinary majority-vote behaviour below, matching plain
            # majority_vote's own resistance to that attack.
            sole = next(iter(established_survivors.values()))
            return FusedBelief(sole.detected_distance, True, "plausibility:single_survivor")

        belief = self._majority.fuse(filtered)
        # relabelled with a "plausibility:" prefix so this is
        # distinguishable from genuine majority_vote output in a mixed
        # trace/log - mirrors the "hazard:" prefix convention used for
        # TickResult.ground_truth_kind in engine.py
        return FusedBelief(belief.distance_to_obstacle, belief.obstacle_present, f"plausibility:{belief.source}")

FUSION_POLICIES = {
    "camera_priority": CameraPriorityFusion,
    "majority_vote": MajorityVoteFusion,
    "confidence_weighted": ConfidenceWeightedFusion,
    "plausibility_filtered": PlausibilityFilteredFusion,
}
