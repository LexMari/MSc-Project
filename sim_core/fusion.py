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
    source: str  # which sensor(s)/logic the fused belief came from — useful for later analysis/plots

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

FUSION_POLICIES = {
    "camera_priority": CameraPriorityFusion,
    "majority_vote": MajorityVoteFusion,
    "confidence_weighted": ConfidenceWeightedFusion,
}
