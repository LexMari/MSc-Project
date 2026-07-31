"""Natural hazards: real events a vehicle's sensors should
correctly perceive, unlike attacks, which corrupt what a sensor reports.

A hazard is ground truth, not a sensor manipulation - while a
PedestrianCrossing is present, every sensor correctly reports it (no
'is_attacked' flag). A real hazard should be detected consistently by every fusion
policy, since every sensor agrees about it; an attack succeeds or fails
depending on the policy, since only the attacked sensor disagrees.
"""
from dataclasses import dataclass

@dataclass
class PedestrianCrossing:
    """A pedestrian present at a named track feature"""

    feature_id: str    # which track feature this hazard is located at
    start_time: float
    duration: float

    def is_present(self, t: float) -> bool:
        return self.start_time <= t < self.start_time + self.duration

HAZARD_TYPES = {
    "pedestrian_crossing": PedestrianCrossing,
}