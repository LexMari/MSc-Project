"""Natural hazards: real events a vehicle's sensors should
correctly perceive

A hazard is ground truth - while a PedestrianCrossing
is present, every sensor correctly reports it (no 'is_attacked' flag).

A real hazard should be detected consistently by every fusion
policy, since every sensor agrees about it
"""
from dataclasses import dataclass

# Multiplier on the braking model's comfort_distance (braking.py), so a higher
# figure means braking starts earlier and more gently - not harder at the same
# distance. Child highest (drivers treat children as least predictable), elderly
# next (slower, less predictable crossing), adult is the baseline.
PEDESTRIAN_CAUTION_MULTIPLIER = {
    "child": 1.3,
    "elderly": 1.15,
    "adult": 1.0,
}

WALKING_SPEED = 1.4   # m/s, average adult walking pace, crossing one lane at this pace takes ~2.5s, so a duration of ~5s crosses both lanes

@dataclass
class PedestrianCrossing:
    """A pedestrian crossing at a named track feature, one lane at a time

    'duration' is the total time to cross both lanes: first half in lane 0,
    second in lane 1. Anything deciding whether this hazard applies to a specific
    vehicle should use lane_at(), not is_present() alone.

    pedestrian_type "child" / "adult" / "elderly" sets how cautiously vehicles
    brake - see PEDESTRIAN_CAUTION_MULTIPLIER.

    start_time of this may be None, resolved to a random value by Simulation.__init__
    from the scenario's seeded RNG before is_present() is ever called.
    """

    feature_id: str    # which track feature this hazard is located at
    duration: float    # seconds to cross both lanes
    start_time: float | None = None
    pedestrian_type: str = "adult"
    struck: bool = False   # set True once this pedestrian has been hit by any vehicle

    def is_present(self, t: float) -> bool:
        """Whether the pedestrian is in the road at time t.

          A struck pedestrian is no longer present, so a vehicle can't collide with
          the same one twice and following vehicles are not blocked by them.
          """
        if self.start_time is None:
            raise ValueError("start_time is None")
        if self.struck:
            return False
        return self.start_time <= t < self.start_time + self.duration

    def lane_at(self, t: float) -> int | None:
        """Which lane the pedestrian occupies at t, or None if not crossing.

        Splits duration evenly: lane 0 first, then lane 1. Direction is fixed -
        pedestrians always cross 0 to 1, never the reverse.
        """
        if not self.is_present(t):
            return None
        elapsed = t - self.start_time
        half = self.duration / 2
        return 0 if elapsed < half else 1

    @property
    def caution_multiplier(self) -> float:
        return PEDESTRIAN_CAUTION_MULTIPLIER.get(self.pedestrian_type, 1.0)

@dataclass
class ObstacleInRoad:
    """A stationary obstacle blocking one lane

    Unlike PedestrianCrossing this occupies a fixed lane for its whole duration,
    which is what determines whether a vehicle can stay put or must swerve into
    the oncoming lane
    """

    feature_id: str
    start_time: float = 0.0
    duration: float = 10_000.0 # permanent obstacle
    lane: int = 0

    def is_present(self, t: float) -> bool:
        return self.start_time <= t < self.start_time + self.duration

    def lane_at(self, t: float) -> int | None:
        return self.lane if self.is_present(t) else None

HAZARD_TYPES = {
    "pedestrian_crossing": PedestrianCrossing,
    "obstacle_in_road": ObstacleInRoad,
}
