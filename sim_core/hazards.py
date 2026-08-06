"""Natural hazards: real events a vehicle's sensors should
correctly perceive, unlike attacks, which corrupt what a sensor reports.

A hazard is ground truth, not a sensor manipulation - while a
PedestrianCrossing is present, every sensor correctly reports it (no
'is_attacked' flag). A real hazard should be detected consistently by every fusion
policy, since every sensor agrees about it, an attack succeeds or fails
depending on the policy, since only the attacked sensor disagrees.
"""
from dataclasses import dataclass

# How much more cautiously a vehicle should treat a pedestrian of each
# type - applied as a multiplier on the braking model's comfort_distance
# (see braking.py), so a higher figure means braking begins earlier and
# more gently, rather than harder at the same distance. Child modelled
# as most cautious (drivers react more cautiously to children in
# practice, reflecting their unpredictability), elderly next (slower,
# less predictable crossing speed), adult as the baseline figure every
# other hazard in this project already uses.
PEDESTRIAN_CAUTION_MULTIPLIER = {
    "child": 1.3,
    "elderly": 1.15,
    "adult": 1.0,
}

WALKING_SPEED = 1.4   # m/s, a commonly-cited average adult walking pace, at LANE_WIDTH=3.5m (track.py), crossing one lane at this pace takes ~2.5s, so a duration of ~5s crosses both lanes

@dataclass
class PedestrianCrossing:
    """A pedestrian crossing the road at a named track feature.

    A pedestrian walks across the road one lane at a time - not blocking
    every lane for the entire crossing, and not standing still in the
    road for an arbitrarily long duration. `duration` is the total time
    to cross both lanes, first half in lane 0, second half in lane 1 -
    see lane_at() below, which every hazard-aware part of the engine
    should use instead of is_present() alone when deciding whether this
    hazard is currently relevant to a specific vehicle's lane.

    pedestrian_type is one of "child" | "adult" | "elderly" (default
    "adult") and affects how cautiously a vehicle brakes for it - see
    PEDESTRIAN_CAUTION_MULTIPLIER and engine.py's use of it.

    start_time may be left as None, in which case a scenario-level seeded
    RNG (see ScenarioConfig.random_seed) picks a start time uniformly at
    random within the scenario's duration when the simulation is built -
    see Simulation.__init__ in engine.py, which is where None gets
    resolved to a concrete value before is_present() is ever called.
    """

    feature_id: str    # which track feature this hazard is located at
    duration: float
    start_time: float | None = None
    pedestrian_type: str = "adult"

    def is_present(self, t: float) -> bool:
        if self.start_time is None:
            raise ValueError("start_time is None - it must be resolved (e.g. by Simulation.__init__) before is_present() can be called")
        return self.start_time <= t < self.start_time + self.duration

    def lane_at(self, t: float) -> int | None:
        """Which lane (0 or 1) the pedestrian is actually in at time t,
        or None if they haven't started crossing yet or have already
        finished. Splits `duration` evenly, first half in lane 0, second
        half in lane 1 - a vehicle in lane 0 should stop treating this
        as a hazard once the pedestrian has moved on to lane 1."""
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
    """A stationary obstacle blocking a lane at a named track feature -
    e.g. a broken-down car, debris, roadworks. Unlike PedestrianCrossing,
    this always occupies a specific lane (default 0, the normal driving
    lane), since which lane an obstacle blocks determines whether a
    vehicle needs to swerve into the oncoming lane to avoid it, or can
    stay in its own lane. start_time/duration behave the same as
    PedestrianCrossing - omit duration (leave it very large) for an
    obstacle that's present for the whole scenario."""

    feature_id: str
    start_time: float = 0.0
    duration: float = 10_000.0
    lane: int = 0

    def is_present(self, t: float) -> bool:
        return self.start_time <= t < self.start_time + self.duration

    def lane_at(self, t: float) -> int | None:
        """Same interface as PedestrianCrossing.lane_at(), but simpler:
        an obstacle occupies one fixed lane for as long as it's
        present."""
        return self.lane if self.is_present(t) else None

HAZARD_TYPES = {
    "pedestrian_crossing": PedestrianCrossing,
    "obstacle_in_road": ObstacleInRoad,
}
