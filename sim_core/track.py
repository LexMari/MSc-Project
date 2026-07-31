"""Track geometry: a closed stadium loop - two straight segments joined
by two semicircular ends.

Vehicles are positioned and move using a single scalar distance-along-
track value, 's', rather than raw (x, y). Named track features (junctions,
roundabouts, and in future other hazards such as pedestrian crossings)
are placed on the track as a configurable list, 'features', rather than
fixed fields - this allows a scenario to define its own track
layout (multiple junctions, a differently-placed roundabout, etc.)
instead of being stuck with one hardcoded arrangement. If no features are
configured, a default single junction and roundabout are used,
matching the concept art.
"""
from dataclasses import dataclass, field
import math

@dataclass
class TrackFeature:
    """A named point of interest on the track - a junction, roundabout,
    or something like a pedestrian crossing"""
    feature_id: str
    feature_type: str   # "junction" | "roundabout" | ...
    position: float      # distance along the track

@dataclass
class Track:
    straight_length: float = 200.0   # length of each straight segment, metres
    radius: float = 60.0             # radius of each rounded end, metres
    features: list[TrackFeature] = field(default_factory=list)

    def __post_init__(self):
        if not self.features:
            # Default layout, matching the original concept art: one
            # junction on the top straight, one roundabout near the end
            # of the bottom straight.
            self.features = [
                TrackFeature("junction_1", "junction", 1.5 * self.straight_length + math.pi * self.radius),
                TrackFeature("roundabout_1", "roundabout", 0.75 * self.straight_length),
            ]

    def feature(self, feature_id: str) -> TrackFeature:
        for f in self.features:
            if f.feature_id == feature_id:
                return f
        raise KeyError(f"no track feature named {feature_id!r} -- available: "
                        f"{[f.feature_id for f in self.features]}")

    def features_of_type(self, feature_type: str) -> list[TrackFeature]:
        return [f for f in self.features if f.feature_type == feature_type]

    @property
    def total_length(self) -> float:
        """full perimeter: two straights plus two semicircles"""
        return 2 * self.straight_length + 2 * math.pi * self.radius

    def normalise(self, s: float) -> float:
        """wrap distance value onto [0, total_length)"""
        return s % self.total_length

    def position_at(self, s: float) -> tuple[float, float, float]:
        """return (x, y, heading_radians) for distance s along loop

        travel direction: bottom straight (left to right, from spawn
        point at s=0) -> right-hand semicircle -> top straight (right to
        left) -> left-hand semicircle, closing
        loop back to spawn point
        """
        s = self.normalise(s)
        sl, r = self.straight_length, self.radius

        if s < sl:  # bottom straight, heading +x (0 rad)
            return s, 0.0, 0.0
        s -= sl

        if s < math.pi * r:  # right-hand semicircle
            angle = s / r
            x = sl + r * math.sin(angle)
            y = r - r * math.cos(angle)
            return x, y, angle
        s -= math.pi * r

        if s < sl:  # top straight, heading pi (travelling -x) - junction sits here
            x = sl - s
            y = 2 * r
            return x, y, math.pi
        s -= sl

        # left-hand semicircle, closing the loop
        angle = s / r
        x = -r * math.sin(angle)
        y = r + r * math.cos(angle)
        return x, y, math.pi + angle

    def distance_ahead(self, s_from: float, s_to: float) -> float:
        """Forward distance travelling from s_from to reach s_to, wrapping
        around the loop if needed (always >= 0)."""
        return self.normalise(s_to - s_from)