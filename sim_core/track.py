"""Track geometry: a closed stadium loop - two straight segments joined
by two semicircular ends.

Vehicles are positioned and move using a single scalar distance-along-
track value, 's', rather than raw (x, y).

Track features like junctions/roundabouts are a configurable list
so a scenario can define its own layout. An empty list falls back to
one junction and roundabout.

Lanes: the centreline is a single line, offset sideways by half a lane width.
Lane 0 is the correct-side lane for traffic with increasing s; lane 1 is the
opposite lane, used by oncoming traffic and by swerving vehicles.
"""
from dataclasses import dataclass, field
import math

LANE_WIDTH = 3.5   # metres

@dataclass
class TrackFeature:
    """A named point of interest on the track - a junction, roundabout,
    or something like a pedestrian crossing"""
    feature_id: str
    feature_type: str   # "junction" | "roundabout" |
    position: float      # distance along the track
    radius: float | None = None  # the radius of the roundabout island, used to compute the extra distance a confused vehicle travels taking an extra lap

DEFAULT_ROUNDABOUT_RADIUS = 25.0  # metres

@dataclass
class Track:
    straight_length: float = 200.0   # length of each straight segment, metres
    radius: float = 60.0             # radius of each rounded end, metres
    features: list[TrackFeature] = field(default_factory=list)

    def __post_init__(self):
        if not self.features:
            # Default layout, one junction on the top straight,
            # one roundabout near the end of the bottom straight.
            self.features = [
                TrackFeature("junction_1", "junction", 1.5 * self.straight_length + math.pi * self.radius),
                TrackFeature("roundabout_1", "roundabout", 0.75 * self.straight_length, radius=DEFAULT_ROUNDABOUT_RADIUS),
            ]

    def feature(self, feature_id: str) -> TrackFeature:
        for f in self.features:
            if f.feature_id == feature_id:
                return f
        raise KeyError(f"no track feature named {feature_id!r} - available: "
                        f"{[f.feature_id for f in self.features]}")

    def features_of_type(self, feature_type: str) -> list[TrackFeature]:
        return [f for f in self.features if f.feature_type == feature_type]

    def feature_circumference(self, feature_id: str) -> float:
        """The physical distance travelled taking one full lap of a
        roundabout-type feature, used to model a missed exit - a
        confused vehicle's true position advances by this much
        extra before it continues on the main loop. Raises if the
        feature has no radius set."""
        f = self.feature(feature_id)
        if f.radius is None:
            raise ValueError(f"feature {feature_id!r} has no radius set - feature_circumference "
                              f"only applies to roundabout features")
        return 2 * math.pi * f.radius

    @property
    def total_length(self) -> float:
        """full perimeter: two straights plus two semicircles"""
        return 2 * self.straight_length + 2 * math.pi * self.radius

    def normalise(self, s: float) -> float:
        """wrap distance value onto [0, total_length)"""
        return s % self.total_length

    def position_at(self, s: float) -> tuple[float, float, float]:
        """return (x, y, heading_radians) for distance s along loop
        (centreline - see lane_position_at() for a lane-offset version)

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

        if s < math.pi * r:  # right semicircle
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

    def lane_position_at(self, s: float, lane: int = 0) -> tuple[float, float, float]:
        """Like position_at, but offset sideways from the centreline by
        half a lane width, to the correct side for two-way traffic.

        lane=0 -> near/correct-side lane (right of the direction of
        travel, i.e. normal traffic increasing s)
        lane=1 -> opposite/oncoming lane (left of the direction of
        travel)

        The offset is perpendicular to the heading at that point, so this
        works correctly on the curved sections as well as the straights.
        Heading is unchanged by the lane offset - only (x, y) shifts.
        """
        x, y, heading = self.position_at(s)
        side = -1 if lane == 0 else 1
        offset = side * (LANE_WIDTH / 2)
        lane_x = x + offset * math.sin(heading)
        lane_y = y - offset * math.cos(heading)
        return lane_x, lane_y, heading

    def distance_ahead(self, s_from: float, s_to: float) -> float:
        """Forward distance travelling from s_from to reach s_to, wrapping
        around the loop if needed (always >= 0)"""
        return self.normalise(s_to - s_from)

    def signed_gap(self, s_from: float, s_to: float) -> float:
        """Shortest signed distance from s_from to s_to around the loop,
        in (-total_length/2, total_length/2]

        Positive means s_to is nearer going forward, negative means nearer
        going backward. distance_ahead() only wraps forward so this is used
        when the two vehicle's travel in opposite directions - see
        engine.py's _ahead_distance_to_vehicle"""
        raw = self.normalise(s_to - s_from)
        if raw > self.total_length / 2:
            raw -= self.total_length
        return raw