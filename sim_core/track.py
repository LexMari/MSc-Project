from dataclasses import dataclass
import math

@dataclass
class Track:
    straight_length: float = 200.0  # length of straight segments in meters
    radius: float = 60.0  # radius of rounded ends in metres
    junction_at: float = None  # distance along track
    roundabout_at: float = None  # distance along track

    def __post_init__(self):
        if self.junction_at is None:
            self.junction_at = 1.5 * self.straight_length + math.pi * self.radius
        if self.roundabout_at is None:
            self.roundabout_at = 0.75 * self.straight_length

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