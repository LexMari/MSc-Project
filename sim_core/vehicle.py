"""Vehicle ground-truth state

Vehicles move along a single scalar coordinate, 's' (distance travelled
along the Track - see track.py), rather than raw (x, y). This is what
makes placing a vehicle "150m before the junction" a one-line config
value, and makes "distance to the vehicle ahead" a subtraction
along the loop rather than a straight-line calculation.

Converting 's' to (x, y, heading) for rendering/GPS purposes is the
Track's job, not in this module -- see Track.position_at().
"""
from dataclasses import dataclass

@dataclass
class VehicleState:
    """ground-truth physical state of a vehicle at a point in time"""

    vehicle_id: str
    s: float  # distance travelled along the track, metres
    speed: float  # m/s, always >= 0
    acceleration: float = 0.0  # m/s^2, set by vehicle's controller

    def step(self, dt: float) -> None:
        """Advance vehicle ground-truth state forward by dt seconds"""
        self.speed = max(0.0, self.speed + self.acceleration * dt)
        self.s += self.speed * dt