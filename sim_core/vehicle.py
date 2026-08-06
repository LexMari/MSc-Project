"""Vehicle ground-truth state

Vehicles move along a single scalar coordinate, 's' (distance travelled
along the Track - see track.py), rather than raw (x, y). This is what
makes placing a vehicle "150m before the junction" a one-line config
value, and makes "distance to the vehicle ahead" a subtraction
along the loop rather than a straight-line calculation.

Converting 's' to (x, y, heading) for rendering/GPS purposes is the
Track's job, not in this module - see Track.position_at() and
Track.lane_position_at().

direction distinguishes normal traffic (direction=1, s increases over
time) from oncoming traffic (direction=-1, s decreases over time) on the
same shared loop - see VehicleState.step(). lane distinguishes which
side of the centreline a vehicle currently occupies (0 = its own,
correct-side lane, 1 = the opposite lane) - see engine.py for how a
vehicle ends up in lane 1 (either as oncoming traffic's normal lane, or
as a swerve response to an obstacle in lane 0).
"""
from dataclasses import dataclass
from .braking import REACTION_TIME, MAX_DECELERATION

@dataclass
class VehicleState:
    """ground-truth physical state of a vehicle at a point in time"""

    vehicle_id: str
    s: float                    # distance travelled along the track, metres
    speed: float                # m/s, always >= 0 (a magnitude - direction controls which way it moves)
    cruise_speed: float         # m/s - the speed the vehicle resumes toward once no obstacle is believed present
    acceleration: float = 0.0   # m/s^2, set by the vehicle's controller each tick
    lane: int = 0                # 0 = own/correct-side lane, 1 = opposite lane
    direction: int = 1           # 1 = normal traffic (s increases), -1 = oncoming traffic (s decreases)

    # per-vehicle braking parameters, defaulting to the Highway
    # Code-derived constants in braking.py, but overridable per vehicle
    # (see scenario.py's VehicleConfig) to model driver diversity - a
    # more attentive or more aggressive driver braking differently from
    # the 'typical' figures every other vehicle uses by default.
    reaction_time: float = REACTION_TIME
    max_deceleration: float = MAX_DECELERATION

    # scratch state for an in-progress swerve manoeuvre (see engine.py),
    # not meaningful unless swerve_active is True
    swerve_active: bool = False
    swerve_progress: float = 0.0          # metres travelled since the swerve began
    swerve_return_progress: float = 0.0   # metres of travel needed before swerving back

    crashed: bool = False   # set once this vehicle has been involved in a collision - see engine.py step()
    severity: str | None = None   # "slight" | "serious" | "fatal" - set once, at the moment crashed becomes True, see severity.py

    gps_last_believed: tuple[float, float] | None = None   # last position this vehicle's GPS policy accepted - see navigation.py
    roundabouts_resolved: set = None   # feature_ids of roundabouts already checked for this approach (confused or not), so the check only fires once per approach - see engine.py
    roundabout_excursion_remaining: float = 0.0   # metres of an extra roundabout lap still left to drive - see engine.py and step() below

    def __post_init__(self):
        if self.roundabouts_resolved is None:
            self.roundabouts_resolved = set()

    def step(self, dt: float) -> None:
        """Advance vehicle ground-truth state forward by dt seconds"""
        self.speed = max(0.0, self.speed + self.acceleration * dt)
        travel = self.speed * dt * self.direction

        if self.roundabout_excursion_remaining > 0:
            magnitude = abs(travel)
            consumed = min(magnitude, self.roundabout_excursion_remaining)
            self.roundabout_excursion_remaining -= consumed
            leftover = magnitude - consumed
            travel = leftover if travel >= 0 else -leftover

        self.s += travel
        if self.swerve_active:
            self.swerve_progress += self.speed * dt