"""GPS belief policies.

Fusion (see fusion.py) arbitrates *obstacle* detection between sensors -
a different question from "where does the vehicle believe it is". This
module exists to answer that second question, since it needs its own
state (a policy has to remember the last believed position to judge
whether a new reading is plausible), unlike FusionPolicy's stateless
fuse() calls.

Two policies are implemented, mirroring the naive vs.
resilient contrast already explored with fusion policies:

  - NaiveGPSPolicy trusts every GPS reading unconditionally, however
    large a jump it implies.
  - PlausibilityCheckedGPSPolicy rejects a reading that implies the
    vehicle moved further than physically possible given its own speed
    since the last reading, falling back to holding its last known good
    position rather than trusting an implausible jump.

A small GPS spoof (a few metres) is plausible under both policies -
indistinguishable from ordinary GPS receiver noise. A large, sudden spoof
gets caught by the plausibility check, but still fools the naive policy -
this is what lets a large-enough spoof cause a vehicle to be confused
about its position at a roundabout (see engine.py), rather than
every spoof having an identical effect regardless of size.
"""
from .sensors import SensorReading
from .vehicle import VehicleState

GPS_NOISE_MARGIN = 5.0   # metres - roughly a real consumer GPS receiver's ordinary accuracy

class GPSPolicy:
    """base class for GPS belief policies. subclasses override 'resolve'"""

    name = "base"

    def resolve(self, reading: SensorReading, vehicle: VehicleState, dt: float) -> tuple[float, float]:
        raise NotImplementedError

class NaiveGPSPolicy(GPSPolicy):
    """trusts every GPS reading exactly as reported, however implausible"""

    name = "naive"

    def resolve(self, reading: SensorReading, vehicle: VehicleState, dt: float) -> tuple[float, float]:
        vehicle.gps_last_believed = reading.detected_position
        return reading.detected_position

class PlausibilityCheckedGPSPolicy(GPSPolicy):
    """rejects a GPS reading that implies a jump too large for the
    vehicle to have physically travelled since the last reading, holding
    its last known good position instead of trusting the implausible one"""

    name = "plausibility_checked"

    def resolve(self, reading: SensorReading, vehicle: VehicleState, dt: float) -> tuple[float, float]:
        if vehicle.gps_last_believed is None:
            vehicle.gps_last_believed = reading.detected_position
            return reading.detected_position

        plausible_max_jump = (vehicle.speed * dt) + GPS_NOISE_MARGIN
        lx, ly = vehicle.gps_last_believed
        rx, ry = reading.detected_position
        jump = ((rx - lx) ** 2 + (ry - ly) ** 2) ** 0.5

        if jump > plausible_max_jump:
            return vehicle.gps_last_believed  # reject: hold last known good position

        vehicle.gps_last_believed = reading.detected_position
        return reading.detected_position

GPS_POLICIES = {
    "naive": NaiveGPSPolicy,
    "plausibility_checked": PlausibilityCheckedGPSPolicy,
}