"""GPS belief policies.

This exists so that a vehicle can determine where it is which is
different to what a FusionPolicy can do.

Two policies are implemented:
  - NaiveGPSPolicy trusts every GPS reading unconditionally, however
    large a jump it implies
  - PlausibilityCheckedGPSPolicy rejects a reading that implies the
    vehicle moved further than physically possible given its own speed
    since the last reading, falling back to holding its last known good
    position rather than trusting the implausible jump

A small GPS spoof (a few metres) is plausible under both policies -
indistinguishable from ordinary GPS receiver noise. A large, sudden spoof
gets caught by the plausibility check, but still fools the naive policy -
this is what lets a large-enough spoof cause a vehicle to be confused
about its position at a roundabout (see engine.py), rather than
every spoof having an identical effect regardless of size.
"""
from .sensors import SensorReading
from .vehicle import VehicleState

GPS_NOISE_MARGIN = 5.0   # metres - roughly a real GPS's accuracy

class GPSPolicy:
    """base class for GPS belief policies. subclasses override resolve()"""
    name = "base"

    def resolve(self, reading: SensorReading, vehicle: VehicleState, dt: float) -> tuple[float, float]:
        """Return the position the vehicle should believe it occupies"""
        raise NotImplementedError

class NaiveGPSPolicy(GPSPolicy):
    """trusts every GPS reading as reported, however implausible the reading change is"""

    name = "naive"

    def resolve(self, reading: SensorReading, vehicle: VehicleState, dt: float) -> tuple[float, float]:
        """Return the position the vehicle should believe it occupies"""
        vehicle.gps_last_believed = reading.detected_position
        return reading.detected_position

class PlausibilityCheckedGPSPolicy(GPSPolicy):
    """rejects a GPS reading that implies a jump too large for the
    vehicle to have physically travelled since the last reading, uses
    its last known good position instead of trusting the implausible one"""

    name = "plausibility_checked"

    def resolve(self, reading: SensorReading, vehicle: VehicleState, dt: float) -> tuple[float, float]:
        """Return the position the vehicle should believe it occupies"""
        if vehicle.gps_last_believed is None:
            vehicle.gps_last_believed = reading.detected_position
            return reading.detected_position

        # Furthest the vehicle could've moved this tick based on its speed
        # takes into consideration the potential inaccuracy of GPS
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