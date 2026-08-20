"""Proportional braking model based on the Highway Code Rule 126

Total stopping distance = thinking distance + braking distance.
This uses the figures (30mph: 9m + 14m and 60mph: 18m + 55m)
the two constants at both speeds are a 0.67s reaction time and a 6.5m/s^2 deceleration rate.

Braking ramps with how close the vehicle is to the hazard so further away will be a more gentle
braking speed. All distances are measured to the SAFETY_MARGIN metres short of the obstacle
"""

REACTION_TIME = 0.67          # seconds, based on the Highway Code thinking-distance figures
MAX_DECELERATION = 6.5        # m/s^2, from the Highway Code braking-distance figures
COMFORT_MARGIN = 1.5          # braking begins gently at this multiple of the safe stopping distance
SAFETY_MARGIN = 2.0           # metres - the gap a vehicle aims to leave between itself and an obstacle

def safe_stopping_distance(speed: float, reaction_time: float = REACTION_TIME, max_deceleration: float = MAX_DECELERATION) -> float:
    """The total distance needed to stop safely from the vehicle's current speed under maximum
    braking, this uses the default values above but they can be overridden per vehicle using
    VehicleState.reaction_time or .max_deceleration to model driver diversity (for more attentive
    or aggressive braking)"""
    if speed <= 0:
        return 0.0
    return (speed * reaction_time) + (speed ** 2 / (2 * max_deceleration))

def can_stop_safely(speed: float, distance_to_obstacle: float, reaction_time: float = REACTION_TIME, max_deceleration: float = MAX_DECELERATION, caution_multiplier: float = 1.0) -> bool:
    """Whether maximum braking from now can stop short of the obstacle

    Unlike required_deceleration(), which always returns a capped value this
    method asks whether braking is sufficient.

    This method is used for checking whether a vehicle should swerve."""
    effective_distance = max(0.0, distance_to_obstacle - SAFETY_MARGIN)
    return effective_distance >= safe_stopping_distance(speed, reaction_time, max_deceleration) * caution_multiplier

def required_deceleration(speed: float, distance_to_obstacle: float, reaction_time: float = REACTION_TIME, max_deceleration: float = MAX_DECELERATION, caution_multiplier: float = 1.0) -> float:
    """Returns the deceleration a vehicle should apply
    given its speed and the distance to a reported obstacle.

      - 0.0 if the effective distance is at or beyond 'comfort_distance'
        (no braking needed yet)
      - max_deceleration if the effective distance is at or within the
        reaction-time distance alone (stopping in time, with margin,
        isn't achievable even under maximum braking)
      - linear between the two otherwise, so braking
        intensifies as the situation gets more urgent, rather
        than jumping straight from nothing to maximum

    caution_multiplier scales comfort_distance up, so a vehicle starts
    responding sooner"""
    if speed <= 0:
        return 0.0

    effective_distance = max(0.0, distance_to_obstacle - SAFETY_MARGIN)
    thinking_distance = speed * reaction_time
    comfort_distance = safe_stopping_distance(speed, reaction_time, max_deceleration) * COMFORT_MARGIN * caution_multiplier

    if effective_distance >= comfort_distance:
        return 0.0
    if effective_distance <= thinking_distance:
        return max_deceleration

    fraction_of_the_way_to_critical = (comfort_distance - effective_distance) / (comfort_distance - thinking_distance)
    return max_deceleration * fraction_of_the_way_to_critical