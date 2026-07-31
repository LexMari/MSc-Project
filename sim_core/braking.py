"""A proportional braking model, using the UK Highway Code
"typical stopping distances" figures cited in the literature review
(Rule 126): total stopping distance = thinking distance + braking
distance, both of which scale with speed.

From the published figures (30mph -> 23m total, 9m thinking + 14m
braking; 60mph -> 73m total, 18m thinking + 55m braking), two constants
can be derived and held fixed across both reference speeds:

  - a reaction time of ~0.67s (thinking distance / speed is constant
    across both reference points)
  - an effective braking deceleration of ~6.5 m/s^2 (from braking
    distance = speed^2 / (2 * deceleration), also consistent across
    both reference points)

Braking response is scaled between two distances, measured from a
target point that sits 'SAFETY_MARGIN' metres short of the
reported obstacle, not at it:
  - at or beyond 'comfort_distance' (a configurable multiple of the
    Highway Code safe stopping distance), no braking is needed at all.
  - at or within the reaction-time distance alone ('thinking_distance'),
    stopping in time is not achievable even under maximum braking, so
    the model applies MAX_DECELERATION.
  - in between, deceleration scales linearly from 0 up to
    MAX_DECELERATION as the obstacle gets closer.

This replaces the earlier fixed-threshold, fixed-deceleration controller,
which applied the same braking response regardless of how close a
reported obstacle actually was. An earlier version of this model defined
"required deceleration" as exactly the deceleration needed to stop at the
obstacle, but that has a flaw: because the Highway Code's own
distance figure is itself defined using MAX_DECELERATION, any distance
under it always demanded deceleration >= MAX_DECELERATION - the model
reverts to a binary 0-or-maximum response with a better-justified
threshold, not a genuinely proportional one.

SAFETY_MARGIN exists to fix a separate, later-discovered issue: without
it, the model targets stopping *exactly at* the reported obstacle
distance, which at very low speeds (where thinking_distance shrinks
towards zero) allows the vehicle to creep to within a fraction of a
metre of a real hazard before finally halting -- mathematically "never
collides," but not what a safety-critical system should actually do.
SAFETY_MARGIN shifts every distance calculation below to treat the
obstacle as if it were SAFETY_MARGIN metres closer than reported, so the
vehicle brakes to a stop with a genuine buffer of clearance, not zero.
"""

REACTION_TIME = 0.67          # seconds, derived from Highway Code thinking-distance figures
MAX_DECELERATION = 6.5        # m/s^2, derived from Highway Code braking-distance figures
COMFORT_MARGIN = 1.5          # braking begins gently at this multiple of the safe stopping distance
SAFETY_MARGIN = 2.0           # metres - the buffer a vehicle aims to leave between itself and a stopped-for obstacle

def safe_stopping_distance(speed: float) -> float:
    """The total distance (thinking + braking) needed to stop safely from
    the given speed under maximum braking"""
    if speed <= 0:
        return 0.0
    return (speed * REACTION_TIME) + (speed ** 2 / (2 * MAX_DECELERATION))

def required_deceleration(speed: float, distance_to_obstacle: float) -> float:
    """Returns the deceleration (m/s^2, positive value) a vehicle should
    apply given its speed and the distance to a reported obstacle. All
    braking is calculated against an effective distance that reserves
    SAFETY_MARGIN metres of clearance - the vehicle brakes as if the
    obstacle were that much closer than it actually is reported to be:

      - 0.0 if the effective distance is at or beyond 'comfort_distance'
        (plenty of room, no braking needed yet)
      - MAX_DECELERATION if the effective distance is at or within the
        reaction-time distance alone (stopping in time, with margin,
        isn't achievable even under maximum braking)
      - linear between the two otherwise, so braking
        intensifies as the situation gets more urgent, rather
        than jumping straight from nothing to maximum
    """
    if speed <= 0:
        return 0.0

    effective_distance = max(0.0, distance_to_obstacle - SAFETY_MARGIN)
    thinking_distance = speed * REACTION_TIME
    comfort_distance = safe_stopping_distance(speed) * COMFORT_MARGIN

    if effective_distance >= comfort_distance:
        return 0.0
    if effective_distance <= thinking_distance:
        return MAX_DECELERATION

    fraction_of_the_way_to_critical = (comfort_distance - effective_distance) / (comfort_distance - thinking_distance)
    return MAX_DECELERATION * fraction_of_the_way_to_critical