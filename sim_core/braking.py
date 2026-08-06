"""A proportional braking model, using the UK Highway Code
"typical stopping distances" figures cited in the literature review
(Rule 126): total stopping distance = thinking distance + braking
distance, both of which scale with speed.

From the published figures (30mph -> 23m total, 9m thinking + 14m
braking, 60mph -> 73m total, 18m thinking + 55m braking), two constants
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
threshold, not a proportional one.

SAFETY_MARGIN exists to fix a separate, later-discovered issue: without
it, the model targets stopping *exactly at* the reported obstacle
distance, which at very low speeds (where thinking_distance shrinks
towards zero) allows the vehicle to creep to within a fraction of a
metre of a real hazard before finally halting - mathematically "never
collides," but not what a safety-critical system should actually do.
SAFETY_MARGIN shifts every distance calculation below to treat the
obstacle as if it were SAFETY_MARGIN metres closer than reported, so the
vehicle brakes to a stop with a genuine buffer of clearance, not zero.
"""

REACTION_TIME = 0.67          # seconds, derived from Highway Code thinking-distance figures
MAX_DECELERATION = 6.5        # m/s^2, derived from Highway Code braking-distance figures
COMFORT_MARGIN = 1.5          # braking begins gently at this multiple of the safe stopping distance
SAFETY_MARGIN = 2.0           # metres - the buffer a vehicle aims to leave between itself and a stopped-for obstacle

def safe_stopping_distance(speed: float, reaction_time: float = REACTION_TIME, max_deceleration: float = MAX_DECELERATION) -> float:
    """The total distance (thinking + braking) needed to stop safely from
    the given speed under maximum braking. reaction_time and
    max_deceleration default to the Highway Code-derived constants above,
    but can be overridden per vehicle - see VehicleState.reaction_time /
    .max_deceleration in vehicle.py, which model driver diversity (a
    more attentive or more aggressive driver braking differently from
    the Highway Code's 'typical' figures)."""
    if speed <= 0:
        return 0.0
    return (speed * reaction_time) + (speed ** 2 / (2 * max_deceleration))

def can_stop_safely(speed: float, distance_to_obstacle: float, reaction_time: float = REACTION_TIME, max_deceleration: float = MAX_DECELERATION, caution_multiplier: float = 1.0) -> bool:
    """Whether braking alone (even at max_deceleration, from right now)
    can bring the vehicle to a stop before reaching the obstacle, with a
    SAFETY_MARGIN of clearance to spare. This is exactly
    safe_stopping_distance() compared against the (margin-adjusted)
    distance available - it answers a different question to
    required_deceleration(): that function always returns *some*
    deceleration to apply, capped at max_deceleration, even when maximum
    braking isn't enough. This function is what a vehicle should
    check before deciding braking is sufficient on its own, versus needing
    a more drastic response (e.g. a swerve) instead.

    caution_multiplier scales the safe-stopping-distance threshold up
    for context that should make a vehicle more cautious than usual --
    e.g. a child pedestrian nearby (see PEDESTRIAN_CAUTION_MULTIPLIER in
    hazards.py) - without changing the vehicle's actual physical braking
    capability (reaction_time/max_deceleration stay as given)."""
    effective_distance = max(0.0, distance_to_obstacle - SAFETY_MARGIN)
    return effective_distance >= safe_stopping_distance(speed, reaction_time, max_deceleration) * caution_multiplier

def required_deceleration(speed: float, distance_to_obstacle: float, reaction_time: float = REACTION_TIME, max_deceleration: float = MAX_DECELERATION, caution_multiplier: float = 1.0) -> float:
    """Returns the deceleration (m/s^2, positive value) a vehicle should
    apply given its speed and the distance to a reported obstacle. All
    braking is calculated against an effective distance that reserves
    SAFETY_MARGIN metres of clearance - the vehicle brakes as if the
    obstacle were that much closer than it actually is reported to be:

      - 0.0 if the effective distance is at or beyond 'comfort_distance'
        (plenty of room, no braking needed yet)
      - max_deceleration if the effective distance is at or within the
        reaction-time distance alone (stopping in time, with margin,
        isn't achievable even under maximum braking)
      - linear between the two otherwise, so braking
        intensifies as the situation gets more urgent, rather
        than jumping straight from nothing to maximum

    caution_multiplier scales comfort_distance up, so a vehicle starts
    responding sooner (at a greater distance) for context that warrants
    extra caution - see can_stop_safely() above for the same idea."""
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