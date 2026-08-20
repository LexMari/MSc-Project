"""Collision severity classification.

Pedestrian and vehicle-vehicle/obstacle collisions use separate models,
since the underlying crash literature differs between the two.

Turning that into slight/serious/fatal happens in two steps: fatal vs non-fatal
is sampled against the cited probability using the scenario's seeded RNG, so
repeated runs give a distribution rather than one fixed answer; serious vs
slight for non-fatal cases uses SERIOUS_FLOOR, which is this project's own
threshold and not from either cited source.
"""

import math
from .units import ms_to_mph

SERIOUS_FLOOR = 0.05   # project's own choice - non-fatal collisions above this risk are "serious" rather than "slight"
                       # this roughly equates to a 27mph collision with a pedestrian and a 33mph collision with a vehicle

def pedestrian_fatality_risk(impact_speed_ms: float) -> float:
    """Probability of pedestrian fatality given the striking vehicle's
    impact speed (pedestrian assumed stationary).

    Rosen & Sander (2009), Accident Analysis & Prevention 41(3), 536-542.
    Coefficients from Table 6 (impact-speed-only model), reproduced in
    Richards (2010), TRL RSWP16, UK Department for Transport.
    P(v) = 1 / (1 + exp(-(-6.9 + 0.090*v))), v in km/h."""
    v_kmh = impact_speed_ms * 3.6
    return 1 / (1 + math.exp(-(-6.9 + 0.090 * v_kmh)))

def vehicle_fatality_risk(closing_speed_ms: float) -> float:
    """Probability of driver fatality given closing/delta-v speed.

    Fitted to three published risk points for belted drivers
    in frontal impacts, Richards (2010) Fig 3.3: ~3% at 30mph,
    ~17% at 40mph, ~60% at 50mph. Frontal rather than side impact
    since this simulator never produces a T-bone collision - side
    impacts are considerably more severe at the same delta-v."""
    v_mph = ms_to_mph(closing_speed_ms)
    return 1 / (1 + math.exp(-(-9.5131 + 0.1983 * v_mph)))

def classify_severity(fatality_risk: float, rng) -> str:
    """Classifies "slight" | "serious" | "fatal" from a fatality-risk
    probability, using the same seeded RNG as driver diversity,
    pedestrian timing, and the traffic spawner"""
    if rng.random() < fatality_risk:
        return "fatal"
    return "serious" if fatality_risk >= SERIOUS_FLOOR else "slight"
