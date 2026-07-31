"""Unit conversions. The physics engine works entirely in meters and meters/second.
"""

MPH_TO_MS = 0.44704


def mph_to_ms(mph: float) -> float:
    return mph * MPH_TO_MS


def ms_to_mph(ms: float) -> float:
    return ms / MPH_TO_MS