"""Unit conversions - The physics engine works in meters and meters/second, the interface works in mph"""

# 1609.344 / 3600 = 0.44704 (3600 = number of seconds in an hour)
MPH_TO_MS = 0.44704

def mph_to_ms(mph: float) -> float:
    return mph * MPH_TO_MS

def ms_to_mph(ms: float) -> float:
    return ms / MPH_TO_MS