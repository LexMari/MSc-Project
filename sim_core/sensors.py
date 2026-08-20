"""Sensor reading models.

Each sensor reports what it perceives, which may differ from ground truth
once an attack is injected. Keeping SensorReading as plain data
is what lets attacks.py manipulate readings without
touching the underlying simulation.
"""
from dataclasses import dataclass
from enum import Enum

class SensorType(Enum):
    """the possible sensor types that a scenario can use"""
    GPS = "gps"
    RADAR = "radar"
    CAMERA = "camera"
    LIDAR = "lidar"

@dataclass
class SensorReading:
    """a single sensor's perceived state at one simulation tick.

    fields are populated per sensor type and left as None otherwise:
        - GPS fills detected_position
        - Camera fills signal_state
        - radar/LiDAR fill detected_distance and detected_velocity
    None means that a sensor does not report this metric

    'confidence' is a 0-1 value the fusion layer can use to weight this
    reading. A successful attack can fake a reading with high confidence.
    """

    sensor_type: SensorType
    detected_distance: float | None = None                  # metres to nearest obstacle/vehicle
    detected_velocity: float | None = None                  # m/s, relative closing speed
    detected_position: tuple[float, float] | None = None    # (x,y) for GPS
    signal_state: str | None = None                         # for camera reading a traffic light, e.g. "red"/"green"
    confidence: float = 1.0
    is_attacked: bool = False                               # ground-truth flag for logging only - is not read by fusion.py
