"""Traffic light state machine for the track's junction. This exists so
the junction is functional, not just decorative (see Discussion of
Approaches, Section 4) - vehicles will use this to decide whether to
stop (see engine.py TODO)
"""
from dataclasses import dataclass


@dataclass
class TrafficLight:
    """fixed-cycle traffic light: red -> green -> amber -> red, repeating"""

    red_duration: float = 8.0
    green_duration: float = 6.0
    amber_duration: float = 2.0

    @property
    def cycle_length(self) -> float:
        return self.red_duration + self.green_duration + self.amber_duration

    def state_at(self, t: float) -> str:
        """returns 'red', 'green', or 'amber' for a given simulation time"""
        phase = t % self.cycle_length
        if phase < self.red_duration:
            return "red"
        phase -= self.red_duration
        if phase < self.green_duration:
            return "green"
        return "amber"