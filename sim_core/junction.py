"""Traffic light state machine for the track's junction"""
from dataclasses import dataclass

STOP_LINE_OFFSET = 3.0   # metres before a junction's marked position where a vehicle should actually stop

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
        if phase < self.green_duration:
            return "green"
        phase -= self.green_duration
        if phase < self.amber_duration:
            return "amber"
        return "red"