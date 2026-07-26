from sim_core.vehicle import VehicleState
from sim_core.track import Track
from sim_core.junction import TrafficLight
from sim_core.sensors import SensorReading, SensorType
from sim_core.fusion import CameraPriorityFusion, MajorityVoteFusion
from sim_core.scenario import load_scenario
from sim_core.engine import Simulation

def test_vehicle_moves_forward_along_track():
    v = VehicleState("test", s=0.0, speed=10.0)
    v.step(dt=1.0)
    assert v.s == 10.0

def test_vehicle_speed_does_not_go_negative():
    v = VehicleState("test", s=0.0, speed=1.0, acceleration=-10.0)
    v.step(dt=1.0)
    assert v.speed == 0.0

def test_track_position_is_continuous_across_segment_boundaries():
    """four segments need to join up
    without issues - no geometry mistakes pls"""
    track = Track()
    boundaries = [track.straight_length, track.straight_length + 3.14159 * track.radius]
    for s in boundaries:
        x1, y1, _ = track.position_at(s - 0.01)
        x2, y2, _ = track.position_at(s + 0.01)
        assert abs(x1 - x2) < 0.1
        assert abs(y1 - y2) < 0.1

def test_track_loop_closes():
    """position at s=0 and s=total_length should be the same point"""
    track = Track()
    x1, y1, _ = track.position_at(0)
    x2, y2, _ = track.position_at(track.total_length)
    assert abs(x1 - x2) < 1e-6
    assert abs(y1 - y2) < 1e-6

def test_distance_ahead_wraps_around_loop():
    """a vehicle just before end of the loop and one just past the
    start should report a small forward distance, not almost a full lap"""
    track = Track()
    d = track.distance_ahead(track.total_length - 5, 5)
    assert abs(d - 10) < 1e-6

def test_traffic_light_cycles_through_states():
    light = TrafficLight(red_duration=8.0, green_duration=6.0, amber_duration=2.0)
    assert light.state_at(0.0) == "red"
    assert light.state_at(8.0) == "green"
    assert light.state_at(14.0) == "amber"
    assert light.state_at(16.0) == "red"  # cycle repeats

def test_camera_priority_trusts_camera_over_missing_radar():
    readings = {SensorType.CAMERA: SensorReading(SensorType.CAMERA, detected_distance=5.0)}
    belief = CameraPriorityFusion().fuse(readings)
    assert belief.obstacle_present
    assert belief.distance_to_obstacle == 5.0

def test_majority_vote_rejects_lone_spoofed_sensor():
    """a radar spoof alone should not fool majority vote fusion,
    since only one of two sensors disagrees with reality"""
    readings = {
        SensorType.RADAR: SensorReading(SensorType.RADAR, detected_distance=3.0, is_attacked=True),
        SensorType.CAMERA: SensorReading(SensorType.CAMERA, detected_distance=40.0),
    }
    belief = MajorityVoteFusion().fuse(readings)
    assert not belief.obstacle_present

def test_camera_priority_ignores_radar_only_attack():
    """an attack on a sensor the fusion policy doesn't even
    consult should have zero effect - attack success depends on the
    fusion policy, not just attack itself"""
    readings = {
        SensorType.CAMERA: SensorReading(SensorType.CAMERA, detected_distance=40.0),
        SensorType.RADAR: SensorReading(SensorType.RADAR, detected_distance=3.0, is_attacked=True),
    }
    belief = CameraPriorityFusion().fuse(readings)
    assert belief.distance_to_obstacle == 40.0

def test_position_triggered_attack_arms_only_near_the_junction():
    """an attack configured to fire 50m before junction should stay dormant far from it, and
    arm the moment vehicle comes within range"""
    from sim_core.attacks import CameraPhantom

    track = Track()
    attack = CameraPhantom(phantom_distance=3.0, duration=0.5, trigger_before_junction=50)

    far_from_junction = track.junction_at - 200
    attack.check_trigger(t=1.0, vehicle_s=far_from_junction, track=track)
    assert not attack.is_active(1.0)

    within_trigger_range = track.junction_at - 50
    attack.check_trigger(t=2.0, vehicle_s=within_trigger_range, track=track)
    assert attack.is_active(2.0)

def test_phantom_brake_scenario_runs_and_triggers_braking():
    """load the example scenario, run it, and check
    the attacked vehicle's speed actually drops at some point and stays
    reduced. uses a camera phantom attack, triggered by position (50m
    before the junction) against camera_priority fusion"""
    config = load_scenario("scenarios/phantom_brake.yaml")
    sim = Simulation(config)
    log = sim.run()

    follower_log = [e for e in log if e.vehicle_id == "follower"]
    speeds = [e.speed for e in follower_log]

    assert speeds[0] == 20.0, "should start at its configured cruising speed"
    assert min(speeds) < 20.0, "phantom attack should trigger braking at some point"
    assert speeds[-1] < 20.0, "speed should remain reduced" #  no re-acceleration modelled yet