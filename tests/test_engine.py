import os
import tempfile
from sim_core.vehicle import VehicleState
from sim_core.track import Track
from sim_core.junction import TrafficLight
from sim_core.sensors import SensorReading, SensorType
from sim_core.fusion import CameraPriorityFusion, MajorityVoteFusion
from sim_core.scenario import load_scenario
from sim_core.engine import Simulation
from sim_core.units import mph_to_ms, ms_to_mph
from sim_core.braking import required_deceleration, safe_stopping_distance

def test_safe_stopping_distance_matches_highway_code_figures():
    """The derived model should reproduce the UK Highway Code's published
    stopping distances within rounding tolerance: 23m at
    30mph, 73m at 60mph."""
    assert abs(safe_stopping_distance(mph_to_ms(30)) - 23) < 0.5
    assert abs(safe_stopping_distance(mph_to_ms(60)) - 73) < 1.0

def test_required_deceleration_is_zero_with_plenty_of_room():
    assert required_deceleration(speed=20.0, distance_to_obstacle=500.0) == 0.0

def test_required_deceleration_scales_with_how_close_the_obstacle_is():
    """A closer obstacle should demand harder braking than a further one,
    at the same speed"""
    speed = mph_to_ms(45)
    harder = required_deceleration(speed, distance_to_obstacle=5.0)
    softer = required_deceleration(speed, distance_to_obstacle=40.0)
    assert harder > softer

def test_required_deceleration_caps_at_maximum_when_stopping_is_impossible():
    """If the obstacle is closer than the reaction-time distance alone,
    even maximum braking can't stop in time by the model's own logic -
    it should still cap at MAX_DECELERATION rather than exceed it"""
    from sim_core.braking import MAX_DECELERATION
    speed = mph_to_ms(70)
    assert required_deceleration(speed, distance_to_obstacle=1.0) == MAX_DECELERATION

def test_safety_margin_forces_maximum_braking_at_the_margin_boundary():
    """At any speed, a reported obstacle exactly SAFETY_MARGIN metres away
    should demand maximum braking - the effective distance (after
    subtracting the margin) is 0, which is always <= thinking_distance."""
    from sim_core.braking import SAFETY_MARGIN, MAX_DECELERATION
    speed = mph_to_ms(20)
    assert required_deceleration(speed, distance_to_obstacle=SAFETY_MARGIN) == MAX_DECELERATION

def test_pedestrian_crossing_is_present_only_during_its_window():
    from sim_core.hazards import PedestrianCrossing

    hazard = PedestrianCrossing(feature_id="crossing_1", start_time=3.0, duration=2.0)
    assert not hazard.is_present(2.9)
    assert hazard.is_present(3.0)
    assert hazard.is_present(4.9)
    assert not hazard.is_present(5.0)

def test_pedestrian_crossing_scenario_triggers_braking_under_every_fusion_policy():
    """The key contrast with attacks: a real hazard should be detected
    consistently regardless of fusion policy, since every sensor agrees
    about it"""
    for policy in ("camera_priority", "majority_vote", "confidence_weighted"):
        config = load_scenario("scenarios/pedestrian_crossing_test.yaml")
        config.vehicles[0].fusion_policy = policy
        sim = Simulation(config)
        log = sim.run()

        speeds = [e.speed for e in log if e.vehicle_id == config.vehicles[0].vehicle_id]
        assert min(speeds) < speeds[0], \
            f"expected braking for a genuine pedestrian hazard under {policy} fusion"

def test_mph_conversion_round_trips():
    original_mph = 45.0
    assert abs(ms_to_mph(mph_to_ms(original_mph)) - original_mph) < 1e-9

def test_scenario_speed_is_converted_from_mph_to_ms():
    """45 mph should load as roughly 20.1 m/s, not 45 m/s - catches the
    case where the mph->m/s conversion is silently skipped"""
    config = load_scenario("scenarios/phantom_brake.yaml")
    follower = next(v for v in config.vehicles if v.vehicle_id == "follower")
    assert 20.0 < follower.start_speed < 20.2

def test_vehicle_resumes_toward_cruise_speed_after_braking():
    """Re-acceleration: once no obstacle is believed present, a vehicle
    below its cruise speed should speed back up, not just hold at
    whatever reduced speed braking left it at"""
    import yaml
    scenario_dict = {
        "name": "resume test",
        "duration": 10,
        "timestep": 0.1,
        "vehicles": [{"vehicle_id": "solo", "start_distance": 0, "start_speed_mph": 30, "fusion_policy": "camera_priority"}],
        "hazards": [{"type": "pedestrian_crossing", "feature_id": "crossing_1", "start_time": 1.0, "duration": 1.0}],
        "track": {"straight_length": 200, "radius": 60,
                  "features": [{"feature_id": "crossing_1", "feature_type": "pedestrian_crossing", "position": 30}]},
    }
    scenario_path = os.path.join(tempfile.gettempdir(), "_resume_test.yaml")
    with open(scenario_path, "w") as f:
        yaml.dump(scenario_dict, f)

    config = load_scenario(scenario_path)
    sim = Simulation(config)
    log = sim.run()
    solo_log = [e for e in log if e.vehicle_id == "solo"]

    min_speed = min(e.speed for e in solo_log)
    final_speed = solo_log[-1].speed
    assert final_speed > min_speed, "expected the vehicle to speed back up after the hazard cleared"

def test_red_light_is_treated_as_an_obstacle_but_green_is_not():
    """A junction's traffic light should act as a virtual obstacle at its
    stop line while red/amber, and contribute nothing at all once green"""
    light = TrafficLight(red_duration=8.0, green_duration=6.0, amber_duration=2.0)
    assert light.state_at(0.0) == "red"      # should be treated as an obstacle
    assert light.state_at(8.0) == "green"    # should be treated as if it weren't there

def test_vehicle_stops_at_red_light_and_resumes_at_green():
    """End-to-end: a vehicle approaching a junction should brake for a
    red light, then resume toward cruise speed once it turns green -
    traffic-light equivalent of the pedestrian-crossing using the same underlying mechanism
    (virtual obstacle at a fixed point, cleared once the condition ends)."""
    import yaml
    scenario_dict = {
        "name": "traffic light test",
        "duration": 20,
        "timestep": 0.1,
        "vehicles": [{"vehicle_id": "car", "start_distance": 0, "start_speed_mph": 20, "fusion_policy": "camera_priority"}],
        "track": {"straight_length": 150, "radius": 60,
                  "features": [{"feature_id": "junction_1", "feature_type": "junction", "position": 60}]},
    }
    scenario_path = os.path.join(tempfile.gettempdir(), "_traffic_light_test.yaml")
    with open(scenario_path, "w") as f:
        yaml.dump(scenario_dict, f)

    config = load_scenario(scenario_path)
    sim = Simulation(config)
    log = sim.run()
    car_log = [e for e in log if e.vehicle_id == "car"]

    min_speed = min(e.speed for e in car_log)
    assert min_speed < car_log[0].speed, "expected the vehicle to slow down for the red light"
    assert car_log[-1].speed > min_speed, "expected the vehicle to resume once the light turned green"


def test_vehicle_moves_forward_along_track():
    v = VehicleState("test", s=0.0, speed=10.0, cruise_speed=10.0)
    v.step(dt=1.0)
    assert v.s == 10.0

def test_vehicle_speed_does_not_go_negative():
    v = VehicleState("test", s=0.0, speed=1.0, cruise_speed=1.0, acceleration=-10.0)
    v.step(dt=1.0)
    assert v.speed == 0.0

def test_track_position_is_continuous_across_segment_boundaries():
    """The four segments (two straights, two semicircles) need to join up
    without discontinuities - this would catch a geometry mistake"""
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

def test_position_triggered_attack_arms_only_near_the_named_feature():
    """Direct test of the trigger mechanism: an attack configured to fire
    50m before a named feature should stay dormant far from it, and arm
    the moment the vehicle comes within range"""
    from sim_core.attacks import CameraPhantom

    track = Track()
    attack = CameraPhantom(phantom_distance=3.0, duration=0.5,
                            trigger_before_feature="junction_1", trigger_distance=50)
    junction_position = track.feature("junction_1").position

    far_from_junction = junction_position - 200
    attack.check_trigger(t=1.0, vehicle_s=far_from_junction, track=track)
    assert not attack.is_active(1.0)

    within_trigger_range = junction_position - 50
    attack.check_trigger(t=2.0, vehicle_s=within_trigger_range, track=track)
    assert attack.is_active(2.0)

def test_track_supports_multiple_named_junctions():
    """A track with two junctions should let an attack be timed against
    either one independently, by name"""
    from sim_core.attacks import CameraPhantom
    from sim_core.track import TrackFeature

    track = Track(features=[
        TrackFeature("junction_1", "junction", 100.0),
        TrackFeature("junction_2", "junction", 400.0),
    ])

    attack_on_second_junction = CameraPhantom(
        phantom_distance=3.0, duration=0.5,
        trigger_before_feature="junction_2", trigger_distance=20,
    )

    # near junction_1 (s=90), but the attack cares about junction_2 (at s=400) - should NOT arm
    attack_on_second_junction.check_trigger(t=1.0, vehicle_s=90.0, track=track)
    assert not attack_on_second_junction.is_active(1.0)

    # now near junction_2 - should arm
    attack_on_second_junction.check_trigger(t=2.0, vehicle_s=385.0, track=track)
    assert attack_on_second_junction.is_active(2.0)

def test_track_falls_back_to_default_features_when_none_configured():
    """A Track built with no explicit features should still get a sensible
    default junction and roundabout."""
    track = Track()
    assert track.feature("junction_1").feature_type == "junction"
    assert track.feature("roundabout_1").feature_type == "roundabout"

def test_vehicle_with_no_real_vehicle_ahead_is_not_confused_by_one_behind():
    """Regression test for a bug: the front vehicle in a two-vehicle
    scenario has no vehicle ahead of it, but the other vehicle (behind
    it) still has some forward wraparound distance to reach it (nearly
    a full lap). Without MAX_SENSOR_RANGE filtering this out, the front
    vehicle would believe an obstacle is present hundreds of metres away,
    which blocks re-acceleration forever even though no braking ever
    actually triggers (the distance is far too large for that)."""
    import yaml
    scenario_dict = {
        "name": "front vehicle resume test",
        "duration": 5,
        "timestep": 0.1,
        "vehicles": [
            {"vehicle_id": "front", "start_distance": 20, "start_speed_mph": 20, "fusion_policy": "camera_priority"},
            {"vehicle_id": "back", "start_distance": 0, "start_speed_mph": 20, "fusion_policy": "camera_priority"},
        ],
    }
    scenario_path = os.path.join(tempfile.gettempdir(), "_front_vehicle_test.yaml")
    with open(scenario_path, "w") as f:
        yaml.dump(scenario_dict, f)

    config = load_scenario(scenario_path)
    sim = Simulation(config)
    log = sim.run()
    front_log = [e for e in log if e.vehicle_id == "front"]

    assert all(not e.fused_belief.obstacle_present for e in front_log), \
        "the front vehicle has nothing ahead of it"

def test_trigger_after_feature_does_not_fire_before_the_vehicle_reaches_it():
    """The specific bug this trigger type has to avoid: a vehicle that
    hasn't reached the feature yet must NOT be treated as if it were
    already some distance past it. If this used the same wraparound
    distance_ahead() helper as trigger_before_feature, a vehicle at
    vehicle_s=0 approaching a feature at position=300 on an ~800m track
    would appear to be ~500m "past" it - comfortably past any realistic
    trigger_distance - and the attack would incorrectly arm at t=0,
    before the vehicle has moved at all"""
    from sim_core.attacks import CameraPhantom
    from sim_core.track import TrackFeature

    track = Track(features=[TrackFeature("junction_1", "junction", 300.0)])
    attack = CameraPhantom(phantom_distance=3.0, duration=0.5,
                            trigger_after_feature="junction_1", trigger_distance=50)

    attack.check_trigger(t=0.0, vehicle_s=0.0, track=track)  # nowhere near the feature yet
    assert not attack.is_active(0.0)

def test_trigger_after_feature_arms_once_the_vehicle_has_passed_by_enough():
    from sim_core.attacks import CameraPhantom
    from sim_core.track import TrackFeature

    track = Track(features=[TrackFeature("junction_1", "junction", 300.0)])
    attack = CameraPhantom(phantom_distance=3.0, duration=0.5,
                            trigger_after_feature="junction_1", trigger_distance=50)

    attack.check_trigger(t=1.0, vehicle_s=320.0, track=track)  # only 20m past -- not yet
    assert not attack.is_active(1.0)

    attack.check_trigger(t=2.0, vehicle_s=355.0, track=track)  # 55m past -- should arm
    assert attack.is_active(2.0)

def test_phantom_brake_scenario_triggers_measurable_braking():
    """End-to-end test: load the example scenario, run it, and check
    the attacked vehicle's speed measurably drops shortly after the
    phantom attack fires. Deliberately checks only the few ticks right
    around the attack, not the scenario's later behaviour - once the
    vehicle also approaches the real junction later in the same run, it
    additionally brakes for the red light and for the lead vehicle also
    slowing for it"""
    config = load_scenario("scenarios/phantom_brake.yaml")
    sim = Simulation(config)
    log = sim.run()

    follower_log = [e for e in log if e.vehicle_id == "follower"]
    speeds = [e.speed for e in follower_log]
    assert 20.0 < speeds[0] < 20.2, "should start at its configured cruising speed (45 mph)"

    attack_tick = next(i for i, e in enumerate(follower_log) if e.fused_belief.distance_to_obstacle == 3.0)
    speed_just_before_attack = follower_log[attack_tick - 1].speed
    speed_shortly_after_attack_starts = follower_log[attack_tick + 3].speed
    assert speed_shortly_after_attack_starts < speed_just_before_attack, \
        "expected the phantom camera attack to cause measurable braking shortly after it fires"

def test_a_detected_but_harmless_obstacle_does_not_block_resuming():
    """A vehicle that detects something ahead, but far enough away that
    required_deceleration is genuinely 0, must still be free to resume
    toward cruise speed - 'something was detected' and 'I need to brake for it'
    are not the same condition, and conflating them meant a vehicle that had
    dipped below its cruise speed once could get permanently stuck there, as long
    as anything at all remained within sensor range, however harmlessly far away

    To create a dip to recover from, the follower starts close
    enough behind a faster-moving lead to require brief initial braking;
    since the lead is faster, the gap opens quickly, braking stops being
    required, and the follower should then be free to climb back toward
    its own cruise speed"""
    import yaml
    scenario_dict = {
        "name": "harmless detection resume test",
        "duration": 10,
        "timestep": 0.1,
        "vehicles": [
            {"vehicle_id": "follower", "start_distance": 0, "start_speed_mph": 20, "fusion_policy": "camera_priority"},
            {"vehicle_id": "lead", "start_distance": 10, "start_speed_mph": 30, "fusion_policy": "camera_priority"},
        ],
    }
    scenario_path = os.path.join(tempfile.gettempdir(), "_harmless_detection_test.yaml")
    with open(scenario_path, "w") as f:
        yaml.dump(scenario_dict, f)

    config = load_scenario(scenario_path)
    sim = Simulation(config)
    log = sim.run()
    follower_log = [e for e in log if e.vehicle_id == "follower"]
    speeds = [e.speed for e in follower_log]

    min_speed = min(speeds)
    assert min_speed < speeds[0], "expected the follower to brake initially, given the close starting gap"
    assert speeds[-1] > min_speed, \
        "expected the follower to recover and accelerate again once the lead stopped requiring braking"

def test_phantom_attack_can_fabricate_a_detection_from_nothing():
    """A camera phantom attack against a vehicle with genuinely nothing
    else ahead of it (no other vehicle, hazard, or red light) must still succeed.
    The whole point of a phantom attack is fabricating a detection where ground
    truth had none - if the engine only applies attacks to sensors that already
    have a reading, a phantom can never appear on an otherwise-clear
    road, which defeats the purpose of the attack entirely"""
    import yaml
    scenario_dict = {
        "name": "phantom from nothing test",
        "duration": 2,
        "timestep": 0.1,
        "vehicles": [
            {"vehicle_id": "solo", "start_distance": 0, "start_speed_mph": 20, "fusion_policy": "camera_priority"},
        ],
        "attacks": [
            {"type": "camera_phantom", "target_vehicle": "solo", "start_time": 0.5, "duration": 0.5, "phantom_distance": 3.0},
        ],
    }
    scenario_path = os.path.join(tempfile.gettempdir(), "_phantom_from_nothing_test.yaml")
    with open(scenario_path, "w") as f:
        yaml.dump(scenario_dict, f)

    config = load_scenario(scenario_path)
    sim = Simulation(config)
    log = sim.run()
    solo_log = [e for e in log if e.vehicle_id == "solo"]

    speed_before_attack = next(e.speed for e in solo_log if e.time < 0.5)
    speed_during_attack = next(e.speed for e in solo_log if 0.9 < e.time < 1.0)
    assert speed_during_attack < speed_before_attack, \
        "expected the phantom attack to cause braking even with nothing else detected"