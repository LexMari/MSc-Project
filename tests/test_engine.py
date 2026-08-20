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
    """Model should match stopping distances in highway code"""
    assert abs(safe_stopping_distance(mph_to_ms(30)) - 23) < 0.5
    assert abs(safe_stopping_distance(mph_to_ms(60)) - 73) < 1.0

def test_required_deceleration_is_zero_with_plenty_of_room():
    assert required_deceleration(speed=20.0, distance_to_obstacle=500.0) == 0.0

def test_required_deceleration_scales_with_how_close_obstacle_is():
    """A closer obstacle should demand harder braking than a further one at the same speed"""
    speed = mph_to_ms(45)
    harder = required_deceleration(speed, distance_to_obstacle=5.0)
    softer = required_deceleration(speed, distance_to_obstacle=40.0)
    assert harder > softer

def test_required_deceleration_caps_at_maximum_when_stopping_is_impossible():
    """If obstacle is closer than reaction-time distance alone, and maximum
    braking is still not enough it should still cap at MAX_DECELERATION"""
    from sim_core.braking import MAX_DECELERATION
    speed = mph_to_ms(70)
    assert required_deceleration(speed, distance_to_obstacle=1.0) == MAX_DECELERATION

def test_safety_margin_forces_maximum_braking_at_the_margin_boundary():
    """A reported obstacle exactly SAFETY_MARGIN metres away should have maximum braking"""
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
    """A real hazard should be detected consistently regardless of fusion policy"""
    for policy in ("camera_priority", "majority_vote", "confidence_weighted"):
        config = load_scenario("scenarios/pedestrian_crossing_test.yaml")
        config.vehicles[0].fusion_policy = policy
        sim = Simulation(config)
        log = sim.run()

        speeds = [e.speed for e in log if e.vehicle_id == config.vehicles[0].vehicle_id]
        assert min(speeds) < speeds[0], \
            f"expected braking for a pedestrian hazard under {policy} fusion"

def test_mph_conversion_round_trips():
    original_mph = 45.0
    assert abs(ms_to_mph(mph_to_ms(original_mph)) - original_mph) < 1e-9

def test_scenario_speed_is_converted_from_mph_to_ms():
    """45 mph should load as roughly 20.1 m/s, not 45 m/s"""
    config = load_scenario("scenarios/phantom_brake.yaml")
    follower = next(v for v in config.vehicles if v.vehicle_id == "follower")
    assert 20.0 < follower.start_speed < 20.2

def test_vehicle_resumes_toward_cruise_speed_after_braking():
    """Once no obstacle is believed present, a vehicle below its cruise speed should speed
    back up, not just hold at whatever reduced speed braking left it at"""
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
    assert final_speed > min_speed, "expected vehicle to speed back up after hazard cleared"

def test_red_light_is_treated_as_obstacle_but_green_is_not():
    """A junction's traffic light should act as an obstacle at its stop line while red/amber"""
    light = TrafficLight(red_duration=8.0, green_duration=6.0, amber_duration=2.0)
    assert light.state_at(0.0) == "green"    # should be treated as if it weren't there
    assert light.state_at(6.0) == "amber"    # should be treated as an obstacle
    assert light.state_at(8.0) == "red"      # should be treated as an obstacle

def test_vehicle_stops_at_red_light_and_resumes_at_green():
    """A vehicle approaching a junction should brake for a red light, then resume toward
    cruise speed once it turns green"""
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
    assert car_log[-1].speed > min_speed, "expected the vehicle to resume once light turned green"

def test_vehicle_moves_forward_along_track():
    v = VehicleState("test", s=0.0, speed=10.0, cruise_speed=10.0)
    v.step(dt=1.0)
    assert v.s == 10.0

def test_vehicle_speed_does_not_go_negative():
    v = VehicleState("test", s=0.0, speed=1.0, cruise_speed=1.0, acceleration=-10.0)
    v.step(dt=1.0)
    assert v.speed == 0.0

def test_track_position_is_continuous_across_segment_boundaries():
    """The four segments need to join up without discontinuities"""
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
    assert light.state_at(0.0) == "green"
    assert light.state_at(6.0) == "amber"
    assert light.state_at(8.0) == "red"
    assert light.state_at(16.0) == "green"

def test_camera_priority_trusts_camera_over_missing_radar():
    readings = {SensorType.CAMERA: SensorReading(SensorType.CAMERA, detected_distance=5.0)}
    belief = CameraPriorityFusion().fuse(readings)
    assert belief.obstacle_present
    assert belief.distance_to_obstacle == 5.0

def test_majority_vote_rejects_lone_spoofed_sensor():
    """a radar spoof should not fool majority vote fusion"""
    readings = {
        SensorType.RADAR: SensorReading(SensorType.RADAR, detected_distance=3.0, is_attacked=True),
        SensorType.CAMERA: SensorReading(SensorType.CAMERA, detected_distance=40.0),
    }
    belief = MajorityVoteFusion().fuse(readings)
    assert not belief.obstacle_present

def test_camera_priority_ignores_radar_only_attack():
    """attack on a sensor the fusion policy doesn't consult should have zero effect
    attack success depends on the fusion policy"""
    readings = {
        SensorType.CAMERA: SensorReading(SensorType.CAMERA, detected_distance=40.0),
        SensorType.RADAR: SensorReading(SensorType.RADAR, detected_distance=3.0, is_attacked=True),
    }
    belief = CameraPriorityFusion().fuse(readings)
    assert belief.distance_to_obstacle == 40.0

def test_position_triggered_attack_arms_only_near_named_feature():
    """an attack configured to fire 50m before a named feature should stay dormant far from it
    and arm the moment the vehicle comes within range"""
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
    """A track with two junctions should let an attack be timed against either one"""
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

    # near junction_1 (s=90), but the attack is armed for junction_2 (at s=400) - should not arm
    attack_on_second_junction.check_trigger(t=1.0, vehicle_s=90.0, track=track)
    assert not attack_on_second_junction.is_active(1.0)

    # now near junction_2 - should arm
    attack_on_second_junction.check_trigger(t=2.0, vehicle_s=385.0, track=track)
    assert attack_on_second_junction.is_active(2.0)

def test_track_falls_back_to_default_features_when_none_configured():
    """A track built without features should still get the default junction and roundabout"""
    track = Track()
    assert track.feature("junction_1").feature_type == "junction"
    assert track.feature("roundabout_1").feature_type == "roundabout"

def test_vehicle_with_no_vehicle_ahead_is_not_confused_by_one_behind():
    """Regression test - check if a vehicle in front does not register a vehicle
    that is behind it"""
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
        "front vehicle has nothing ahead of it"

def test_trigger_after_feature_does_not_fire_before_vehicle_reaches_it():
    """a vehicle that has not reached the feature yet must not be treated as if it were
    already some distance past it"""
    from sim_core.attacks import CameraPhantom
    from sim_core.track import TrackFeature

    track = Track(features=[TrackFeature("junction_1", "junction", 300.0)])
    attack = CameraPhantom(phantom_distance=3.0, duration=0.5,
                            trigger_after_feature="junction_1", trigger_distance=50)

    attack.check_trigger(t=0.0, vehicle_s=0.0, track=track)
    assert not attack.is_active(0.0)

def test_trigger_after_feature_arms_once_vehicle_has_passed_by_enough():
    from sim_core.attacks import CameraPhantom
    from sim_core.track import TrackFeature

    track = Track(features=[TrackFeature("junction_1", "junction", 300.0)])
    attack = CameraPhantom(phantom_distance=3.0, duration=0.5,
                            trigger_after_feature="junction_1", trigger_distance=50)

    attack.check_trigger(t=1.0, vehicle_s=320.0, track=track) # 20m past - too soon
    assert not attack.is_active(1.0)

    attack.check_trigger(t=2.0, vehicle_s=355.0, track=track)  # 55m past - should arm
    assert attack.is_active(2.0)

def test_phantom_brake_scenario_triggers_braking():
    """load scenario, run it, and check attacked vehicle's speed drops after the phantom attack fires"""
    config = load_scenario("scenarios/phantom_brake.yaml")
    sim = Simulation(config)
    log = sim.run()

    follower_log = [e for e in log if e.vehicle_id == "follower"]
    speeds = [e.speed for e in follower_log]
    assert 20.0 < speeds[0] < 20.2, "should start at its configured cruising speed"

    attack_tick = next(i for i, e in enumerate(follower_log) if e.fused_belief.distance_to_obstacle == 55.0)
    speed_just_before_attack = follower_log[attack_tick - 1].speed
    speed_shortly_after_attack_starts = follower_log[attack_tick + 3].speed
    assert speed_shortly_after_attack_starts < speed_just_before_attack, \
        "expected the phantom camera attack to cause significant braking shortly after firing"

def test_a_detected_harmless_obstacle_does_not_block_resuming():
    """tests if a vehicle resumes after detecting a harmless obstacle"""
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
    assert min_speed < speeds[0], "expected the follower to brake given close starting gap"
    assert speeds[-1] > min_speed, \
        "expected follower to accelerate again once lead stopped requiring braking"

def test_phantom_attack_can_fabricate_a_detection():
    """Tests if a phantom attack can appear unprompted"""
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
        "expected phantom attack to cause braking"

def _obstacle_swerve_scenario_dict(solo_mph: float, oncoming_mph: float, reveal_distance: float = 30.0) -> dict:
    from sim_core.units import mph_to_ms
    obstacle_s = 150.0
    reveal_time = (obstacle_s - reveal_distance) / mph_to_ms(solo_mph)
    return {
        "name": "obstacle swerve test",
        "duration": 20,
        "timestep": 0.1,
        "track": {
            "straight_length": 200, "radius": 60,
            "features": [{"feature_id": "obstacle_1", "feature_type": "obstacle_marker", "position": obstacle_s}],
        },
        "vehicles": [
            {"vehicle_id": "solo", "start_distance": 0, "start_speed_mph": solo_mph,
             "fusion_policy": "camera_priority", "lane": 0, "direction": 1},
            {"vehicle_id": "oncoming", "start_distance": 250, "start_speed_mph": oncoming_mph,
             "fusion_policy": "camera_priority", "lane": 1, "direction": -1},
        ],
        "hazards": [
            {"type": "obstacle_in_road", "feature_id": "obstacle_1",
             "start_time": reveal_time, "duration": 10000, "lane": 0},
        ],
    }

def _run_obstacle_swerve(solo_mph: float, oncoming_mph: float, reveal_distance: float = 30.0):
    import yaml
    scenario_path = os.path.join(tempfile.gettempdir(), "_obstacle_swerve_test.yaml")
    with open(scenario_path, "w") as f:
        yaml.dump(_obstacle_swerve_scenario_dict(solo_mph, oncoming_mph, reveal_distance), f)
    config = load_scenario(scenario_path)
    sim = Simulation(config)

    steps = int(config.duration / config.timestep)
    swerved, collided, min_gap = False, False, None
    for _ in range(steps):
        results = sim.step(config.timestep)
        by_id = {r.vehicle_id: r for r in results}
        solo, onc = by_id["solo"], by_id["oncoming"]
        if solo.lane == 1:
            swerved = True
        gap = ((solo.position[0] - onc.position[0]) ** 2 + (solo.position[1] - onc.position[1]) ** 2) ** 0.5
        if min_gap is None or gap < min_gap:
            min_gap = gap
        if solo.collision or onc.collision:
            collided = True
    return swerved, collided, min_gap

def test_vehicle_swerves_when_it_cannot_stop_in_time():
    """vehicle should swerve into the next lane if it can't stop in time"""
    swerved, _collided, _min_gap = _run_obstacle_swerve(solo_mph=44.74, oncoming_mph=20.0)
    assert swerved

def test_vehicle_declines_unsafe_swerve():
    """checks if a vehicle declines an unsafe swerve"""
    swerved, collided, min_gap_to_oncoming = _run_obstacle_swerve(solo_mph=44.74, oncoming_mph=33.55)
    assert not swerved, "expected vehicle to correctly decline a swerve"
    assert collided, "expected a collision with the original obstacle"
    assert min_gap_to_oncoming > 3.0, "expected oncoming traffic to never be approached"

def test_swerve_can_pass_with_slower_oncoming_vehicle():
    """checks if a vehicle can swerve if the gap is large enough"""
    swerved, collided, min_gap = _run_obstacle_swerve(solo_mph=44.74, oncoming_mph=20.0)
    assert swerved
    assert not collided
    assert min_gap > 3.0

def test_vehicle_does_not_swerve_if_it_can_brake_to_a_safe_stop():
    """vehicle should brake if it has enough room to successfully"""
    swerved, _collided, _min_gap = _run_obstacle_swerve(solo_mph=20.0, oncoming_mph=20.0, reveal_distance=150.0)
    assert not swerved

def test_crashed_vehicle_stops_properly():
    """a collision should disable a vehicle from moving further"""
    config = load_scenario(os.path.join("scenarios", "radar_spoof_masking.yaml"))
    config.vehicles[0].fusion_policy = "confidence_weighted"
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)

    crashed_at = None
    for i in range(steps):
        results = sim.step(config.timestep)
        t = round(sim.time - config.timestep, 2)
        solo = next(r for r in results if r.vehicle_id == "solo")
        if solo.collision and crashed_at is None:
            crashed_at = t
        elif crashed_at is not None and t > crashed_at:
            assert solo.speed < 0.5, \
                "expected a crashed vehicle to stop immediately"
            assert solo.speed == 0.0 or t < crashed_at + 0.5, \
                "expected speed to reach zero after the collision and stay there"
            if t > crashed_at + 2.0:
                assert solo.speed == 0.0
                return

    assert crashed_at is not None, "expected scenario to produce a collision... (what happened bro)"

def test_opposite_direction_vehicle_is_sensed_through_crossing_point():
    """two vehicles should still be aware of each other (this is for testing swerve suitability)"""
    import yaml
    scenario_dict = {
        "name": "crossing point test", "duration": 10, "timestep": 0.1,
        "track": {"straight_length": 200, "radius": 60, "features": []},
        "vehicles": [
            {"vehicle_id": "solo", "start_distance": 0, "start_speed_mph": 30, "fusion_policy": "camera_priority", "lane": 1, "direction": 1},
            {"vehicle_id": "onc", "start_distance": 20, "start_speed_mph": 30, "fusion_policy": "camera_priority", "lane": 1, "direction": -1},
        ],
    }
    path = os.path.join(tempfile.gettempdir(), "_crossing_point_test.yaml")
    with open(path, "w") as f:
        yaml.dump(scenario_dict, f)
    config = load_scenario(path)
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)

    for _ in range(steps):
        results = sim.step(config.timestep)
        solo = sim.vehicles["solo"]
        onc = sim.vehicles["onc"]
        gap = abs(solo.s - onc.s)
        solo_result = next(r for r in results if r.vehicle_id == "solo")
        if gap < 5.0:
            belief = solo_result.fused_belief
            assert belief.obstacle_present, \
                "expected oncoming vehicle to be sensed at close range"
            return

    raise AssertionError("expected solo and onc to reach a close gap in this scenario")

def _run_gps_roundabout_scenario(policy: str, offset: tuple[float, float]) -> tuple[bool, float]:
    """builds and runs a GPS-spoof scenario for given policy and offset"""
    import yaml
    scenario_dict = {
        "name": "gps roundabout test", "duration": 20, "timestep": 0.1,
        "vehicles": [
            {"vehicle_id": "solo", "start_distance": 0, "start_speed_mph": 44.74,
             "fusion_policy": "camera_priority", "gps_policy": policy},
        ],
        "attacks": [
            {"type": "gps_spoof", "target_vehicle": "solo",
             "trigger_before_feature": "roundabout_1", "trigger_distance": 30,
             "duration": 5.0, "offset": list(offset)},
        ],
    }
    scenario_path = os.path.join(tempfile.gettempdir(), "_gps_roundabout_test.yaml")
    with open(scenario_path, "w") as f:
        yaml.dump(scenario_dict, f)
    config = load_scenario(scenario_path)
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    confused = False
    for _ in range(steps):
        results = sim.step(config.timestep)
        if results[0].roundabout_confused:
            confused = True
    return confused, sim.vehicles["solo"].s

def test_naive_gps_policy_is_confused_by_a_spoof_at_roundabout():
    confused, final_s = _run_gps_roundabout_scenario("naive", (100.0, 100.0))
    assert confused

def test_plausibility_checked_gps_policy_resists_same_spoof():
    """identical attack should not confuse the vehicle"""
    confused, _final_s = _run_gps_roundabout_scenario("plausibility_checked", (100.0, 100.0))
    assert not confused

def test_small_gps_spoof_does_not_confuse_either_policy():
    """small offset should not confuse either policy"""
    for policy in ("naive", "plausibility_checked"):
        confused, _final_s = _run_gps_roundabout_scenario(policy, (3.0, 3.0))
        assert not confused, f"expected small spoof not to confuse the {policy} policy"

def test_roundabout_confusion_does_not_teleport_cars():
    """tests if the car takes an extra lap around the roundabout instead of teleporting it
    to the spoofed offset"""
    from sim_core.track import Track
    track = Track()
    circumference = track.feature_circumference("roundabout_1")

    import yaml
    scenario_dict = {
        "name": "gps roundabout timing test", "duration": 20, "timestep": 0.1,
        "vehicles": [
            {"vehicle_id": "solo", "start_distance": 0, "start_speed_mph": 44.74,
             "fusion_policy": "camera_priority", "gps_policy": "naive"},
        ],
        "attacks": [
            {"type": "gps_spoof", "target_vehicle": "solo",
             "trigger_before_feature": "roundabout_1", "trigger_distance": 30,
             "duration": 5.0, "offset": [100.0, 100.0]},
        ],
    }
    scenario_path = os.path.join(tempfile.gettempdir(), "_gps_timing_test.yaml")
    with open(scenario_path, "w") as f:
        yaml.dump(scenario_dict, f)
    config = load_scenario(scenario_path)
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)

    confused_at_t = None
    excursion_cleared_at_t = None
    s_at_confusion = None
    s_at_cleared = None
    for _ in range(steps):
        t_before = round(sim.time, 2)
        results = sim.step(config.timestep)
        if results[0].roundabout_confused:
            confused_at_t = t_before
            s_at_confusion = sim.vehicles["solo"].s
        if confused_at_t is not None and excursion_cleared_at_t is None:
            if sim.vehicles["solo"].roundabout_excursion_remaining <= 0:
                excursion_cleared_at_t = round(sim.time, 2)
                s_at_cleared = sim.vehicles["solo"].s
                break

    assert confused_at_t is not None, "expected scenario to trigger confusion"
    assert excursion_cleared_at_t is not None, "expected excursion to end"

    elapsed = excursion_cleared_at_t - confused_at_t
    expected_elapsed = circumference / mph_to_ms(44.74)
    assert elapsed > 1.0, \
        f"expected extra lap to take multiple seconds, not be instant - (took {elapsed:.2f}s)"
    assert abs(elapsed - expected_elapsed) < 1.0, \
        f"expected roughly ({expected_elapsed:.2f}s) of extra driving, got {elapsed:.2f}s"

    total_s_change = s_at_cleared - s_at_confusion
    assert abs(total_s_change) < 5.0, \
        (f"expected s to stay still while doing extra lap")

def test_gps_plausibility_check_rejects_large_jumps_and_accepts_small_ones():
    """tests plausibility checked GPS works as intended"""
    from sim_core.navigation import PlausibilityCheckedGPSPolicy
    from sim_core.sensors import SensorReading, SensorType

    vehicle = VehicleState(vehicle_id="solo", s=0.0, speed=20.0, cruise_speed=20.0)
    policy = PlausibilityCheckedGPSPolicy()

    first = SensorReading(SensorType.GPS, detected_position=(0.0, 0.0))
    believed = policy.resolve(first, vehicle, dt=0.1)
    assert believed == (0.0, 0.0)

    plausible = SensorReading(SensorType.GPS, detected_position=(2.0, 0.0))
    believed = policy.resolve(plausible, vehicle, dt=0.1)
    assert believed == (2.0, 0.0), "small jump should be accepted"

    implausible = SensorReading(SensorType.GPS, detected_position=(500.0, 500.0), is_attacked=True)
    believed = policy.resolve(implausible, vehicle, dt=0.1)
    assert believed == (2.0, 0.0), "large jump should be rejected, holding last known good position"

def test_radar_spoof_masking_is_ignored_by_camera_priority():
    """camera priority should never acknowledge a radar spoof"""
    config = load_scenario("scenarios/radar_spoof_masking.yaml")
    config.vehicles[0].fusion_policy = "camera_priority"
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    min_speed = min(sim.step(config.timestep)[0].speed for _ in range(steps))
    assert min_speed < 1.0, "expected camera_priority to brake despite radar spoof"

def _braking_start_time(reaction_time, max_deceleration):
    from sim_core.units import ms_to_mph
    config = load_scenario(os.path.join("scenarios", "pedestrian_crossing_test.yaml"))
    config.vehicles[0].fusion_policy = "camera_priority"
    if reaction_time is not None:
        config.vehicles[0].reaction_time = reaction_time
        config.vehicles[0].max_deceleration = max_deceleration
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    cruise_mph = ms_to_mph(config.vehicles[0].start_speed)
    for _ in range(steps):
        t_before = round(sim.time, 2)
        results = sim.step(config.timestep)
        if ms_to_mph(results[0].speed) < cruise_mph - 0.5:
            return t_before
    return None

def test_driver_diversity_produces_different_braking_onsets():
    """aggressive driver should brake later, cautious one should brake earlier"""
    aggressive_start = _braking_start_time(0.45, 8.0)
    default_start = _braking_start_time(None, None)
    cautious_start = _braking_start_time(0.95, 5.0)
    assert cautious_start < default_start < aggressive_start

def test_braking_variation_is_reproducible_with_same_seed():
    def sampled_params(seed):
        import yaml
        scenario_dict = {
            "name": "diversity reproducibility test", "duration": 5, "timestep": 0.1,
            "random_seed": seed,
            "vehicles": [
                {"vehicle_id": "solo", "start_distance": 0, "start_speed_mph": 30,
                 "fusion_policy": "camera_priority", "braking_variation": 0.2},
            ],
        }
        path = os.path.join(tempfile.gettempdir(), "_diversity_repro_test.yaml")
        with open(path, "w") as f:
            yaml.dump(scenario_dict, f)
        config = load_scenario(path)
        sim = Simulation(config)
        v = sim.vehicles["solo"]
        return v.reaction_time, v.max_deceleration

    a1 = sampled_params(seed=42)
    a2 = sampled_params(seed=42)
    b = sampled_params(seed=7)
    assert a1 == a2
    assert a1 != b

def _pedestrian_braking_start(pedestrian_type):
    from sim_core.units import ms_to_mph
    config = load_scenario(os.path.join("scenarios", "pedestrian_crossing_test.yaml"))
    config.vehicles[0].fusion_policy = "camera_priority"
    config.hazards[0]["pedestrian_type"] = pedestrian_type
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    cruise_mph = ms_to_mph(config.vehicles[0].start_speed)
    for _ in range(steps):
        t_before = round(sim.time, 2)
        results = sim.step(config.timestep)
        if ms_to_mph(results[0].speed) < cruise_mph - 0.5:
            return t_before
    return None

def test_pedestrian_type_affects_how_early_braking_begins():
    """vehicle should brake earliest for child, then elderly, then adult"""
    child_start = _pedestrian_braking_start("child")
    adult_start = _pedestrian_braking_start("adult")
    elderly_start = _pedestrian_braking_start("elderly")
    assert child_start < elderly_start < adult_start

def test_pedestrian_crossing_random_start_time_is_reproducible_per_seed():
    """should be random per seed but reproducible start time for said seed"""
    config = load_scenario(os.path.join("scenarios", "pedestrian_crossing_test.yaml"))
    config.hazards[0]["start_time"] = None

    config.random_seed = 99
    sim1 = Simulation(config)
    sim2 = Simulation(config)
    assert sim1.hazards[0].start_time == sim2.hazards[0].start_time

    config.random_seed = 100
    sim3 = Simulation(config)
    assert sim3.hazards[0].start_time != sim1.hazards[0].start_time

    assert 0.0 <= sim1.hazards[0].start_time <= config.duration - config.hazards[0]["duration"]

def test_vehicle_stops_near_persistent_hazard_instead_of_creeping():
    """bug test - vehicles would oscillate in speed every other tick"""
    import yaml
    scenario_dict = {
        "name": "firm stop unit test", "duration": 30, "timestep": 0.1,
        "track": {"straight_length": 200, "radius": 60,
                  "features": [{"feature_id": "crossing_1", "feature_type": "pedestrian_crossing", "position": 150}]},
        "vehicles": [{"vehicle_id": "solo", "start_distance": 0, "start_speed_mph": 44.74, "fusion_policy": "camera_priority"}],
        "hazards": [{"type": "pedestrian_crossing", "feature_id": "crossing_1", "start_time": 0.0, "duration": 60.0}],
    }
    path = os.path.join(tempfile.gettempdir(), "_firm_stop_test.yaml")
    with open(path, "w") as f:
        yaml.dump(scenario_dict, f)
    config = load_scenario(path)
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)

    stopped_at = None
    distance_when_stopped = None
    for _ in range(steps):
        t = round(sim.time, 2)
        results = sim.step(config.timestep)
        solo = results[0]
        if stopped_at is None and solo.speed == 0.0:
            stopped_at = t
            distance_when_stopped = solo.fused_belief.distance_to_obstacle
        elif stopped_at is not None and t > stopped_at + 3.0:
            assert solo.speed < 0.1
            assert distance_when_stopped - solo.fused_belief.distance_to_obstacle < 0.1
            return

    assert stopped_at is not None, "expected the vehicle to reach stop in this scenario"

def test_radar_and_lidar_spoof_defeats_majority_vote():
    """spoofed majority is still a majority"""
    from sim_core.units import ms_to_mph
    config = load_scenario("scenarios/multi_sensor_spoof.yaml")
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)

    collided = False
    min_speed_before_collision = None
    for _ in range(steps):
        results = sim.step(config.timestep)
        solo = results[0]
        if solo.collision:
            collided = True
            break
        if min_speed_before_collision is None or solo.speed < min_speed_before_collision:
            min_speed_before_collision = solo.speed

    assert collided, "expected the vehicle to collide with the pedestrian"
    assert ms_to_mph(min_speed_before_collision) > 15.0, \
        "did not expect braking before the collision"

def test_pedestrian_collision_is_detected():
    """a vehicle that collides with a pedestrian should be detected"""
    config = load_scenario("scenarios/multi_sensor_spoof.yaml")
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    collided = any(sim.step(config.timestep)[0].collision for _ in range(steps))
    assert collided

def test_camera_jam_defeats_camera_priority():
    """camera jam should defeat camera priority"""
    config = load_scenario("scenarios/camera_jam_defeats_camera_priority.yaml")
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    collided = any(sim.step(config.timestep)[0].collision for _ in range(steps))
    assert collided

def test_camera_jam_does_not_defeat_majority_vote_or_confidence_weighted():
    """both policies should withstand a single sensor being jammed"""
    for policy in ("majority_vote", "confidence_weighted"):
        config = load_scenario("scenarios/camera_jam_defeats_camera_priority.yaml")
        config.vehicles[0].fusion_policy = policy
        sim = Simulation(config)
        steps = int(config.duration / config.timestep)
        collided = any(sim.step(config.timestep)[0].collision for _ in range(steps))
        assert not collided, f"expected {policy} to correctly fall back to radar/LiDAR and avoid a collision"

def test_vehicle_resumes_once_pedestrian_has_walked_clear_of_lane():
    """vehicle should stop for pedestrian and continue once it is passed"""
    import yaml
    from sim_core.units import ms_to_mph
    scenario_dict = {
        "name": "pedestrian walks clear test", "duration": 20, "timestep": 0.1,
        "track": {"straight_length": 200, "radius": 60,
                  "features": [{"feature_id": "crossing_1", "feature_type": "pedestrian_crossing", "position": 150}]},
        "vehicles": [{"vehicle_id": "solo", "start_distance": 0, "start_speed_mph": 44.74, "fusion_policy": "camera_priority"}],
        "hazards": [{"type": "pedestrian_crossing", "feature_id": "crossing_1", "start_time": 0.0, "duration": 20.0}],
    }
    path = os.path.join(tempfile.gettempdir(), "_pedestrian_resume_test.yaml")
    with open(path, "w") as f:
        yaml.dump(scenario_dict, f)
    config = load_scenario(path)
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)

    braked = False
    resumed = False
    for _ in range(steps):
        r = sim.step(config.timestep)[0]
        if ms_to_mph(r.speed) < 5.0:
            braked = True
        if braked and ms_to_mph(r.speed) > 20.0:
            resumed = True
            break

    assert braked, "expected vehicle to brake for pedestrian while in its lane"
    assert resumed, "expected vehicle to resume once pedestrian walked clear of its lane"

def test_low_visibility_can_cause_collision():
    """tests if visibility works as intended"""
    config = load_scenario("scenarios/low_visibility_collision.yaml")
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    collided = any(sim.step(config.timestep)[0].collision for _ in range(steps))
    assert collided

def test_visibility_defaults_to_clear():
    """unset visibiltiy should default to 100%"""
    config = load_scenario("scenarios/pedestrian_crossing_test.yaml")
    assert config.visibility == 1.0

def test_visibility_scales_detection_distance():
    """lower visibility should correlate to lower detection distance"""
    import yaml
    def make_scenario(visibility):
        return {
            "name": "visibility scaling test", "duration": 20, "timestep": 0.1, "visibility": visibility,
            "track": {"straight_length": 200, "radius": 60,
                      "features": [{"feature_id": "crossing_1", "feature_type": "pedestrian_crossing", "position": 150}]},
            "vehicles": [{"vehicle_id": "solo", "start_distance": 0, "start_speed_mph": 44.74, "fusion_policy": "camera_priority"}],
            "hazards": [{"type": "pedestrian_crossing", "feature_id": "crossing_1", "start_time": 0.0, "duration": 20.0}],
        }
    path = os.path.join(tempfile.gettempdir(), "_visibility_scaling_test.yaml")
    with open(path, "w") as f:
        yaml.dump(make_scenario(0.5), f)
    config = load_scenario(path)
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    first_distance = None
    for _ in range(steps):
        r = sim.step(config.timestep)[0]
        if first_distance is None and r.fused_belief.obstacle_present:
            first_distance = r.fused_belief.distance_to_obstacle
    assert abs(first_distance - 75.0) < 2.0  # 150m * 0.5 visibility

def test_spawner_aligns_spawns_with_traffic_light_cycle():
    """new vehicle should spawn at the correct point in a cycle (the end)"""
    config = load_scenario("scenarios/background_traffic.yaml")
    sim = Simulation(config)
    cycle_length = sim.traffic_lights["junction_1"].cycle_length
    steps = int(config.duration / config.timestep)

    prev_ids = set(sim.vehicles.keys())
    spawn_times = []
    for _ in range(steps):
        sim.step(config.timestep)
        t = round(sim.time, 2)
        current_ids = set(sim.vehicles.keys())
        for v in current_ids - prev_ids:
            spawn_times.append(t)
        prev_ids = current_ids

    assert len(spawn_times) >= 3, "expected several spawns over scenario's duration"
    for i in range(1, len(spawn_times)):
        gap = spawn_times[i] - spawn_times[i - 1]
        remainder = gap % cycle_length
        assert remainder < 0.5 or remainder > cycle_length - 0.5, \
            f"expected spawns aligned to multiples of {cycle_length}s, got a {gap:.1f}s gap"

def test_spawner_never_exceeds_max_concurrent():
    """number of spawns should never exceed max_concurrent"""
    config = load_scenario("scenarios/background_traffic.yaml")
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    for _ in range(steps):
        sim.step(config.timestep)
        assert len(sim._spawned_vehicle_ids) <= config.spawner.max_concurrent

def test_spawned_vehicle_despawns_after_one_lap():
    """spawned vehicle should despawn once reaching junction again"""
    config = load_scenario("scenarios/background_traffic.yaml")
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)

    prev_ids = set(sim.vehicles.keys())
    first_spawn_id = None
    first_spawn_origin = None
    for _ in range(steps):
        sim.step(config.timestep)
        current_ids = set(sim.vehicles.keys())
        if first_spawn_id is None:
            new = current_ids - prev_ids
            if new:
                first_spawn_id = next(iter(new))
                first_spawn_origin = sim._spawn_origin_s.get(first_spawn_id)
        elif first_spawn_id not in current_ids and first_spawn_id not in prev_ids:
            pass
        elif first_spawn_id in prev_ids and first_spawn_id not in current_ids:
            break
        prev_ids = current_ids

    assert first_spawn_id is not None, "expected at least one vehicle to have spawned"

def test_spawner_produces_both_lane_choices():
    """checks if its actually random"""
    import yaml
    scenario_dict = {
        "name": "spawner variety test", "duration": 400, "timestep": 0.1, "random_seed": 7,
        "vehicles": [{"vehicle_id": "solo", "start_distance": 0, "start_speed_mph": 30, "fusion_policy": "camera_priority"}],
        "spawner": {"feature_id": "junction_1", "max_concurrent": 5, "speed_mph": 30.0},
    }
    path = os.path.join(tempfile.gettempdir(), "_spawner_variety_test.yaml")
    with open(path, "w") as f:
        yaml.dump(scenario_dict, f)
    config = load_scenario(path)
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)

    lanes_seen = set()
    prev_ids = set(sim.vehicles.keys())
    for _ in range(steps):
        sim.step(config.timestep)
        current_ids = set(sim.vehicles.keys())
        for v in current_ids - prev_ids:
            lanes_seen.add(sim.vehicles[v].lane)
        prev_ids = current_ids

    assert lanes_seen == {0, 1}, f"expected both lane choices over many spawns, only saw {lanes_seen}"

def test_ground_truth_distinguishes_traffic_light_from_vehicle():
    """check if vehicle can discern object in front of it (As opposed to general 'thing')"""
    import yaml
    scenario_dict = {
        "name": "ground truth kind test", "duration": 45, "timestep": 0.1,
        "vehicles": [
            {"vehicle_id": "solo", "start_distance": 0, "start_speed_mph": 30, "fusion_policy": "camera_priority"},
        ],
    }
    path = os.path.join(tempfile.gettempdir(), "_ground_truth_kind_test.yaml")
    with open(path, "w") as f:
        yaml.dump(scenario_dict, f)
    config = load_scenario(path)
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    kinds_seen = set()
    for _ in range(steps):
        r = sim.step(config.timestep)[0]
        if r.ground_truth_kind is not None:
            kinds_seen.add(r.ground_truth_kind)
    assert kinds_seen == {"traffic_light"}, \
        f"expected only traffic light to be detected in scenario, got {kinds_seen}"

def test_pedestrian_fatality_risk_matches_rosen_sander():
    """fatality risk should match cited source"""
    from sim_core.severity import pedestrian_fatality_risk
    from sim_core.units import mph_to_ms
    assert abs(pedestrian_fatality_risk(mph_to_ms(30)) - 0.07) < 0.01
    assert abs(pedestrian_fatality_risk(mph_to_ms(40)) - 0.25) < 0.02

def test_vehicle_fatality_risk_matches_richards():
    """fatality risk should match cited source"""
    from sim_core.severity import vehicle_fatality_risk
    from sim_core.units import mph_to_ms
    assert abs(vehicle_fatality_risk(mph_to_ms(30)) - 0.03) < 0.01
    assert abs(vehicle_fatality_risk(mph_to_ms(40)) - 0.17) < 0.02
    assert abs(vehicle_fatality_risk(mph_to_ms(50)) - 0.60) < 0.02

def test_vehicle_fatality_risk_is_higher_than_pedestrian_at_low_speed():
    """checks if pedestrian fatality is higher than vehicle fatality"""
    from sim_core.severity import pedestrian_fatality_risk, vehicle_fatality_risk
    from sim_core.units import mph_to_ms
    for mph in [20, 25, 30]:
        v = mph_to_ms(mph)
        assert pedestrian_fatality_risk(v) > vehicle_fatality_risk(v)

def test_classify_severity_is_reproducible_with_seeded_rng():
    """same seed, same fatal/non-fatal sampling outcome every time"""
    import random
    from sim_core.severity import classify_severity
    results_a = [classify_severity(0.5, random.Random(99)) for _ in range(20)]
    results_b = [classify_severity(0.5, random.Random(99)) for _ in range(20)]
    assert results_a == results_b

def test_classify_severity_respects_serious_floor():
    """fatality risk that is too low should never be classed as serious"""
    import random
    from sim_core.severity import classify_severity, SERIOUS_FLOOR
    assert classify_severity(SERIOUS_FLOOR / 2, random.Random(1)) == "slight"

def test_collision_severity_is_consistent_across_ticks():
    """bug test - vehicle's severity would be resampled every tick leading to update
    due to vehicle's new speed (stationary) being used as the new collision speed (so severe drops to slight)"""
    for seed in range(10):
        config = load_scenario("scenarios/multi_sensor_spoof.yaml")
        config.random_seed = seed
        sim = Simulation(config)
        steps = int(config.duration / config.timestep)
        severities_seen = set()
        first_severity = None
        for _ in range(steps):
            r = sim.step(config.timestep)[0]
            if r.collision:
                severities_seen.add(r.severity)
                if first_severity is None:
                    first_severity = r.severity
        assert first_severity in ("slight", "serious", "fatal")
        assert severities_seen == {first_severity}, \
            f"severity changed across ticks, (seed={seed}): saw {severities_seen}"

def test_follower_avoids_lead_crash_under_every_policy():
    """following vehicle should successfully stop after lead car crashes"""
    for policy in ("camera_priority", "majority_vote", "confidence_weighted", "plausibility_filtered"):
        config = load_scenario("scenarios/follower_avoids_lead_crash.yaml")
        config.vehicles[1].fusion_policy = policy
        sim = Simulation(config)
        log = sim.run()
        lead_collided = any(e.collision for e in log if e.vehicle_id == "lead")
        follower_collided = any(e.collision for e in log if e.vehicle_id == "follower")
        assert lead_collided, "expected lead to be defeated by radar spoof"
        assert not follower_collided, f"expected follower to avoid crash under {policy}"

def test_follower_camera_jam_causes_crash_only_for_camera_priority():
    """fusion policy with no fallback should crash as it is now blind"""
    config = load_scenario("scenarios/follower_camera_jam_causes_crash.yaml")
    config.vehicles[1].fusion_policy = "camera_priority"
    sim = Simulation(config)
    log = sim.run()
    assert any(e.collision for e in log if e.vehicle_id == "follower")

    for policy in ("majority_vote", "confidence_weighted", "plausibility_filtered"):
        config = load_scenario("scenarios/follower_camera_jam_causes_crash.yaml")
        config.vehicles[1].fusion_policy = policy
        sim = Simulation(config)
        log = sim.run()
        follower_collided = any(e.collision for e in log if e.vehicle_id == "follower")
        assert not follower_collided, f"expected {policy} to fall back to radar/LiDAR and avoid crash"

def test_correctly_stopped_vehicle_does_not_register_as_vehicle_collision():
    """vehicle that is close to a second vehicle (that stopped successfully) should not be considered as collision"""
    import yaml
    scenario_dict = {
        "name": "correct stop is not a collision test", "duration": 20, "timestep": 0.1,
        "track": {"straight_length": 200, "radius": 60, "features": []},
        "vehicles": [
            {"vehicle_id": "lead", "start_distance": 50, "start_speed_mph": 0, "fusion_policy": "camera_priority"},
            {"vehicle_id": "follower", "start_distance": 0, "start_speed_mph": 30, "fusion_policy": "camera_priority"},
        ],
    }
    path = os.path.join(tempfile.gettempdir(), "_vehicle_no_false_collision_test.yaml")
    with open(path, "w") as f:
        yaml.dump(scenario_dict, f)
    config = load_scenario(path)
    sim = Simulation(config)
    log = sim.run()
    collided = any(e.collision for e in log if e.vehicle_id == "follower")
    assert not collided, "correctly-braked stop behind another vehicle should not register as collision"

def test_already_crashed_vehicle_visible_to_different_vehicle():
    """bug test - vehicle's that were crashed would be driven through"""
    import yaml
    scenario_dict = {
        "name": "crashed vehicle still visible test", "duration": 20, "timestep": 0.1,
        "track": {"straight_length": 200, "radius": 60, "features": []},
        "vehicles": [
            {"vehicle_id": "lead", "start_distance": 5, "start_speed_mph": 0, "fusion_policy": "camera_priority"},
            {"vehicle_id": "follower", "start_distance": 0, "start_speed_mph": 44.74, "fusion_policy": "camera_priority"},
            {"vehicle_id": "oncoming", "start_distance": 10, "start_speed_mph": 30, "lane": 1, "direction": -1, "fusion_policy": "camera_priority"},
        ],
    }
    path = os.path.join(tempfile.gettempdir(), "_crashed_visible_test.yaml")
    with open(path, "w") as f:
        yaml.dump(scenario_dict, f)
    config = load_scenario(path)
    sim = Simulation(config)
    log = sim.run()
    collided = any(e.collision for e in log if e.vehicle_id == "follower")
    assert collided, "expected follower to either stop or crash with lead"

def test_ground_truth_correctly_shows_vehicle_not_hazard_when_occluding():
    """a vehicle stopped near a hazard should report the closest hazard not what is behind it"""
    config = load_scenario("scenarios/follower_camera_jam_causes_crash.yaml")
    config.vehicles[1].fusion_policy = "majority_vote"
    sim = Simulation(config)
    log = sim.run()
    follower = [e for e in log if e.vehicle_id == "follower"]
    kinds_seen = {e.ground_truth_kind for e in follower if e.ground_truth_kind}
    assert kinds_seen == {"vehicle"}, f"expected only 'vehicle' to be reported, got {kinds_seen}"

def test_vehicle_successfully_braking_does_not_swerve():
    """a vehicle that can brake successfully should not swerve"""
    config = load_scenario("scenarios/radar_spoof_masking.yaml")
    config.vehicles[0].fusion_policy = "camera_priority"
    sim = Simulation(config)
    log = sim.run()
    lanes_used = {e.lane for e in log if e.vehicle_id == "solo"}
    assert lanes_used == {0}, f"expected solo to stay in lane 0 throughout, saw lanes {lanes_used}"

def test_camera_phantom_forces_braking_but_swerve_check_prevents_collision():
    """updated swerve logic - tests if it works as intended (vehicles were not aware to oncoming lane)"""
    config = load_scenario("scenarios/phantom_induced_swerve_collision.yaml")
    sim = Simulation(config)
    log = sim.run()
    solo = [e for e in log if e.vehicle_id == "solo"]
    swerved = any(e.lane == 1 for e in solo)
    collided = any(e.collision for e in solo)
    min_speed = min(e.speed for e in solo)
    assert not swerved, "expected oncoming-lane check to correctly decline unsafe swerve"
    assert not collided, "expected no collision, since swerve was correctly denied"
    from sim_core.units import ms_to_mph
    assert ms_to_mph(min_speed) < 35.0, "expected the phantom to produce braking"

def test_roundabout_dual_attack_produces_a_collision():
    """both cars should collide on a roundabout"""
    config = load_scenario("scenarios/roundabout_dual_attack.yaml")
    sim = Simulation(config)
    log = sim.run()
    collided = any(e.collision for e in log if e.vehicle_id == "second")
    assert collided

def test_roundabout_baseline_without_attacks_is_safe():
    """should not lead to a collision - no attacks present to cause one"""
    import yaml
    scenario_dict = {
        "name": "roundabout baseline control", "duration": 25, "timestep": 0.1,
        "vehicles": [
            {"vehicle_id": "first", "start_distance": 140, "start_speed_mph": 5, "fusion_policy": "camera_priority", "gps_policy": "naive"},
            {"vehicle_id": "second", "start_distance": 0, "start_speed_mph": 44.74, "fusion_policy": "camera_priority"},
        ],
    }
    path = os.path.join(tempfile.gettempdir(), "_roundabout_baseline_test.yaml")
    with open(path, "w") as f:
        yaml.dump(scenario_dict, f)
    config = load_scenario(path)
    sim = Simulation(config)
    log = sim.run()
    collided = any(e.collision for e in log)
    assert not collided

def test_oncoming_lane_sensing_is_scaled_by_visibility():
    """check if visibility also changes oncoming lane visibility"""
    import yaml
    def make_scenario(visibility):
        return {
            "name": "oncoming visibility test", "duration": 1, "timestep": 0.1, "visibility": visibility,
            "track": {"straight_length": 200, "radius": 60, "features": []},
            "vehicles": [
                {"vehicle_id": "solo", "start_distance": 0, "start_speed_mph": 30, "fusion_policy": "camera_priority"},
                {"vehicle_id": "onc", "start_distance": 100, "start_speed_mph": 30, "lane": 1, "direction": -1, "fusion_policy": "camera_priority"},
            ],
        }
    path = os.path.join(tempfile.gettempdir(), "_oncoming_visibility_test.yaml")

    with open(path, "w") as f:
        yaml.dump(make_scenario(1.0), f)
    config = load_scenario(path)
    sim = Simulation(config)
    sim.step(config.timestep)
    belief_clear = sim._last_oncoming_belief["solo"]
    assert belief_clear.obstacle_present, "expected oncoming vehicle to be sensed at full visibility"

    with open(path, "w") as f:
        yaml.dump(make_scenario(0.3), f)
    config = load_scenario(path)
    sim = Simulation(config)
    sim.step(config.timestep)
    belief_foggy = sim._last_oncoming_belief["solo"]
    assert not belief_foggy.obstacle_present, "expected oncoming vehicle to be beyond reduced effective range at low visibility"

def test_jam_attacks_affect_oncoming_lane_but_spoofs_do_not():
    """tests if jam attacks make the vehicle blind to the oncoming lane"""
    import yaml
    def make_scenario(attack):
        return {
            "name": "oncoming attack scope test", "duration": 1, "timestep": 0.1,
            "track": {"straight_length": 200, "radius": 60, "features": []},
            "vehicles": [
                {"vehicle_id": "solo", "start_distance": 0, "start_speed_mph": 30, "fusion_policy": "camera_priority"},
                {"vehicle_id": "onc", "start_distance": 50, "start_speed_mph": 30, "lane": 1, "direction": -1, "fusion_policy": "camera_priority"},
            ],
            "attacks": [attack],
        }
    path = os.path.join(tempfile.gettempdir(), "_oncoming_attack_scope_test.yaml")

    with open(path, "w") as f:
        yaml.dump(make_scenario({"type": "camera_jam", "target_vehicle": "solo", "start_time": 0.0, "duration": 5.0}), f)
    config = load_scenario(path)
    sim = Simulation(config)
    sim.step(config.timestep)
    belief_jammed = sim._last_oncoming_belief["solo"]
    assert not belief_jammed.obstacle_present, "expected a jammed camera to be blind to oncoming lane"

    with open(path, "w") as f:
        yaml.dump(make_scenario({"type": "camera_phantom", "target_vehicle": "solo", "start_time": 0.0, "duration": 5.0, "phantom_distance": 3.0}), f)
    config = load_scenario(path)
    sim = Simulation(config)
    sim.step(config.timestep)
    belief_phantom = sim._last_oncoming_belief["solo"]
    assert belief_phantom.obstacle_present, "expected real oncoming vehicle to be correctly sensed"
    assert abs(belief_phantom.distance_to_obstacle - 3.0) > 1.0, \
        "expected oncoming-lane distance to reflect real oncoming vehicle"

def test_phantom_brake_scenario_does_not_swerve():
    """tests updated swerve logic"""
    config = load_scenario("scenarios/phantom_brake.yaml")
    sim = Simulation(config)
    log = sim.run()
    follower_log = [e for e in log if e.vehicle_id == "follower"]
    lanes_used = {e.lane for e in follower_log}
    assert lanes_used == {0}, f"expected follower to stay in lane 0, saw lanes {lanes_used}"

def test_struck_pedestrian_stops_being_present_and_crossing():
    """a pedestrian should remain stationary after it has been hit"""
    import yaml
    scenario_dict = {
        "name": "struck pedestrian test", "duration": 15, "timestep": 0.1,
        "track": {"straight_length": 200, "radius": 60,
                  "features": [{"feature_id": "crossing_1", "feature_type": "pedestrian_crossing", "position": 150}]},
        "vehicles": [{"vehicle_id": "solo", "start_distance": 0, "start_speed_mph": 44.74, "fusion_policy": "camera_priority"}],
        "hazards": [{"type": "pedestrian_crossing", "feature_id": "crossing_1", "start_time": 6.0, "duration": 2.0}],
    }
    path = os.path.join(tempfile.gettempdir(), "_struck_pedestrian_test.yaml")
    with open(path, "w") as f:
        yaml.dump(scenario_dict, f)
    config = load_scenario(path)
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)

    struck_at = None
    for i in range(steps):
        t = round(sim.time, 2)
        results = sim.step(config.timestep)
        solo = next(r for r in results if r.vehicle_id == "solo")
        pedestrian = sim.hazards[0]
        if solo.collision and struck_at is None:
            struck_at = t
            assert pedestrian.struck, "expected hazard to be marked struck the same tick the collision registers"
        if struck_at is not None and t > struck_at:
            assert pedestrian.lane_at(sim.time) is None, \
                "expected a struck pedestrian to no longer be in any lane"
            assert not pedestrian.is_present(sim.time), \
                "expected a struck pedestrian to no longer be present"

    assert struck_at is not None, "expected this scenario to produce a collision"
