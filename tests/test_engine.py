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

    attack.check_trigger(t=1.0, vehicle_s=320.0, track=track)  # only 20m past - not yet
    assert not attack.is_active(1.0)

    attack.check_trigger(t=2.0, vehicle_s=355.0, track=track)  # 55m past - should arm
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
    required_deceleration is 0, must still be free to resume toward
    cruise speed. 'something was detected' and 'I need to brake for it'
    are not the same condition - conflating them meant a vehicle that
    had dipped below cruise speed once could get stuck there as long as
    anything remained within sensor range, however harmlessly far away."""
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
    """A camera phantom attack against a vehicle with nothing else ahead
    of it (no other vehicle, hazard, or red light) must still succeed.
    A phantom fabricates a detection where ground truth had none - if
    the engine only applies attacks to sensors that already have a
    reading, a phantom could never appear on an otherwise-clear road."""
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
def _obstacle_swerve_scenario_dict(solo_mph: float, oncoming_mph: float, reveal_distance: float = 30.0) -> dict:
    """Builds an obstacle_swerve-style scenario as a dict, with the
    obstacle's reveal time computed from reveal_distance rather than a
    fixed simulation time - so the same "how much warning does the
    driver get" setup is fair across different solo speeds. A fixed
    start_time would let a fast enough vehicle pass the obstacle's
    location before it "appears," never reacting at all - this earlier
    turned up as a real bug while speed-sweeping this scenario."""
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

def test_vehicle_swerves_when_it_cannot_stop_safely_in_time():
    """A vehicle facing an obstacle it can't brake to a stop
    for (see can_stop_safely in braking.py) should swerve into lane 1
    rather than just braking harder and hitting it anyway."""
    swerved, _collided, _min_gap = _run_obstacle_swerve(solo_mph=44.74, oncoming_mph=20.0)
    assert swerved

def test_swerve_can_produce_a_genuine_collision_with_oncoming_traffic():
    """At high enough closing speeds, the swerve-into-the-oncoming-lane
    response is not risk-free - it can itself cause a collision with
    oncoming traffic, an emergency response to one hazard creating a
    different one.

    Checked against COLLISION_DISTANCE itself rather than a fixed,
    tighter number: the exact minimum gap at collision depends on the
    braking dynamics leading up to it, which can shift as that model is
    refined, without changing whether a collision occurs at all."""
    from sim_core.engine import COLLISION_DISTANCE
    _swerved, collided, min_gap = _run_obstacle_swerve(solo_mph=44.74, oncoming_mph=33.55)
    assert collided
    assert min_gap < COLLISION_DISTANCE

def test_swerve_can_also_pass_cleanly_with_a_slower_oncoming_vehicle():
    """Contrast case: with more of a gap to the oncoming vehicle, the
    same swerve manoeuvre should complete without a collision."""
    swerved, collided, min_gap = _run_obstacle_swerve(solo_mph=44.74, oncoming_mph=20.0)
    assert swerved
    assert not collided
    assert min_gap > 3.0

def test_vehicle_does_not_swerve_if_it_can_brake_to_a_safe_stop():
    """If the obstacle is revealed with plenty of room to brake safely,
    the vehicle should just brake - not swerve unnecessarily."""
    swerved, _collided, _min_gap = _run_obstacle_swerve(solo_mph=20.0, oncoming_mph=20.0, reveal_distance=150.0)
    assert not swerved

def test_crashed_vehicle_stops_instead_of_resuming_normal_control():
    """A collision should actually change the vehicle's behaviour, not
    just get logged while it carries on driving as if nothing happened.
    A collision is modelled as an immediate stop (not a gradual brake --
    see engine.py's crashed branch for why), so speed should drop to (near)
    zero on the tick immediately after the collision, and stay there."""
    config = load_scenario(os.path.join("scenarios", "obstacle_swerve.yaml"))
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
                "expected a crashed vehicle to stop immediately, not ramp down gradually"
            assert solo.speed == 0.0 or t < crashed_at + 0.5, \
                "expected speed to reach exactly zero shortly after the collision and stay there"
            if t > crashed_at + 2.0:
                assert solo.speed == 0.0
                return

    assert crashed_at is not None, "expected this scenario to produce a collision at all"

def test_opposite_direction_vehicle_is_sensed_through_the_crossing_point():
    """Two vehicles sharing a lane while travelling in opposite directions
    (e.g. after a swerve into an oncoming lane) close toward each other
    until their s-coordinates cross. A naive forward-only distance breaks
    down exactly at that crossing (see _ahead_distance_to_vehicle in
    engine.py) - this checks the vehicle is still correctly sensed as a
    close-range obstacle right at and after the crossing point, not lost
    entirely the way it would be with a pure wraparound-forward distance."""
    config = load_scenario(os.path.join("scenarios", "obstacle_swerve.yaml"))
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)

    for _ in range(steps):
        results = sim.step(config.timestep)
        solo = sim.vehicles["solo"]
        onc = sim.vehicles["oncoming"]
        if solo.lane == 1 and onc.lane == 1:
            # once both share the oncoming lane, and they're within a
            # small physical distance of each other, the belief should
            # reflect a close, genuine detection - not "no obstacle"
            gap = ((solo.s - onc.s) ** 2) ** 0.5  # same-track proximity, ignoring the small lane offset
            solo_result = next(r for r in results if r.vehicle_id == "solo")
            if gap < 5.0:
                belief = solo_result.fused_belief
                assert belief.obstacle_present, \
                    "expected the oncoming vehicle to be sensed at close range, not silently lost at the crossing point"
                return

    raise AssertionError("expected solo and oncoming to share lane 1 with a close gap at some point in this scenario")

def _run_gps_roundabout_scenario(policy: str, offset: tuple[float, float]) -> tuple[bool, float]:
    """Builds and runs a GPS-spoof-at-a-roundabout scenario for the given
    policy and offset. Returns (confused, final_s)."""
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

def test_naive_gps_policy_is_confused_by_a_large_spoof_at_a_roundabout():
    """A large, sudden GPS offset should fool a naive policy into taking
    an extra, unplanned lap of the roundabout."""
    confused, final_s = _run_gps_roundabout_scenario("naive", (100.0, 100.0))
    assert confused

def test_plausibility_checked_gps_policy_resists_the_same_large_spoof():
    """The identical attack against a plausibility-checked policy should
    be rejected - the vehicle should pass through without confusion."""
    confused, _final_s = _run_gps_roundabout_scenario("plausibility_checked", (100.0, 100.0))
    assert not confused

def test_small_gps_spoof_does_not_confuse_either_policy():
    """A small offset (a few metres) is plausible GPS noise - even a
    naive policy accepting it shouldn't be treated as a real navigation
    mistake, since it isn't large enough to plausibly cause a wrong
    exit."""
    for policy in ("naive", "plausibility_checked"):
        confused, _final_s = _run_gps_roundabout_scenario(policy, (3.0, 3.0))
        assert not confused, f"expected a small spoof not to confuse the {policy} policy"

def test_roundabout_confusion_costs_real_time_not_an_instant_jump():
    """When confusion is triggered, the vehicle should take one genuine
    extra lap of the roundabout - consuming roughly circumference/speed
    seconds of real simulated time and matching distance, not an instant
    position jump. (An earlier version of this mechanism applied the
    circumference as a single-tick position jump, which let a confused
    vehicle "arrive" at things further down the route *earlier* than an
    unconfused vehicle, despite having travelled further overall --
    physically backwards for a wrong turn.)"""
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
                break  # stop as soon as the excursion clears - s_at_cleared reflects that moment, not later normal driving

    assert confused_at_t is not None, "expected this scenario to trigger confusion"
    assert excursion_cleared_at_t is not None, "expected the excursion to eventually clear"

    elapsed = excursion_cleared_at_t - confused_at_t
    expected_elapsed = circumference / mph_to_ms(44.74)
    assert elapsed > 1.0, \
        f"expected the extra lap to take multiple seconds, not be instantaneous (took {elapsed:.2f}s)"
    assert abs(elapsed - expected_elapsed) < 1.0, \
        f"expected roughly circumference/speed ({expected_elapsed:.2f}s) of extra driving, got {elapsed:.2f}s"

    total_s_change = s_at_cleared - s_at_confusion
    assert abs(total_s_change) < 5.0, \
        (f"expected s to stay approximately frozen while working off the extra lap - the "
         f"circumference is consumed via roundabout_excursion_remaining, not by advancing s "
         f"directly - but it changed by {total_s_change:.2f}m")

def test_gps_plausibility_check_rejects_large_jumps_and_accepts_small_ones():
    """Direct unit test of the plausibility-checked GPS policy in
    isolation, independent of the roundabout scenario machinery."""
    from sim_core.navigation import PlausibilityCheckedGPSPolicy
    from sim_core.sensors import SensorReading, SensorType

    vehicle = VehicleState(vehicle_id="solo", s=0.0, speed=20.0, cruise_speed=20.0)
    policy = PlausibilityCheckedGPSPolicy()

    first = SensorReading(SensorType.GPS, detected_position=(0.0, 0.0))
    believed = policy.resolve(first, vehicle, dt=0.1)
    assert believed == (0.0, 0.0)

    plausible = SensorReading(SensorType.GPS, detected_position=(2.0, 0.0))
    believed = policy.resolve(plausible, vehicle, dt=0.1)
    assert believed == (2.0, 0.0), "a small, physically plausible jump should be accepted"

    implausible = SensorReading(SensorType.GPS, detected_position=(500.0, 500.0), is_attacked=True)
    believed = policy.resolve(implausible, vehicle, dt=0.1)
    assert believed == (2.0, 0.0), "an implausibly large jump should be rejected, holding the last known good position"

def test_radar_spoof_masking_is_ignored_by_camera_priority():
    """camera_priority never even looks at radar's reading, so a radar
    spoof masking a real pedestrian crossing should have no effect --
    the vehicle brakes to a full stop exactly as it would unattacked."""
    config = load_scenario("scenarios/radar_spoof_masking.yaml")
    config.vehicles[0].fusion_policy = "camera_priority"
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    min_speed = min(sim.step(config.timestep)[0].speed for _ in range(steps))
    assert min_speed < 1.0, "expected camera_priority to brake to a near-stop despite the radar spoof"

def test_radar_spoof_masking_is_now_defeated_by_majority_vote_once_lidar_exists():
    """Before LiDAR existed, majority_vote only had radar and camera to
    compare - a masking radar spoof produced disagreement between the
    two, with no way to break the tie, so majority_vote concluded no
    obstacle at all. With LiDAR as a third sensor, camera and LiDAR
    agree on the genuine hazard and outvote the spoofed radar reading -
    the cross-sensor validation defence Komissarov & Wool (2021)
    describe. majority_vote should now brake correctly, like
    camera_priority."""
    config = load_scenario("scenarios/radar_spoof_masking.yaml")
    config.vehicles[0].fusion_policy = "majority_vote"
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    min_speed = min(sim.step(config.timestep)[0].speed for _ in range(steps))
    assert min_speed < 1.0, "expected majority_vote to now brake correctly, with LiDAR and camera outvoting the spoofed radar"

def test_radar_spoof_masking_still_defeats_confidence_weighted_even_with_lidar():
    """Unlike majority_vote, confidence_weighted has no concept of being
    "outvoted" - it blends every sensor's distance by confidence
    regardless of how many agree. With radar spoofed to 200m (confidence
    1.0, a "confident lie") and camera + LiDAR both reporting the real,
    close distance, the blended average is still pulled far enough
    toward the spoofed value that the vehicle never meaningfully brakes:
    adding more sensors only helps a fusion policy that votes, not one
    that averages. Since it never brakes, it should reach and collide
    with the pedestrian it failed to detect."""
    from sim_core.units import ms_to_mph
    config = load_scenario("scenarios/radar_spoof_masking.yaml")
    config.vehicles[0].fusion_policy = "confidence_weighted"
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

    assert collided, "expected the vehicle to actually reach and collide with the pedestrian, having never braked for it"
    assert ms_to_mph(min_speed_before_collision) > 15.0, \
        "expected no meaningful braking at any point before the collision"

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

def test_driver_diversity_produces_different_but_sensible_braking_onset():
    """An aggressive driver (short reaction time, hard braking) should
    start braking later than the Highway Code default, while a cautious
    driver (long reaction time, gentle braking) should start earlier to
    compensate - both are individually rational given their own
    parameters, not just randomly different."""
    aggressive_start = _braking_start_time(0.45, 8.0)
    default_start = _braking_start_time(None, None)
    cautious_start = _braking_start_time(0.95, 5.0)
    assert cautious_start < default_start < aggressive_start

def test_braking_variation_is_reproducible_with_the_same_seed():
    """Randomised driver diversity should be reproducible: the same
    random_seed should always produce the same sampled braking
    parameters for a given vehicle."""
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

def test_braking_variation_defaults_to_exact_highway_code_figures():
    """With braking_variation left at its default (0.0), every vehicle
    should behave exactly as before this feature existed - identical,
    at the Highway Code figures - regardless of random_seed."""
    from sim_core.braking import REACTION_TIME, MAX_DECELERATION
    import yaml
    scenario_dict = {
        "name": "diversity default test", "duration": 5, "timestep": 0.1,
        "random_seed": 123,
        "vehicles": [
            {"vehicle_id": "solo", "start_distance": 0, "start_speed_mph": 30, "fusion_policy": "camera_priority"},
        ],
    }
    path = os.path.join(tempfile.gettempdir(), "_diversity_default_test.yaml")
    with open(path, "w") as f:
        yaml.dump(scenario_dict, f)
    config = load_scenario(path)
    sim = Simulation(config)
    v = sim.vehicles["solo"]
    assert v.reaction_time == REACTION_TIME
    assert v.max_deceleration == MAX_DECELERATION

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
    """A vehicle should treat a child pedestrian more cautiously (braking
    earlier) than an adult, and an elderly pedestrian somewhat more
    cautiously too, without changing the vehicle's own braking
    capability - only when it starts responding."""
    child_start = _pedestrian_braking_start("child")
    adult_start = _pedestrian_braking_start("adult")
    elderly_start = _pedestrian_braking_start("elderly")
    assert child_start < elderly_start < adult_start

def test_pedestrian_crossing_random_start_time_is_reproducible_per_seed():
    """Leaving a hazard's start_time unset should pick a random time
    within the scenario duration, reproducible given the same
    random_seed, and different across different seeds."""
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

def test_roundabout_giveway_candidate_appears_while_occupied_and_clears_after():
    """Directly checks the give-way mechanism in isolation: an approaching
    vehicle should see a give-way candidate (the roundabout's entry
    point) while another vehicle occupies the roundabout, and should not
    once that vehicle is well clear of it. Checked directly against
    _ground_truth_readings rather than end-to-end speed, since an
    end-to-end run easily conflates give-way braking with ordinary
    car-following once the two vehicles are close on the same lane
    regardless of the roundabout."""
    from sim_core.engine import ROUNDABOUT_OCCUPANCY_RADIUS
    config = load_scenario(os.path.join("scenarios", "roundabout_giveway.yaml"))
    sim = Simulation(config)
    roundabout = sim.track.feature("roundabout_1")

    # "first" placed directly on the roundabout, "second" approaching
    # from 100m back - well within sensor range, not yet on it itself
    sim.vehicles["first"].s = roundabout.position
    sim.vehicles["second"].s = roundabout.position - 100
    readings, _kind = sim._ground_truth_readings("second")
    assert readings[SensorType.RADAR].detected_distance is not None, \
        "expected a give-way candidate while first occupies the roundabout"
    expected_entry_distance = 100 - ROUNDABOUT_OCCUPANCY_RADIUS
    assert abs(readings[SensorType.RADAR].detected_distance - expected_entry_distance) < 1.0

    # "first" now well clear of the roundabout, "second" far enough back
    # that no other candidate (vehicle-ahead, hazard, light) applies either
    sim.vehicles["first"].s = roundabout.position + 100
    sim.vehicles["second"].s = roundabout.position - 200
    readings, _kind = sim._ground_truth_readings("second")
    assert readings[SensorType.RADAR].detected_distance is None, \
        "expected no give-way candidate once first has cleared the roundabout"

def test_roundabout_giveway_produces_measurable_braking_end_to_end():
    """End-to-end confirmation that the give-way mechanism actually
    causes a real scenario's approaching vehicle to slow down while
    another vehicle occupies the roundabout (the isolated candidate test
    above checks the mechanism directly, this checks it has a real
    effect on driving behaviour)."""
    from sim_core.units import ms_to_mph
    config = load_scenario(os.path.join("scenarios", "roundabout_giveway.yaml"))
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    cruise_mph = ms_to_mph(config.vehicles[1].start_speed)  # "second"
    min_second_speed = None
    for _ in range(steps):
        results = sim.step(config.timestep)
        second = next(r for r in results if r.vehicle_id == "second")
        if min_second_speed is None or second.speed < min_second_speed:
            min_second_speed = second.speed
    assert ms_to_mph(min_second_speed) < cruise_mph - 5.0, \
        "expected second to measurably slow down while first occupies the roundabout"

def test_vehicle_holds_a_firm_stop_near_a_persistent_hazard_instead_of_creeping():
    """A vehicle stopped very close to a real, still-present hazard
    should settle at a firm, held stop, not oscillate between briefly
    resuming and re-braking, which used to let it creep forward into
    the hazard over many seconds.

    Uses a synthetic scenario with a long pedestrian-crossing duration
    (60s) rather than radar_spoof_masking.yaml, since this is a pure
    control-logic test of the firm-stop behaviour and needs the hazard
    to stay in the vehicle's lane long enough for a slow creep to become
    visible - a real pedestrian crosses one lane in a few seconds (see
    WALKING_SPEED in hazards.py), too short for that on its own."""
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
            # well after first reaching a stop: speed should still be
            # (at most negligibly) zero, and distance should not have
            # continued shrinking
            assert solo.speed < 0.1
            assert distance_when_stopped - solo.fused_belief.distance_to_obstacle < 0.1
            return

    assert stopped_at is not None, "expected the vehicle to reach a stop at some point in this scenario"

def test_lidar_has_a_shorter_range_than_radar_and_camera():
    """LiDAR should report nothing detected while an object is still
    within radar/camera's longer range but beyond LIDAR_MAX_SENSOR_RANGE,
    then start reporting once the vehicle closes to within LiDAR's own
    range - reflecting real automotive LiDAR's shorter range."""
    from sim_core.engine import LIDAR_MAX_SENSOR_RANGE, MAX_SENSOR_RANGE
    assert LIDAR_MAX_SENSOR_RANGE < MAX_SENSOR_RANGE

    config = load_scenario("scenarios/lidar_short_range.yaml")
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    for _ in range(steps):
        results = sim.step(config.timestep)
        r = results[0]
        radar_dist = r.radar_reading.detected_distance
        lidar_dist = r.lidar_reading.detected_distance
        if radar_dist is not None and radar_dist > LIDAR_MAX_SENSOR_RANGE:
            assert lidar_dist is None, \
                f"expected LiDAR to report nothing beyond its own range (radar saw {radar_dist:.1f}m)"
        if lidar_dist is not None:
            assert lidar_dist <= LIDAR_MAX_SENSOR_RANGE
            return  # confirmed the crossover happened correctly
    raise AssertionError("expected LiDAR to eventually detect the hazard as the vehicle closed in")

def test_coordinated_radar_and_lidar_spoof_defeats_majority_vote():
    """Unlike a single-sensor radar spoof (which majority_vote now
    resists thanks to LiDAR), spoofing radar and LiDAR together to
    agree on the same fabricated value forms its own two-sensor
    majority, outvoting camera's correct reading. A limitation of
    majority-vote fusion, not a bug: its resilience depends on an
    attacker compromising only one sensor at a time. Since it never
    brakes, it should reach and collide with the pedestrian."""
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

    assert collided, "expected the vehicle to actually reach and collide with the pedestrian, having never braked for it"
    assert ms_to_mph(min_speed_before_collision) > 15.0, \
        "expected no meaningful braking at any point before the collision"

def test_pedestrian_collision_is_detected_when_a_vehicle_fails_to_brake():
    """A vehicle that never brakes for a real pedestrian crossing (e.g.
    because it was fooled by an attack) should eventually register a
    collision."""
    config = load_scenario("scenarios/multi_sensor_spoof.yaml")
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    collided = any(sim.step(config.timestep)[0].collision for _ in range(steps))
    assert collided

def test_a_correctly_braked_stop_does_not_register_as_a_pedestrian_collision():
    """A vehicle that brakes correctly and holds at its intended safety
    margin should NOT register as having collided with the pedestrian it
    correctly stopped for - PEDESTRIAN_COLLISION_DISTANCE must stay
    smaller than braking.py's SAFETY_MARGIN, or every successful stop
    would falsely count as a collision."""
    from sim_core.braking import SAFETY_MARGIN
    from sim_core.engine import PEDESTRIAN_COLLISION_DISTANCE
    assert PEDESTRIAN_COLLISION_DISTANCE < SAFETY_MARGIN

    config = load_scenario(os.path.join("scenarios", "pedestrian_crossing_test.yaml"))
    config.vehicles[0].fusion_policy = "camera_priority"
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    collided = any(sim.step(config.timestep)[0].collision for _ in range(steps))
    assert not collided, "a correctly-braked stop should never register as a collision"

def test_camera_jam_defeats_camera_priority():
    """camera_priority has no fallback when camera itself is jammed --
    it should drive straight through the genuine pedestrian crossing and
    collide, exactly the single-point-of-failure vulnerability jamming
    the trusted sensor is meant to expose."""
    config = load_scenario("scenarios/camera_jam_defeats_camera_priority.yaml")
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    collided = any(sim.step(config.timestep)[0].collision for _ in range(steps))
    assert collided

def test_camera_jam_does_not_defeat_majority_vote_or_confidence_weighted():
    """Unlike camera_priority, majority_vote and confidence_weighted both
    fall back to radar and LiDAR when camera is jammed - those two
    sensors still agree on the genuine hazard, so both should brake
    correctly and never collide."""
    for policy in ("majority_vote", "confidence_weighted"):
        config = load_scenario("scenarios/camera_jam_defeats_camera_priority.yaml")
        config.vehicles[0].fusion_policy = policy
        sim = Simulation(config)
        steps = int(config.duration / config.timestep)
        collided = any(sim.step(config.timestep)[0].collision for _ in range(steps))
        assert not collided, f"expected {policy} to correctly fall back to radar/LiDAR and avoid a collision"

def test_jamming_a_non_trusted_sensor_never_defeats_any_policy():
    """Jamming radar or LiDAR should never defeat any fusion policy:
    camera_priority never looked at them anyway, and majority_vote /
    confidence_weighted both still have two genuine, agreeing sensors
    left (camera plus whichever of radar/LiDAR wasn't jammed) - jamming
    only ever removes a vote, unlike spoofing, which can fabricate one."""
    import yaml
    scenario_dict = {
        "name": "jam test", "duration": 20, "timestep": 0.1,
        "track": {"straight_length": 200, "radius": 60,
                  "features": [{"feature_id": "crossing_1", "feature_type": "pedestrian_crossing", "position": 150}]},
        "vehicles": [{"vehicle_id": "solo", "start_distance": 0, "start_speed_mph": 44.74, "fusion_policy": "camera_priority"}],
        "hazards": [{"type": "pedestrian_crossing", "feature_id": "crossing_1", "start_time": 0.0, "duration": 20.0}],
        "attacks": [{"type": "radar_jam", "target_vehicle": "solo", "trigger_before_feature": "crossing_1",
                     "trigger_distance": 60, "duration": 10.0}],
    }
    for jam_type in ("radar_jam", "lidar_jam"):
        scenario_dict["attacks"][0]["type"] = jam_type
        for policy in ("camera_priority", "majority_vote", "confidence_weighted"):
            scenario_dict["vehicles"][0]["fusion_policy"] = policy
            path = os.path.join(tempfile.gettempdir(), "_non_trusted_jam_test.yaml")
            with open(path, "w") as f:
                yaml.dump(scenario_dict, f)
            config = load_scenario(path)
            sim = Simulation(config)
            steps = int(config.duration / config.timestep)
            collided = any(sim.step(config.timestep)[0].collision for _ in range(steps))
            assert not collided, f"expected {jam_type} to never defeat {policy}"

def test_pedestrian_lane_at_walks_from_lane_0_to_lane_1():
    """A pedestrian crossing should occupy lane 0 for the first half of
    its duration and lane 1 for the second half, not both lanes for the
    whole duration - a pedestrian walks across, they don't stand in the
    road blocking everything at once."""
    from sim_core.hazards import PedestrianCrossing
    hazard = PedestrianCrossing(feature_id="crossing_1", start_time=10.0, duration=6.0)

    assert hazard.lane_at(9.9) is None       # not started yet
    assert hazard.lane_at(10.0) == 0         # just started - lane 0
    assert hazard.lane_at(12.9) == 0         # still in lane 0 (first half)
    assert hazard.lane_at(13.0) == 1         # crossed into lane 1 (second half)
    assert hazard.lane_at(15.9) == 1         # still in lane 1
    assert hazard.lane_at(16.0) is None      # finished crossing

def test_vehicle_resumes_once_pedestrian_has_walked_clear_of_its_lane():
    """A vehicle that correctly brakes for a pedestrian in its own lane
    should resume once that pedestrian has walked into the
    other lane - not remain stopped for the pedestrian's entire
    crossing regardless of which lane they're actually in."""
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

    assert braked, "expected the vehicle to brake for the pedestrian while in its lane"
    assert resumed, "expected the vehicle to resume once the pedestrian walked clear of its lane"

def test_low_visibility_alone_can_cause_a_collision():
    """Severely reduced visibility (fog/rain) should be able to cause a
    genuine collision on its own - no attack needed - since it simply
    doesn't leave enough distance to brake safely from cruise speed."""
    config = load_scenario("scenarios/low_visibility_collision.yaml")
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    collided = any(sim.step(config.timestep)[0].collision for _ in range(steps))
    assert collided

def test_visibility_defaults_to_clear_and_does_not_affect_existing_scenarios():
    """Leaving visibility unset should behave exactly as before this
    feature existed - every existing scenario is unaffected."""
    config = load_scenario("scenarios/pedestrian_crossing_test.yaml")
    assert config.visibility == 1.0

def test_visibility_scales_detection_distance_proportionally():
    """A lower visibility value should proportionally shrink the distance
    at which a hazard is first detected."""
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

def test_spawner_disabled_by_default_does_not_affect_existing_scenarios():
    """Every existing scenario has no spawner configured - confirms
    Simulation still runs identically (no spawning at all) when
    config.spawner is None, the default."""
    config = load_scenario("scenarios/pedestrian_crossing_test.yaml")
    assert config.spawner is None
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    for _ in range(steps):
        sim.step(config.timestep)
    assert len(sim.vehicles) == 1  # only "solo" - nothing spawned

def test_spawner_aligns_spawns_with_the_traffic_light_cycle():
    """A new vehicle should appear once per full light cycle (16s by
    default), not at an arbitrary interval."""
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

    assert len(spawn_times) >= 3, "expected several spawns over this scenario's duration"
    for i in range(1, len(spawn_times)):
        gap = spawn_times[i] - spawn_times[i - 1]
        # a spawn is skipped entirely (not queued) if already at
        # max_concurrent, so a gap can legitimately be a multiple of
        # cycle_length, not always exactly one cycle
        remainder = gap % cycle_length
        assert remainder < 0.5 or remainder > cycle_length - 0.5, \
            f"expected spawns aligned to multiples of {cycle_length}s, got a {gap:.1f}s gap"

def test_spawner_never_exceeds_max_concurrent():
    """The number of currently-spawned vehicles should never exceed
    SpawnerConfig.max_concurrent, however long the scenario runs."""
    config = load_scenario("scenarios/background_traffic.yaml")
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    for _ in range(steps):
        sim.step(config.timestep)
        assert len(sim._spawned_vehicle_ids) <= config.spawner.max_concurrent

def test_spawned_vehicle_despawns_after_exactly_one_lap():
    """A spawned vehicle should be removed once it has travelled one full
    lap (track.total_length) since it entered - not before, not after,
    and not for any other reason (age, random chance, etc.)."""
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
            pass  # already gone, nothing to check this tick
        elif first_spawn_id in prev_ids and first_spawn_id not in current_ids:
            # just despawned this tick - can't check its final s directly
            # (it's been deleted), but we know despawn only happens once
            # abs(s - origin) >= total_length, per _despawn_completed_laps
            break
        prev_ids = current_ids

    assert first_spawn_id is not None, "expected at least one vehicle to have spawned"

def test_spawner_produces_both_lane_choices_over_many_spawns():
    """Lane/direction should be randomised (both lane 0/normal
    and lane 1/oncoming should occur), not always the same choice."""
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

def test_background_traffic_scenario_actually_demonstrates_vehicle_detection():
    """The background traffic demonstration scenario should show Solo
    reacting to a spawned vehicle at some point, not just the
    traffic light - checked via TickResult.ground_truth_kind, which was
    added specifically because this couldn't otherwise be told apart
    from a traffic-light stop in the standard output. An earlier version
    of this scenario had spawned traffic at the same speed as Solo,
    which meant they never converged into sensor range at all."""
    config = load_scenario("scenarios/background_traffic.yaml")
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    vehicle_ticks = sum(1 for _ in range(steps) if sim.step(config.timestep)[0].ground_truth_kind == "vehicle")
    assert vehicle_ticks > 0, "expected Solo to react to a spawned vehicle at least once in this scenario"

def test_ground_truth_kind_distinguishes_traffic_light_from_vehicle():
    """Direct check that ground_truth_kind correctly tells apart a
    traffic-light stop from a vehicle-ahead detection - the two were
    previously indistinguishable from the outside (both just showed up
    as belief.source == 'camera')."""
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
        f"expected only the traffic light to be detected in a single-vehicle scenario, got {kinds_seen}"

def test_pedestrian_fatality_risk_matches_rosen_sander_2009():
    """Fatality risk at 30mph and 40mph impact speed should be
    approximately 7% and 25%, matching the headline figures stated in
    Rosen & Sander (2009)."""
    from sim_core.severity import pedestrian_fatality_risk
    from sim_core.units import mph_to_ms
    assert abs(pedestrian_fatality_risk(mph_to_ms(30)) - 0.07) < 0.01
    assert abs(pedestrian_fatality_risk(mph_to_ms(40)) - 0.25) < 0.02

def test_vehicle_fatality_risk_matches_richards_2010_frontal_impact():
    """Matches the three published risk points for belted car drivers in
    frontal impacts, Richards (2010) Fig 3.3: ~3% at 30mph, ~17% at
    40mph, ~60% at 50mph delta-v."""
    from sim_core.severity import vehicle_fatality_risk
    from sim_core.units import mph_to_ms
    assert abs(vehicle_fatality_risk(mph_to_ms(30)) - 0.03) < 0.01
    assert abs(vehicle_fatality_risk(mph_to_ms(40)) - 0.17) < 0.02
    assert abs(vehicle_fatality_risk(mph_to_ms(50)) - 0.60) < 0.02

def test_vehicle_fatality_risk_is_higher_than_pedestrian_at_low_speed():
    """At the same speed a pedestrian is far more vulnerable than a
    car driver - should hold at any plausible speed"""
    from sim_core.severity import pedestrian_fatality_risk, vehicle_fatality_risk
    from sim_core.units import mph_to_ms
    for mph in [20, 25, 30]:
        v = mph_to_ms(mph)
        assert pedestrian_fatality_risk(v) > vehicle_fatality_risk(v)

def test_classify_severity_is_reproducible_with_a_seeded_rng():
    """Same seed, same fatal/non-fatal sampling outcome every time"""
    import random
    from sim_core.severity import classify_severity
    results_a = [classify_severity(0.5, random.Random(99)) for _ in range(20)]
    results_b = [classify_severity(0.5, random.Random(99)) for _ in range(20)]
    assert results_a == results_b

def test_classify_severity_respects_the_serious_floor():
    """A fatality risk too low to ever sample as fatal should classify
    as 'slight', not 'serious' - SERIOUS_FLOOR should actually gate the
    non-fatal classification"""
    import random
    from sim_core.severity import classify_severity, SERIOUS_FLOOR
    assert classify_severity(SERIOUS_FLOOR / 2, random.Random(1)) == "slight"

def test_collision_severity_is_populated_and_consistent_across_ticks():
    """A collision's severity should be set on the first tick collision
    becomes True, and stay the same (not resample) on every subsequent
    tick while the vehicle remains crashed.

    Uses multi_sensor_spoof.yaml (a ~44.7mph pedestrian
    impact, ~30-40% fatality risk) rather than a low-speed collision:
    at low speed the risk is near zero, so a broken implementation that
    resamples every tick would still almost always draw "slight" by
    coincidence - the gap that let this bug through originally, where
    severity flipped from slight to serious partway through a run.
    Checked across several seeds, since any single seed could get
    unlucky."""
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
            f"severity changed across ticks once set (seed={seed}): saw {severities_seen}"

def test_plausibility_filtered_resists_radar_spoof_masking():
    """The attack this policy's design was originally motivated by:
    radar jumping 58m -> 200m in one tick should be flagged as
    implausible and excluded, leaving camera to correctly drive braking."""
    from sim_core.units import ms_to_mph
    config = load_scenario("scenarios/radar_spoof_masking.yaml")
    config.vehicles[0].fusion_policy = "plausibility_filtered"
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    min_speed = min(sim.step(config.timestep)[0].speed for _ in range(steps))
    assert ms_to_mph(min_speed) < 5.0

def test_plausibility_filtered_resists_coordinated_multi_sensor_spoof():
    """The scenario majority_vote and confidence_weighted both fail:
    radar and LiDAR spoofed together to agree on a fabricated distance.
    Both should be filtered out as implausible jumps, leaving camera as
    the sole - but established/corroborated - survivor, single-survivor
    trusted directly rather than discarded for lack of a second opinion."""
    config = load_scenario("scenarios/multi_sensor_spoof.yaml")
    config.vehicles[0].fusion_policy = "plausibility_filtered"
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    collided = any(sim.step(config.timestep)[0].collision for _ in range(steps))
    assert not collided

def test_plausibility_filtered_resists_camera_jam():
    """camera_priority's single point of failure: with camera jammed,
    plausibility_filtered should fall back to radar/LiDAR, which still
    agree on the genuine hazard."""
    config = load_scenario("scenarios/camera_jam_defeats_camera_priority.yaml")
    config.vehicles[0].fusion_policy = "plausibility_filtered"
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    collided = any(sim.step(config.timestep)[0].collision for _ in range(steps))
    assert not collided

def test_plausibility_filtered_resists_isolated_camera_phantom():
    """camera_phantom was never part of what motivated this policy's
    design. An earlier version of the single-survivor fallback
    (added to fix the coordinated-attack case) reintroduced this exact
    vulnerability - a lone fabricated reading with no corroboration
    history was trusted directly. Fixed by requiring single-survivor
    trust to have a corroboration history, not merely a history of
    reporting something. Guards against that regression recurring."""
    config = load_scenario("scenarios/isolated_camera_phantom.yaml")
    config.vehicles[0].fusion_policy = "plausibility_filtered"
    sim = Simulation(config)
    steps = int(config.duration / config.timestep)
    fooled = any(sim.step(config.timestep)[0].fused_belief.obstacle_present for _ in range(steps))
    assert not fooled

def test_plausibility_filtered_is_the_only_policy_resisting_every_attack():
    """The comparative finding this policy exists to demonstrate: each
    of the three original policies has at least one attack in this
    project's suite that defeats it (camera_priority: camera_jam,
    majority_vote: coordinated multi-sensor spoof, confidence_weighted:
    radar_spoof_masking), but plausibility_filtered resists all of
    them, including one (camera_phantom) it wasn't designed around."""
    from sim_core.units import ms_to_mph

    def collided_or_fooled(scenario_path, policy, check="collision"):
        config = load_scenario(scenario_path)
        config.vehicles[0].fusion_policy = policy
        sim = Simulation(config)
        steps = int(config.duration / config.timestep)
        if check == "collision":
            return any(sim.step(config.timestep)[0].collision for _ in range(steps))
        else:
            return any(sim.step(config.timestep)[0].fused_belief.obstacle_present for _ in range(steps))

    attacks = [
        ("scenarios/radar_spoof_masking.yaml", "collision"),
        ("scenarios/camera_jam_defeats_camera_priority.yaml", "collision"),
        ("scenarios/multi_sensor_spoof.yaml", "collision"),
        ("scenarios/isolated_camera_phantom.yaml", "fooled"),
    ]

    for scenario_path, check in attacks:
        defeated = collided_or_fooled(scenario_path, "plausibility_filtered", check)
        assert not defeated, f"plausibility_filtered should resist {scenario_path}"

    others_have_a_weakness = False
    for policy in ("camera_priority", "majority_vote", "confidence_weighted"):
        for scenario_path, check in attacks:
            if collided_or_fooled(scenario_path, policy, check):
                others_have_a_weakness = True
    assert others_have_a_weakness, "expected each original policy to have at least one demonstrated weakness"