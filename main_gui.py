"""Real-time playback GUI for a scenario

Usage: python main_gui.py <scenario.yaml>

This is a rendering layer on top of the existing simulation engine
- it calls Simulation.step() in a loop exactly like main_headless.py
does, and draws from the TickResult objects (and the Simulation object
itself, for track/hazard/traffic-light state not carried in
TickResult) that already exist. No simulation logic lives here

Controls:
  SPACE       pause / resume
  UP / DOWN   double / halve playback speed
  R           restart the scenario from the beginning
  ESC         quit
"""
import math
import os
import random
import sys

import pygame

from sim_core.engine import Simulation
from sim_core.scenario import load_scenario
from sim_core.track import LANE_WIDTH
from sim_core.units import ms_to_mph

WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 620
TRACK_MARGIN = 70
SIDEBAR_WIDTH = 300
FPS = 60
VEHICLE_RADIUS = 7
ROAD_HALF_WIDTH = LANE_WIDTH + 2.0   # covers both lanes plus a wider verge either side of the true centreline

COLOR_BG = (58, 102, 56)
COLOR_SIDEBAR_BG = (18, 19, 24)
COLOR_TRACK = (58, 60, 68)
COLOR_LANE_DIVIDER = (225, 225, 225)
COLOR_TEXT = (225, 227, 232)
COLOR_TEXT_DIM = (140, 143, 152)
COLOR_VEHICLE = (90, 190, 255)
COLOR_VEHICLE_ATTACKED = (255, 165, 40)
COLOR_VEHICLE_CRASHED = (235, 60, 60)
COLOR_HAZARD_PEDESTRIAN = (255, 210, 60)
COLOR_HAZARD_OBSTACLE = (200, 120, 60)
COLOR_ROUNDABOUT_ISLAND = (72, 120, 68)
COLOR_ROUNDABOUT_KERB = (225, 225, 225)
COLOR_LIGHT_RED = (220, 60, 60)
COLOR_LIGHT_AMBER = (230, 170, 40)
COLOR_LIGHT_GREEN = (70, 200, 100)
COLOR_BUILDING = (26, 26, 30)
COLOR_BUILDING_ROOF_EDGE = (55, 55, 62)
COLOR_FEATURE_PREVIEW = (255, 235, 80)


class Camera:
    """Maps world (x, y) metres to screen pixels, fitted to the track's
    own bounding box - see Track.position_at in track.py for why the
    box is [-radius, straight_length+radius] x [0, 2*radius]. World Y
    increases "up" (matching the track's own math), screen Y increases
    down, so the vertical axis is flipped here.

    When the track's own aspect ratio doesn't match the available
    drawing area (usually true - the track is wide and short), the
    scale is constrained by whichever dimension runs out first, leaving
    slack in the other. That slack is split evenly as extra margin on
    both sides, so the track is centred in the available space rather
    than anchored to one corner with all the dead space left on one
    side"""

    def __init__(self, track, width, height, margin):
        self.min_x = -track.radius
        self.max_x = track.straight_length + track.radius
        self.min_y = 0.0
        self.max_y = 2 * track.radius
        world_w = self.max_x - self.min_x
        world_h = self.max_y - self.min_y
        avail_w = width - 2 * margin
        avail_h = height - 2 * margin
        self.scale = min(avail_w / world_w, avail_h / world_h)
        used_w = world_w * self.scale
        used_h = world_h * self.scale
        self.margin_x = margin + (avail_w - used_w) / 2
        self.margin_y = margin + (avail_h - used_h) / 2
        self.height = height

    def to_screen(self, x, y):
        sx = self.margin_x + (x - self.min_x) * self.scale
        sy = self.height - self.margin_y - (y - self.min_y) * self.scale
        return int(sx), int(sy)


def offset_track_points(track, offset, step=1.0):
    """Points along the track, offset perpendicular to the true
    centreline by a fixed distance - generalises Track.lane_position_at
    so the same maths can produce the road's outer edge, inner edge, or
    the true centreline (offset=0). Matches lane_position_at's own sign convention
    positive offset is the lane-0 side"""
    points = []
    s = 0.0
    while s < track.total_length:
        x, y, heading = track.position_at(s)
        ox = x + offset * math.sin(heading)
        oy = y - offset * math.cos(heading)
        points.append((ox, oy))
        s += step
    x, y, heading = track.position_at(0.0)
    points.append((x + offset * math.sin(heading), y - offset * math.cos(heading)))
    return points


def build_static_geometry(track):
    """Returns a dict of screen-space-ready world point lists"""
    return {
        "outer_edge": offset_track_points(track, ROAD_HALF_WIDTH),
        "inner_edge": offset_track_points(track, -ROAD_HALF_WIDTH),
        "centreline": offset_track_points(track, 0.0, step=1.5),
    }


def draw_track(screen, camera, geometry):
    outer = [camera.to_screen(x, y) for x, y in geometry["outer_edge"]]
    inner = [camera.to_screen(x, y) for x, y in geometry["inner_edge"]]
    pygame.draw.polygon(screen, COLOR_TRACK, outer)
    pygame.draw.polygon(screen, COLOR_BG, inner)

    centre = [camera.to_screen(x, y) for x, y in geometry["centreline"]]
    for i in range(0, len(centre) - 1, 5):
        if i + 2 < len(centre):
            pygame.draw.line(screen, COLOR_LANE_DIVIDER, centre[i], centre[i + 2], 2)


def _nearest_feature_gap(track, s):
    return min(
        min(track.distance_ahead(s, f.position), track.distance_ahead(f.position, s))
        for f in track.features
    ) if track.features else float("inf")


MAX_BUILDING_HALF_DIAGONAL = ((34.0 ** 2 + 26.0 ** 2) ** 0.5) / 2   # ~21.4m - half the worst-case diagonal across the building size range in generate_buildings, used as a fixed safety margin for both along-road spacing and road-perpendicular setback. An axis-aligned rectangle's true extent along an arbitrary direction can be as large as its diagonal, not just its width or height - the road's local heading varies continuously around the curves, so a size-dependent estimate like max(w, h) genuinely under-measured on angled sections, which is what caused real overlaps rather than just an overly cautious margin.


def generate_buildings(track, seed=1234, feature_clearance=50.0, gap_from_road=(4.0, 10.0), gap_between=(20.0, 45.0)):
    """Generates a fixed, reproducible set of building footprints along
    both sides of the road, following its own curve (including around
    the semicircular ends) for WHERE each building sits, while every
    building stays axis-aligned to the world - not rotated to match the
    road's local heading (see draw_buildings). Each building is a
    single plain rectangle with randomised width and height"""
    rng = random.Random(seed)
    buildings = []

    for side in (1, -1):
        s = MAX_BUILDING_HALF_DIAGONAL
        while s < track.total_length - MAX_BUILDING_HALF_DIAGONAL:
            if _nearest_feature_gap(track, s) < feature_clearance:
                s += 5.0
                continue

            x, y, heading = track.position_at(s)
            w = rng.uniform(14.0, 34.0)
            h = rng.uniform(11.0, 26.0)
            setback = ROAD_HALF_WIDTH + MAX_BUILDING_HALF_DIAGONAL + rng.uniform(*gap_from_road)
            setback = min(setback, track.radius * 0.45)   # on a tight custom track, an inward setback this large could otherwise push a building far enough toward the loop's interior to collide with one on the opposite side of the curve
            bx = x + side * setback * math.sin(heading)
            by = y - side * setback * math.cos(heading)
            buildings.append((bx, by, heading, [(0.0, 0.0, w, h)]))

            s += 2 * MAX_BUILDING_HALF_DIAGONAL + rng.uniform(*gap_between)

    return buildings


def draw_buildings(screen, camera, buildings):
    for bx, by, _heading, rects in buildings:
        for ox, oy, w, h in rects:
            wx0, wy0 = bx + ox - w / 2, by + oy - h / 2
            wx1, wy1 = bx + ox + w / 2, by + oy + h / 2
            corners = [(wx0, wy0), (wx1, wy0), (wx1, wy1), (wx0, wy1)]
            screen_corners = [camera.to_screen(wx, wy) for wx, wy in corners]
            pygame.draw.polygon(screen, COLOR_BUILDING, screen_corners)
            pygame.draw.polygon(screen, COLOR_BUILDING_ROOF_EDGE, screen_corners, 1)


def draw_features(screen, camera, sim, font):
    for feature in sim.track.features:
        x, y, heading = sim.track.position_at(feature.position)
        sx, sy = camera.to_screen(x, y)

        if feature.feature_type == "roundabout":
            island_x, island_y, _h = sim.track.position_at(feature.position)
            isx, isy = camera.to_screen(island_x, island_y)
            outer_r = max(int(feature.radius * camera.scale), 10)
            island_r = max(int(outer_r * 0.55), 5)

            pygame.draw.circle(screen, COLOR_TRACK, (isx, isy), outer_r)
            dash_r = (island_r + outer_r) // 2
            for angle_deg in range(0, 360, 12):
                a1 = math.radians(angle_deg)
                a2 = math.radians(angle_deg + 6)
                p1 = (isx + dash_r * math.cos(a1), isy + dash_r * math.sin(a1))
                p2 = (isx + dash_r * math.cos(a2), isy + dash_r * math.sin(a2))
                pygame.draw.line(screen, COLOR_ROUNDABOUT_KERB, p1, p2, 2)
            pygame.draw.circle(screen, COLOR_ROUNDABOUT_ISLAND, (isx, isy), island_r)
            pygame.draw.circle(screen, COLOR_ROUNDABOUT_KERB, (isx, isy), island_r, 2)

        elif feature.feature_type == "junction":

            edge_x = x + ROAD_HALF_WIDTH * math.sin(heading)
            edge_y = y - ROAD_HALF_WIDTH * math.cos(heading)
            stub_length = 50
            far_x = edge_x + stub_length * math.sin(heading)
            far_y = edge_y - stub_length * math.cos(heading)
            along_dx, along_dy = math.cos(heading), math.sin(heading)
            stub_half_width = 9
            corners_world = [
                (edge_x - along_dx * stub_half_width, edge_y - along_dy * stub_half_width),
                (edge_x + along_dx * stub_half_width, edge_y + along_dy * stub_half_width),
                (far_x + along_dx * stub_half_width, far_y + along_dy * stub_half_width),
                (far_x - along_dx * stub_half_width, far_y - along_dy * stub_half_width),
            ]
            pygame.draw.polygon(screen, COLOR_TRACK, [camera.to_screen(cx, cy) for cx, cy in corners_world])
            dash_steps = 6
            for i in range(dash_steps):
                if i % 2 == 0:
                    t1 = i / dash_steps
                    t2 = (i + 0.6) / dash_steps
                    d1 = (edge_x + (far_x - edge_x) * t1, edge_y + (far_y - edge_y) * t1)
                    d2 = (edge_x + (far_x - edge_x) * t2, edge_y + (far_y - edge_y) * t2)
                    pygame.draw.line(screen, COLOR_LANE_DIVIDER, camera.to_screen(*d1), camera.to_screen(*d2), 2)

            light = sim.traffic_lights.get(feature.feature_id)
            if light is not None:
                state = light.state_at(sim.time)
                colour = {"red": COLOR_LIGHT_RED, "amber": COLOR_LIGHT_AMBER, "green": COLOR_LIGHT_GREEN}[state]
                lx, ly = camera.to_screen(edge_x, edge_y)
                pygame.draw.rect(screen, colour, (lx - 5, ly - 5, 10, 10))
                pygame.draw.rect(screen, COLOR_TEXT_DIM, (lx - 5, ly - 5, 10, 10), 1)

    for hazard in sim.hazards:
        kind = type(hazard).__name__
        if kind == "PedestrianCrossing":
            lane = hazard.lane_at(sim.time)
            if lane is None:
                continue
            feature = sim.track.feature(hazard.feature_id)
            x, y, _h = sim.track.lane_position_at(feature.position, lane)
            sx, sy = camera.to_screen(x, y)
            pygame.draw.circle(screen, COLOR_HAZARD_PEDESTRIAN, (sx, sy), 6)
        elif kind == "ObstacleInRoad":
            if not hazard.is_present(sim.time):
                continue
            feature = sim.track.feature(hazard.feature_id)
            x, y, _h = sim.track.lane_position_at(feature.position, hazard.lane)
            sx, sy = camera.to_screen(x, y)
            pygame.draw.rect(screen, COLOR_HAZARD_OBSTACLE, (sx - 6, sy - 6, 12, 12))


ROUNDABOUT_OCCUPANCY_RADIUS = 15.0   # metres either side of a roundabout's marked position that counts as "on" it - must match engine.py's own constant of the same name, duplicated here rather than imported since it's a private module-level constant, not part of the engine's public interface


def roundabout_swept_position(sim, vehicle_id, real_x, real_y):
    vehicle = sim.vehicles.get(vehicle_id)
    if vehicle is None or vehicle.crashed:
        return real_x, real_y

    if vehicle.roundabout_excursion_remaining > 0:
        for feature in sim.track.features:
            if feature.feature_type != "roundabout":
                continue
            gap = sim.track.signed_gap(vehicle.s, feature.position)
            if abs(gap) > ROUNDABOUT_OCCUPANCY_RADIUS:
                continue
            cx, cy, _h = sim.track.position_at(feature.position)
            _, _, road_heading = sim.track.position_at(feature.position)
            entry_x, entry_y, _h = sim.track.lane_position_at(feature.position - ROUNDABOUT_OCCUPANCY_RADIUS, vehicle.lane)
            exit_x, exit_y, _h = sim.track.lane_position_at(feature.position + ROUNDABOUT_OCCUPANCY_RADIUS, vehicle.lane)
            entry_angle = math.atan2(entry_y - cy, entry_x - cx)
            exit_angle = math.atan2(exit_y - cy, exit_x - cx)

            side = 1 if vehicle.lane == 0 else -1
            preferred_angle = math.atan2(-side * math.cos(road_heading), side * math.sin(road_heading))

            def _angular_distance(a, b):
                d = (a - b) % (2 * math.pi)
                return min(d, 2 * math.pi - d)

            sweep_ccw = (exit_angle - entry_angle) % (2 * math.pi)
            sweep_cw = sweep_ccw - 2 * math.pi
            mid_ccw = entry_angle + sweep_ccw / 2
            mid_cw = entry_angle + sweep_cw / 2
            sweep = sweep_ccw if _angular_distance(mid_ccw, preferred_angle) <= _angular_distance(mid_cw, preferred_angle) else sweep_cw

            t_at_transition = (ROUNDABOUT_OCCUPANCY_RADIUS - gap) / (2 * ROUNDABOUT_OCCUPANCY_RADIUS)
            t_at_transition = max(0.0, min(1.0, t_at_transition))
            base_angle = entry_angle + sweep * t_at_transition

            total_excursion = sim.track.feature_circumference(feature.feature_id)
            progress = 1.0 - (vehicle.roundabout_excursion_remaining / total_excursion)
            progress = max(0.0, min(1.0, progress))
            turn_sign = 1 if vehicle.direction >= 0 else -1
            angle = base_angle + turn_sign * progress * 2 * math.pi
            ring_radius = feature.radius * 0.775   # midpoint of the drivable ring (island sits at 0.55x, outer kerb at 1.0x - see the matching 0.55 factor in draw_features), so the marker stays on the circulating carriageway itself rather than outside the outer kerb
            return cx + ring_radius * math.cos(angle), cy + ring_radius * math.sin(angle)
        return real_x, real_y

    for feature in sim.track.features:
        if feature.feature_type != "roundabout":
            continue
        gap = sim.track.signed_gap(vehicle.s, feature.position)
        if abs(gap) > ROUNDABOUT_OCCUPANCY_RADIUS:
            continue

        cx, cy, _h = sim.track.position_at(feature.position)
        _, _, road_heading = sim.track.position_at(feature.position)
        entry_x, entry_y, _h = sim.track.lane_position_at(feature.position - ROUNDABOUT_OCCUPANCY_RADIUS, vehicle.lane)
        exit_x, exit_y, _h = sim.track.lane_position_at(feature.position + ROUNDABOUT_OCCUPANCY_RADIUS, vehicle.lane)
        entry_angle = math.atan2(entry_y - cy, entry_x - cx)
        exit_angle = math.atan2(exit_y - cy, exit_x - cx)

        # There are two possible arcs between entry and exit (going
        # clockwise or counter-clockwise) - which one is correct
        # depends on the lane, since lane 0 and lane 1 sit on opposite
        # sides of the road and should use opposite halves of the ring.
        side = 1 if vehicle.lane == 0 else -1
        preferred_angle = math.atan2(-side * math.cos(road_heading), side * math.sin(road_heading))

        def _angular_distance(a, b):
            d = (a - b) % (2 * math.pi)
            return min(d, 2 * math.pi - d)

        sweep_ccw = (exit_angle - entry_angle) % (2 * math.pi)
        sweep_cw = sweep_ccw - 2 * math.pi
        mid_ccw = entry_angle + sweep_ccw / 2
        mid_cw = entry_angle + sweep_cw / 2
        sweep = sweep_ccw if _angular_distance(mid_ccw, preferred_angle) <= _angular_distance(mid_cw, preferred_angle) else sweep_cw

        # t=0 at entry (gap=+RADIUS, vehicle just entering the zone),
        # t=1 at exit (gap=-RADIUS, about to leave it) - previously
        # inverted (1.0 at entry, 0.0 at exit), so the marker started
        # at the exit point and swept backwards as the vehicle actually
        # moved forwards, which combined with the over-rotated sweep
        # above produced the reported teleport-and-spin appearance.
        t = (ROUNDABOUT_OCCUPANCY_RADIUS - gap) / (2 * ROUNDABOUT_OCCUPANCY_RADIUS)
        t = max(0.0, min(1.0, t))
        angle = entry_angle + sweep * t
        ring_radius = feature.radius * 0.775   # midpoint of the drivable ring - see the matching comment in the excursion branch above
        return cx + ring_radius * math.cos(angle), cy + ring_radius * math.sin(angle)

    return real_x, real_y


def vehicle_is_attacked(result) -> bool:
    for reading in (result.radar_reading, result.camera_reading, result.lidar_reading):
        if reading is not None and reading.is_attacked:
            return True
    return False


def draw_vehicles(screen, camera, font, latest_by_id, sim):
    screen_positions = {}
    for vid, r in latest_by_id.items():
        vx, vy = roundabout_swept_position(sim, vid, *r.position)
        screen_positions[vid] = camera.to_screen(vx, vy)

    for vehicle_id, result in latest_by_id.items():
        sx, sy = screen_positions[vehicle_id]
        if result.collision:
            colour = COLOR_VEHICLE_CRASHED
        elif vehicle_is_attacked(result):
            colour = COLOR_VEHICLE_ATTACKED
        else:
            colour = COLOR_VEHICLE
        pygame.draw.circle(screen, colour, (sx, sy), VEHICLE_RADIUS)
        pygame.draw.circle(screen, COLOR_TEXT, (sx, sy), VEHICLE_RADIUS, 1)

    # labels drawn in a second pass, staggered vertically for any
    # vehicles whose markers are close enough on screen to make
    # overlapping labels illegible (e.g. two vehicles at the same
    # collision point) - a real crash puts vehicles on top of
    # each other, so the markers themselves are left overlapping,
    # only the text labels are offset
    close_threshold_px = VEHICLE_RADIUS * 3
    placed_label_ids = list(latest_by_id.keys())
    for i, vehicle_id in enumerate(placed_label_ids):
        sx, sy = screen_positions[vehicle_id]
        stack_index = 0
        for other_id in placed_label_ids[:i]:
            ox, oy = screen_positions[other_id]
            if math.hypot(sx - ox, sy - oy) < close_threshold_px:
                stack_index += 1
        label = font.render(vehicle_id, True, COLOR_TEXT_DIM)
        screen.blit(label, (sx + VEHICLE_RADIUS + 2, sy - 8 + stack_index * 14))


def draw_sidebar(screen, font, font_bold, sim, latest_by_id, config, paused, speed_multiplier):
    x0 = WINDOW_WIDTH - SIDEBAR_WIDTH
    pygame.draw.rect(screen, COLOR_SIDEBAR_BG, (x0, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))

    y = 16
    title = config.name
    while font_bold.size(title)[0] > SIDEBAR_WIDTH - 32 and len(title) > 3:
        title = title[:-4] + "..."
    screen.blit(font_bold.render(title, True, COLOR_TEXT), (x0 + 16, y))
    y += 28

    state_str = "PAUSED" if paused else f"{speed_multiplier:g}x"
    screen.blit(font.render(f"t={sim.time:5.1f}s / {config.duration:.0f}s   {state_str}", True, COLOR_TEXT_DIM), (x0 + 16, y))
    y += 26

    pygame.draw.line(screen, COLOR_TRACK, (x0 + 16, y), (WINDOW_WIDTH - 16, y), 1)
    y += 14

    for vehicle_id in sorted(latest_by_id.keys()):
        result = latest_by_id[vehicle_id]
        screen.blit(font_bold.render(vehicle_id, True, COLOR_TEXT), (x0 + 16, y))
        y += 20

        speed_line = f"  {ms_to_mph(result.speed):5.1f} mph   lane {result.lane}"
        screen.blit(font.render(speed_line, True, COLOR_TEXT_DIM), (x0 + 16, y))
        y += 18

        belief = result.fused_belief
        if belief.obstacle_present:
            reacting = result.ground_truth_kind or "?"
            belief_line = f"  reacting to: {reacting}"
        else:
            belief_line = "  no obstacle believed"
        screen.blit(font.render(belief_line, True, COLOR_TEXT_DIM), (x0 + 16, y))
        y += 18

        if vehicle_is_attacked(result):
            screen.blit(font.render("  UNDER ATTACK", True, COLOR_VEHICLE_ATTACKED), (x0 + 16, y))
            y += 18

        if result.collision:
            sev = result.severity or (sim.vehicles[vehicle_id].severity if vehicle_id in sim.vehicles else None)
            screen.blit(font.render(f"  COLLISION ({sev})", True, COLOR_VEHICLE_CRASHED), (x0 + 16, y))
            y += 18

        y += 10

    y = WINDOW_HEIGHT - 90
    pygame.draw.line(screen, COLOR_TRACK, (x0 + 16, y), (WINDOW_WIDTH - 16, y), 1)
    y += 10
    for line in ["SPACE  pause/resume", "UP/DOWN  speed x2 / x0.5", "R  restart   ESC  quit"]:
        screen.blit(font.render(line, True, COLOR_TEXT_DIM), (x0 + 16, y))
        y += 18


def list_scenarios() -> list[str]:
    """Every .yaml file in scenarios/ sorted for the picker
    menu"""
    import glob
    return sorted(glob.glob(os.path.join("scenarios", "*.yaml")))


def run_menu(screen, clock, font, font_bold) -> str | None:
    from editor_widgets import ScrollPanel, COLOR_WIDGET_BG_HOVER

    paths = list_scenarios()
    row_height = 34
    panel_rect = (40, 90, WINDOW_WIDTH - 80, WINDOW_HEIGHT - 130)
    panel = ScrollPanel(panel_rect, max(len(paths) * row_height, panel_rect[3]))
    selected_index = 0 if paths else None

    running = True
    chosen_path = None
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                elif event.key == pygame.K_DOWN and paths:
                    selected_index = min((selected_index or 0) + 1, len(paths) - 1)
                elif event.key == pygame.K_UP and paths:
                    selected_index = max((selected_index or 0) - 1, 0)
                elif event.key == pygame.K_RETURN and selected_index is not None:
                    chosen_path = paths[selected_index]
                    running = False
            elif event.type == pygame.MOUSEWHEEL:
                panel.handle_scroll(event)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                translated = panel.translate_event(event)
                if translated is not None:
                    row = translated.pos[1] // row_height
                    if 0 <= row < len(paths):
                        selected_index = row
                        chosen_path = paths[row]
                        running = False

        screen.fill(COLOR_BG)
        title = font_bold.render("Select a scenario", True, COLOR_TEXT)
        screen.blit(title, (40, 30))
        hint = font.render("UP/DOWN + ENTER, or click - ESC to quit", True, COLOR_TEXT_DIM)
        screen.blit(hint, (40, 60))

        panel.surface.fill(COLOR_SIDEBAR_BG)
        for i, path in enumerate(paths):
            row_rect = pygame.Rect(0, i * row_height, panel.rect.width, row_height)
            if i == selected_index:
                pygame.draw.rect(panel.surface, COLOR_WIDGET_BG_HOVER, row_rect)
            name = os.path.splitext(os.path.basename(path))[0]
            text_surf = font.render(name, True, COLOR_TEXT)
            panel.surface.blit(text_surf, (12, row_rect.y + (row_height - text_surf.get_height()) // 2))
        panel.blit_to(screen)

        pygame.display.flip()

    return chosen_path


def run_playback(screen, clock, font, font_bold, scenario_path: str) -> None:
    config = load_scenario(scenario_path)
    sim = Simulation(config)
    pygame.display.set_caption(f"AV Cyber-Attack Simulator - {config.name}")

    camera = Camera(sim.track, WINDOW_WIDTH - SIDEBAR_WIDTH, WINDOW_HEIGHT, TRACK_MARGIN)
    geometry = build_static_geometry(sim.track)
    buildings = generate_buildings(sim.track)

    paused = False
    speed_multiplier = 1.0
    accumulator = 0.0
    latest_by_id: dict = {}

    running = True
    while running:
        dt_real = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    speed_multiplier = min(speed_multiplier * 2, 16.0)
                elif event.key == pygame.K_DOWN:
                    speed_multiplier = max(speed_multiplier / 2, 0.25)
                elif event.key == pygame.K_r:
                    sim = Simulation(config)
                    latest_by_id = {}
                    accumulator = 0.0
                    paused = False

        if not paused and sim.time < config.duration:
            accumulator += dt_real * speed_multiplier
            while accumulator >= config.timestep and sim.time < config.duration:
                results = sim.step(config.timestep)
                for result in results:
                    latest_by_id[result.vehicle_id] = result
                # prune despawned vehicles - sim.vehicles only ever
                # holds currently-existing ones (see
                # _despawn_completed_laps in engine.py)
                latest_by_id = {vid: r for vid, r in latest_by_id.items() if vid in sim.vehicles}
                accumulator -= config.timestep

        screen.fill(COLOR_BG)
        draw_buildings(screen, camera, buildings)
        draw_track(screen, camera, geometry)
        draw_features(screen, camera, sim, font)
        draw_vehicles(screen, camera, font, latest_by_id, sim)
        draw_sidebar(screen, font, font_bold, sim, latest_by_id, config, paused, speed_multiplier)
        pygame.display.flip()


def _draw_feature_preview_markers(screen, camera, track, font, in_progress=None):
    """Draws a placeholder circle for each track feature, purely
    so an added junction or roundabout is actually visible in the
    editor's live preview

    a junction gets a dot sitting right on the road at its position
    a roundabout gets a hollow circle outline sized to
    its own actual radius"""
    def _draw_one(feature_type, position, radius, confirmed):
        x, y, _heading = track.position_at(position)
        sx, sy = camera.to_screen(x, y)
        width = 3 if confirmed else 2

        if feature_type == "roundabout":
            outer_r = max(int((radius or 25.0) * camera.scale), 10)
            pygame.draw.circle(screen, COLOR_FEATURE_PREVIEW, (sx, sy), outer_r, width)
        else:
            dot_r = 9 if confirmed else 7
            if confirmed:
                pygame.draw.circle(screen, COLOR_FEATURE_PREVIEW, (sx, sy), dot_r)
            else:
                pygame.draw.circle(screen, COLOR_FEATURE_PREVIEW, (sx, sy), dot_r, width)
        label_text = feature_type if confirmed else f"{feature_type} (not yet added)"
        label = font.render(label_text, True, COLOR_FEATURE_PREVIEW)
        screen.blit(label, (sx + 12, sy - 8))

    for f in track.features:
        _draw_one(f.feature_type, f.position, getattr(f, "radius", None), confirmed=True)

    if in_progress is not None and in_progress.get("feature_id"):
        radius = in_progress.get("radius") if in_progress.get("feature_type") == "roundabout" else None
        _draw_one(in_progress.get("feature_type", "junction"), in_progress.get("position", 0.0), radius, confirmed=False)


def _draw_vehicle_preview_markers(screen, camera, track, vehicles, font, in_progress=None):
    for v in vehicles:
        try:
            lane = v.get("lane", 0)
            x, y, _h = track.lane_position_at(v.get("start_distance", 0.0), lane)
        except Exception:
            continue
        sx, sy = camera.to_screen(x, y)
        pygame.draw.circle(screen, COLOR_VEHICLE, (sx, sy), 8)
        pygame.draw.circle(screen, (255, 255, 255), (sx, sy), 8, 1)
        label = font.render(v.get("vehicle_id", "?"), True, COLOR_TEXT_DIM)
        screen.blit(label, (sx + 10, sy - 7))

    if in_progress is not None:
        try:
            lane = in_progress.get("lane", 0)
            x, y, _h = track.lane_position_at(in_progress.get("start_distance", 0.0), lane)
            sx, sy = camera.to_screen(x, y)
            pygame.draw.circle(screen, COLOR_VEHICLE, (sx, sy), 8, 2)
            name = in_progress.get("vehicle_id") or "new vehicle"
            label = font.render(f"{name} (not yet added)", True, COLOR_VEHICLE)
            screen.blit(label, (sx + 10, sy - 7))
        except Exception:
            pass


def run_editor(screen, clock, font, font_bold, initial_path: str | None = None) -> None:
    from editor_widgets import TextField, Button, Dropdown, ScrollPanel, Slider, COLOR_WIDGET_BG_HOVER
    from editable_scenario import EditableScenario, FUSION_POLICIES, ATTACK_TYPES, HAZARD_TYPES, PEDESTRIAN_TYPES, FEATURE_TYPES
    from sim_core.track import Track

    LEFT_WIDTH = 300
    RIGHT_WIDTH = SIDEBAR_WIDTH

    editable = EditableScenario.from_yaml_path(initial_path) if initial_path else EditableScenario()

    paths = list_scenarios()
    scenario_labels = {p: os.path.splitext(os.path.basename(p))[0] for p in paths}
    scenario_dropdown = Dropdown((16, 46, LEFT_WIDTH - 32, 26), paths, value=initial_path, labels=scenario_labels, allow_empty=True)

    state = {"message": ""}

    def _do_load():
        if scenario_dropdown.value:
            nonlocal editable
            editable = EditableScenario.from_yaml_path(scenario_dropdown.value)
            name_field.value = editable.name
            duration_field.value = str(editable.duration)
            tf_straight_length.value = str(editable.track_straight_length)
            tf_radius.value = str(editable.track_radius)
            if editable.spawner:
                sf_feature.value = editable.spawner.get("feature_id")
                sf_max_concurrent.value = str(editable.spawner.get("max_concurrent", 5))
                sf_speed.value = str(editable.spawner.get("speed_mph", 30.0))
                sf_policy.value = editable.spawner.get("fusion_policy", FUSION_POLICIES[0])
            state["message"] = f"loaded {os.path.basename(scenario_dropdown.value)}"
        else:
            state["message"] = "pick a scenario from the dropdown first"

    def _do_new():
        nonlocal editable
        editable = EditableScenario()
        name_field.value = editable.name
        duration_field.value = str(editable.duration)
        tf_straight_length.value = str(editable.track_straight_length)
        tf_radius.value = str(editable.track_radius)
        state["message"] = "started a new scenario"

    def _do_save():
        problems = editable.validate()
        if problems:
            state["message"] = "can't save: " + "; ".join(problems)
            return
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in editable.name.lower()) or "scenario"
        out_path = os.path.join("scenarios", f"{safe_name}.yaml")
        editable.save(out_path)
        state["message"] = f"saved {out_path}"

        refreshed_paths = list_scenarios()
        scenario_dropdown.options = refreshed_paths
        scenario_dropdown.labels = {p: os.path.splitext(os.path.basename(p))[0] for p in refreshed_paths}
        scenario_dropdown.value = out_path
        scenario_dropdown.scroll_offset = 0

    def _do_play():
        import tempfile
        problems = editable.validate()
        if problems:
            state["message"] = "can't play: " + "; ".join(problems)
            return
        preview_path = os.path.join(tempfile.gettempdir(), "_editor_play_preview.yaml")
        editable.save(preview_path)
        state["play_requested"] = preview_path

    load_button = Button((16, 78, 62, 26), "Load", _do_load)
    new_button = Button((82, 78, 62, 26), "New", _do_new)
    save_button = Button((148, 78, 62, 26), "Save", _do_save)
    play_button = Button((214, 78, 70, 26), "Play", _do_play)

    name_field = TextField((16, 122, LEFT_WIDTH - 32, 26), value=editable.name)
    duration_field = TextField((16, 166, 100, 26), value=str(editable.duration), numeric=True)

    # Vehicle add-form - persistent widgets so in-progress typing/slider
    # position survives across frames
    vf_name = TextField((16, 0, 131, 26), value="")
    vf_speed = TextField((155, 0, 89, 26), value="30", numeric=True)
    vf_policy = Dropdown((16, 0, 131, 26), FUSION_POLICIES, value=FUSION_POLICIES[0])
    vf_lane = Dropdown((155, 0, 89, 26), ["0", "1"], value="0")
    vf_position = Slider((16, 0, 140, 20), 0.0, editable.track_total_length(), value=0.0, label="Start distance", show_value=False)
    vf_position_text = TextField((166, 0, 88, 26), value="0.0", numeric=True)

    def _add_vehicle():
        vid = vf_name.value.strip() or f"vehicle_{len(editable.vehicles) + 1}"
        try:
            speed = float(vf_speed.value)
        except ValueError:
            state["message"] = "vehicle speed must be a number"
            return
        editable.vehicles.append({
            "vehicle_id": vid, "start_distance": round(vf_position.value, 1), "start_speed_mph": speed,
            "fusion_policy": vf_policy.value, "lane": int(vf_lane.value), "direction": -1 if vf_lane.value == "1" else 1,
        })
        vf_name.value = ""
        state["message"] = f"added vehicle {vid!r}"

    add_vehicle_button = Button((16, 0, 190, 26), "Add vehicle", _add_vehicle)

    def _make_remove_sync(list_getter, remove_buttons, label_fn):
        def _sync():
            item_list = list_getter()
            current_ids = {id(item) for item in item_list}
            for stale_id in [k for k in remove_buttons if k not in current_ids]:
                del remove_buttons[stale_id]
            for item in item_list:
                if id(item) not in remove_buttons:
                    def _make_remover(it=item):
                        return lambda: (list_getter().remove(it),
                                        state.__setitem__("message", f"removed {label_fn(it)}"))

                    remove_buttons[id(item)] = Button((0, 0, 60, 22), "Remove", _make_remover(), danger=True)

        return _sync

    vehicle_remove_buttons: dict[int, Button] = {}
    _sync_vehicle_removes = _make_remove_sync(lambda: editable.vehicles, vehicle_remove_buttons, lambda v: repr(v.get("vehicle_id")))
    # Attacks form
    af_type = Dropdown((16, 0, 238, 26), ATTACK_TYPES, value=ATTACK_TYPES[0])
    af_target = Dropdown((16, 0, 238, 26), [], value=None, allow_empty=True)
    af_trigger_feature = Dropdown((16, 0, 190, 26), [], value=None, allow_empty=True)
    af_trigger_distance = TextField((16, 0, 70, 26), value="30", numeric=True)
    af_duration = TextField((155, 0, 89, 26), value="1.0", numeric=True)
    af_extra1 = TextField((16, 0, 238, 26), value="0", numeric=True)
    af_extra2 = TextField((16, 0, 238, 26), value="0", numeric=True)

    def _attack_extra_field_labels(attack_type):
        if attack_type in ("radar_spoof", "lidar_spoof"):
            return "Fabricated distance (m)", "Fabricated velocity (m/s)"
        if attack_type == "camera_phantom":
            return "Fabricated distance (m)", None
        if attack_type == "gps_spoof":
            return "GPS offset X (m)", "GPS offset Y (m)"
        return None, None   # jam attacks

    def _add_attack():
        if af_target.value is None:
            state["message"] = "add a vehicle first, then pick a target"
            return
        try:
            trigger_distance = float(af_trigger_distance.value)
            duration = float(af_duration.value)
        except ValueError:
            state["message"] = "trigger distance and duration must be numbers"
            return
        attack = {"type": af_type.value, "target_vehicle": af_target.value, "duration": duration}
        if af_trigger_feature.value is not None:
            attack["trigger_before_feature"] = af_trigger_feature.value
            attack["trigger_distance"] = trigger_distance
        else:
            state["message"] = "pick a trigger feature first"
            return
        label1, label2 = _attack_extra_field_labels(af_type.value)
        try:
            if af_type.value in ("radar_spoof", "lidar_spoof"):
                attack["spoofed_distance"] = float(af_extra1.value)
                attack["spoofed_velocity"] = float(af_extra2.value)
            elif af_type.value == "camera_phantom":
                attack["phantom_distance"] = float(af_extra1.value)
            elif af_type.value == "gps_spoof":
                attack["offset"] = [float(af_extra1.value), float(af_extra2.value)]
        except ValueError:
            state["message"] = "attack parameters must be numbers"
            return
        editable.attacks.append(attack)
        state["message"] = f"added {af_type.value} attack on {af_target.value!r}"

    add_attack_button = Button((16, 0, 190, 26), "Add attack", _add_attack)
    attack_remove_buttons: dict[int, Button] = {}
    _sync_attack_removes = _make_remove_sync(lambda: editable.attacks, attack_remove_buttons, lambda a: f"{a.get('type')} on {a.get('target_vehicle')!r}")
    # Hazards form
    hf_type = Dropdown((16, 0, 238, 26), HAZARD_TYPES, value=HAZARD_TYPES[0])
    hf_feature = Dropdown((16, 0, 238, 26), [], value=None, allow_empty=True)
    hf_start_time = TextField((16, 0, 70, 26), value="0", numeric=True)
    hf_duration = TextField((155, 0, 89, 26), value="10", numeric=True)
    hf_lane = Dropdown((16, 0, 238, 26), ["0", "1"], value="0")
    hf_pedestrian_type = Dropdown((16, 0, 238, 26), PEDESTRIAN_TYPES, value=PEDESTRIAN_TYPES[0])

    def _hazard_arrival_hints():
        if hf_feature.value is None:
            return []
        feature_pos = editable.feature_position(hf_feature.value)
        if feature_pos is None:
            return []
        if not editable.vehicles:
            return ["(no vehicles added yet)"]
        hints = []
        for v in editable.vehicles:
            try:
                speed_ms = float(v.get("start_speed_mph", 0)) * 0.44704
                start = float(v.get("start_distance", 0))
            except (TypeError, ValueError):
                continue
            if speed_ms <= 0:
                continue
            direction = v.get("direction", 1)
            distance = (feature_pos - start) if direction == 1 else (start - feature_pos)
            vid = v.get("vehicle_id", "?")
            if distance < 0:
                hints.append(f"{vid}: already past at t=0")
            else:
                hints.append(f"{vid}: ~{distance / speed_ms:.1f}s")
        return hints

    def _add_hazard():
        if hf_feature.value is None:
            state["message"] = "pick a feature for this hazard first"
            return
        try:
            start_time = float(hf_start_time.value)
            duration = float(hf_duration.value)
        except ValueError:
            state["message"] = "start time and duration must be numbers"
            return
        hazard = {"type": hf_type.value, "feature_id": hf_feature.value, "start_time": start_time, "duration": duration}
        if hf_type.value == "obstacle_in_road":
            hazard["lane"] = int(hf_lane.value)
        elif hf_type.value == "pedestrian_crossing":
            hazard["pedestrian_type"] = hf_pedestrian_type.value
        editable.hazards.append(hazard)
        state["message"] = f"added {hf_type.value} hazard at {hf_feature.value!r}"

    add_hazard_button = Button((16, 0, 190, 26), "Add hazard", _add_hazard)
    hazard_remove_buttons: dict[int, Button] = {}
    _sync_hazard_removes = _make_remove_sync(lambda: editable.hazards, hazard_remove_buttons, lambda h: f"{h.get('type')} at {h.get('feature_id')!r}")

    # Track shape form
    tf_straight_length = TextField((16, 0, 89, 26), value=str(editable.track_straight_length), numeric=True)
    tf_radius = TextField((155, 0, 89, 26), value=str(editable.track_radius), numeric=True)

    def _sync_slider_and_text(slider, text_field):
        if text_field.focused:
            try:
                val = float(text_field.value)
                slider.value = max(slider.min_value, min(slider.max_value, val))
            except ValueError:
                pass
        else:
            text_field.value = f"{slider.value:.1f}"

    def _sync_track_dimensions():
        try:
            editable.track_straight_length = float(tf_straight_length.value)
        except ValueError:
            pass
        try:
            editable.track_radius = float(tf_radius.value)
        except ValueError:
            pass

    ff_id = TextField((16, 0, 238, 26), value="")
    ff_type = Dropdown((16, 0, 238, 26), FEATURE_TYPES, value=FEATURE_TYPES[0])
    ff_position = Slider((16, 0, 140, 20), 0.0, editable.track_total_length(), value=0.0, label="Position", show_value=False)
    ff_position_text = TextField((166, 0, 88, 26), value="0.0", numeric=True)
    ff_radius = TextField((16, 0, 238, 26), value="25", numeric=True)

    def _add_feature():
        fid = ff_id.value.strip()
        if not fid:
            state["message"] = "give the feature an ID first"
            return
        feature = {"feature_id": fid, "feature_type": ff_type.value, "position": round(ff_position.value, 1)}
        if ff_type.value == "roundabout":
            try:
                feature["radius"] = float(ff_radius.value)
            except ValueError:
                state["message"] = "roundabout radius must be a number"
                return
        editable.track_features.append(feature)
        ff_id.value = ""
        state["message"] = f"added {ff_type.value} {fid!r}"

    add_feature_button = Button((16, 0, 190, 26), "Add feature", _add_feature)
    feature_remove_buttons: dict[int, Button] = {}
    _sync_feature_removes = _make_remove_sync(lambda: editable.track_features, feature_remove_buttons, lambda f: repr(f.get("feature_id")))

    # Spawner form
    sf_feature = Dropdown((16, 0, 238, 26), [], value=None, allow_empty=True)
    sf_max_concurrent = TextField((16, 0, 89, 26), value="5", numeric=True)
    sf_speed = TextField((155, 0, 89, 26), value="30", numeric=True)
    sf_policy = Dropdown((16, 0, 238, 26), FUSION_POLICIES, value=FUSION_POLICIES[0])

    def _toggle_spawner():
        if editable.spawner is None:
            if sf_feature.value is None:
                state["message"] = "no track feature available for a spawner yet"
                return
            editable.spawner = {"feature_id": sf_feature.value, "max_concurrent": 5, "speed_mph": 30.0, "fusion_policy": FUSION_POLICIES[0]}
            state["message"] = "spawner enabled"
        else:
            editable.spawner = None
            state["message"] = "spawner disabled"

    toggle_spawner_button = Button((16, 0, 190, 26), "Enable spawner", _toggle_spawner)

    def _sync_spawner_fields():
        if editable.spawner is None:
            return
        editable.spawner["feature_id"] = sf_feature.value
        try:
            editable.spawner["max_concurrent"] = int(float(sf_max_concurrent.value))
        except ValueError:
            pass
        try:
            editable.spawner["speed_mph"] = float(sf_speed.value)
        except ValueError:
            pass
        editable.spawner["fusion_policy"] = sf_policy.value

    editor_panel_rect = pygame.Rect(16, 218, LEFT_WIDTH - 32, WINDOW_HEIGHT - 236)
    editor_panel = ScrollPanel(editor_panel_rect, editor_panel_rect.height)


    running = True
    while running:
        clock.tick(FPS)
        _sync_vehicle_removes()
        _sync_attack_removes()
        _sync_hazard_removes()
        _sync_feature_removes()

        vehicle_ids = [v.get("vehicle_id") for v in editable.vehicles]
        feature_ids = editable.available_feature_ids()
        af_target.options = vehicle_ids
        if af_target.value not in vehicle_ids:
            af_target.value = vehicle_ids[0] if vehicle_ids else None
        af_trigger_feature.options = feature_ids
        if af_trigger_feature.value not in feature_ids:
            af_trigger_feature.value = feature_ids[0] if feature_ids else None
        hf_feature.options = feature_ids
        if hf_feature.value not in feature_ids:
            hf_feature.value = feature_ids[0] if feature_ids else None
        sf_feature.options = feature_ids
        if sf_feature.value not in feature_ids:
            sf_feature.value = feature_ids[0] if feature_ids else None

        track_length = editable.track_total_length()
        if not vf_position.dragging:
            vf_position.max_value = track_length
            vf_position.value = min(vf_position.value, track_length)
        if not ff_position.dragging:
            ff_position.max_value = track_length
            ff_position.value = min(ff_position.value, track_length)
        _sync_slider_and_text(vf_position, vf_position_text)
        _sync_slider_and_text(ff_position, ff_position_text)

        y = 4
        add_vehicle_button.rect.y = y + 26
        y = add_vehicle_button.rect.y + 26 + 20   # + button height + gap before next label
        vf_name.rect.y = y
        vf_speed.rect.y = y
        y += 26 + 28
        vf_policy.rect.y = y
        vf_lane.rect.y = y
        y += 26 + 24
        vf_position.rect.y = y
        vf_position_text.rect.y = y - 3
        y += 20 + 24
        vehicle_row_tops = []
        for v in editable.vehicles:
            vehicle_row_tops.append(y)
            vehicle_remove_buttons[id(v)].rect.topleft = (editor_panel_rect.width - 76, y - 1)
            y += 32
        y += 16   # gap before next section header

        attacks_header_y = y
        y += 26   # "Attacks (n)" header
        add_attack_button.rect.y = y
        y += 26 + 28
        af_type.rect.y = y
        y += 26 + 28
        af_target.rect.y = y
        y += 26 + 28
        af_trigger_feature.rect.y = y
        y += 26 + 28
        af_trigger_distance.rect.y = y
        af_duration.rect.y = y
        y += 26 + 28
        label1, label2 = _attack_extra_field_labels(af_type.value)
        af_extra1.rect.y = y
        if label1:
            y += 26 + 28
        af_extra2.rect.y = y
        if label2:
            y += 26 + 24
        attack_row_tops = []
        for a in editable.attacks:
            attack_row_tops.append(y)
            attack_remove_buttons[id(a)].rect.topleft = (editor_panel_rect.width - 76, y - 1)
            y += 32
        y += 16

        hazards_header_y = y
        y += 26   # "Hazards (n)" header
        add_hazard_button.rect.y = y
        y += 26 + 28
        hf_type.rect.y = y
        y += 26 + 28
        hf_feature.rect.y = y
        y += 26 + 28
        hf_start_time.rect.y = y
        hf_duration.rect.y = y
        y += 26 + 20
        arrival_hints = _hazard_arrival_hints()
        arrival_hints_y = y
        if arrival_hints:
            y += len(arrival_hints) * 16 + 32
        hf_lane.rect.y = y
        hf_pedestrian_type.rect.y = y
        if hf_type.value in ("obstacle_in_road", "pedestrian_crossing"):
            y += 26 + 24
        hazard_row_tops = []
        for h in editable.hazards:
            hazard_row_tops.append(y)
            hazard_remove_buttons[id(h)].rect.topleft = (editor_panel_rect.width - 76, y - 1)
            y += 32
        y += 16

        track_header_y = y
        y += 26 + 16   # "Track shape" header - +16 extra since this goes straight to a labeled field, unlike other sections which have a button first
        tf_straight_length.rect.y = y
        tf_radius.rect.y = y
        y += 26 + 28
        add_feature_button.rect.y = y
        y += 26 + 28
        ff_id.rect.y = y
        y += 26 + 28
        ff_type.rect.y = y
        y += 26 + 28
        ff_position.rect.y = y
        ff_position_text.rect.y = y - 3
        y += 20 + 24
        if ff_type.value == "roundabout":
            ff_radius.rect.y = y
            y += 26 + 24
        feature_row_tops = []
        for f in editable.track_features:
            feature_row_tops.append(y)
            feature_remove_buttons[id(f)].rect.topleft = (editor_panel_rect.width - 76, y - 1)
            y += 32
        y += 16

        spawner_header_y = y
        y += 26   # "Spawner" header
        toggle_spawner_button.rect.y = y
        toggle_spawner_button.label = "Disable spawner" if editable.spawner else "Enable spawner"
        y += 26 + 28
        if editable.spawner:
            sf_feature.rect.y = y
            y += 26 + 28
            sf_max_concurrent.rect.y = y
            sf_speed.rect.y = y
            y += 26 + 28
            sf_policy.rect.y = y
            y += 26 + 28

        panel_content_height = y + 10 + 250

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
                continue

            consumed = scenario_dropdown.handle_event(event)
            if not consumed:
                consumed = load_button.handle_event(event)
            if not consumed:
                consumed = new_button.handle_event(event)
            if not consumed:
                consumed = save_button.handle_event(event)
            if not consumed:
                consumed = play_button.handle_event(event)
            if not consumed:
                name_field.handle_event(event)
                duration_field.handle_event(event)
            editable.name = name_field.value
            try:
                editable.duration = float(duration_field.value)
            except ValueError:
                pass   # leave editable.duration unchanged while mid-edit (e.g. field temporarily empty or just "-")

            if event.type == pygame.MOUSEWHEEL and not scenario_dropdown.open:
                editor_panel.handle_scroll(event)
            translated = editor_panel.translate_event(event) if hasattr(event, "pos") else event
            if not consumed and translated is not None:
                active_dropdowns = [vf_policy, vf_lane, af_type, af_target, af_trigger_feature, hf_type, hf_feature, ff_type, sf_feature, sf_policy]
                if hf_type.value == "obstacle_in_road":
                    active_dropdowns.append(hf_lane)
                elif hf_type.value == "pedestrian_crossing":
                    active_dropdowns.append(hf_pedestrian_type)
                for dd in active_dropdowns:
                    consumed = dd.handle_event(translated)
                    if consumed:
                        break
                if not consumed:
                    consumed = add_vehicle_button.handle_event(translated)
                if not consumed:
                    consumed = vf_name.handle_event(translated)
                if not consumed:
                    consumed = vf_speed.handle_event(translated)
                if not consumed:
                    consumed = vf_position.handle_event(translated)
                if not consumed:
                    consumed = vf_position_text.handle_event(translated)
                _sync_slider_and_text(vf_position, vf_position_text)
                if not consumed:
                    consumed = add_attack_button.handle_event(translated)
                if not consumed:
                    consumed = af_trigger_distance.handle_event(translated)
                if not consumed:
                    consumed = af_duration.handle_event(translated)
                if not consumed:
                    consumed = af_extra1.handle_event(translated)
                if not consumed:
                    consumed = af_extra2.handle_event(translated)
                if not consumed:
                    consumed = add_hazard_button.handle_event(translated)
                if not consumed:
                    consumed = hf_start_time.handle_event(translated)
                if not consumed:
                    consumed = hf_duration.handle_event(translated)
                if not consumed:
                    consumed = tf_straight_length.handle_event(translated)
                if not consumed:
                    consumed = tf_radius.handle_event(translated)
                _sync_track_dimensions()
                if not consumed:
                    consumed = add_feature_button.handle_event(translated)
                if not consumed:
                    consumed = ff_id.handle_event(translated)
                if not consumed:
                    consumed = ff_position.handle_event(translated)
                if not consumed:
                    consumed = ff_position_text.handle_event(translated)
                _sync_slider_and_text(ff_position, ff_position_text)
                if not consumed and ff_type.value == "roundabout":
                    consumed = ff_radius.handle_event(translated)
                if not consumed:
                    consumed = toggle_spawner_button.handle_event(translated)
                if not consumed and editable.spawner:
                    consumed = sf_max_concurrent.handle_event(translated)
                if not consumed and editable.spawner:
                    consumed = sf_speed.handle_event(translated)
                _sync_spawner_fields()
                if not consumed:
                    for btn_dict in (vehicle_remove_buttons, attack_remove_buttons, hazard_remove_buttons, feature_remove_buttons):
                        for btn in btn_dict.values():
                            if btn.handle_event(translated):
                                consumed = True
                                break
                        if consumed:
                            break

        if state.get("play_requested"):
            to_play = state.pop("play_requested")
            run_playback(screen, clock, font, font_bold, to_play)
            continue

        # live preview track
        from sim_core.track import TrackFeature
        custom_features = [TrackFeature(f.get("feature_id"), f.get("feature_type"), f.get("position", 0.0), f.get("radius"))
                            for f in editable.track_features if f.get("feature_id")]
        preview_track = Track(straight_length=editable.track_straight_length, radius=editable.track_radius, features=custom_features)
        preview_camera = Camera(preview_track, WINDOW_WIDTH - LEFT_WIDTH - RIGHT_WIDTH, WINDOW_HEIGHT, TRACK_MARGIN)
        # offset the preview camera's own coordinate space into the middle third of the window
        preview_camera_offset_x = LEFT_WIDTH

        screen.fill(COLOR_BG)

        # left panel
        pygame.draw.rect(screen, COLOR_SIDEBAR_BG, (0, 0, LEFT_WIDTH, WINDOW_HEIGHT))
        screen.blit(font_bold.render("Scenario", True, COLOR_TEXT), (16, 16))
        scenario_dropdown.draw_closed(screen, font)
        load_button.draw(screen, font)
        new_button.draw(screen, font)
        save_button.draw(screen, font)
        play_button.draw(screen, font)
        screen.blit(font.render("Name:", True, COLOR_TEXT_DIM), (16, 108))
        name_field.draw(screen, font)
        screen.blit(font.render("Duration (s):", True, COLOR_TEXT_DIM), (16, 152))
        duration_field.draw(screen, font)
        if state["message"]:
            msg_surf = font.render(state["message"], True, COLOR_TEXT_DIM)
            screen.blit(msg_surf, (16, 196))

        editor_panel.set_content_height(max(panel_content_height, editor_panel_rect.height))
        editor_panel.surface.fill(COLOR_SIDEBAR_BG)

        raw_mouse = pygame.mouse.get_pos()
        if editor_panel.rect.collidepoint(raw_mouse):
            panel_mouse = (raw_mouse[0] - editor_panel.rect.x, raw_mouse[1] - editor_panel.rect.y + editor_panel.scroll_y)
        else:
            panel_mouse = (-1, -1)

        def _draw_label(text, widget):
            editor_panel.surface.blit(font.render(text, True, COLOR_TEXT_DIM), (widget.rect.x, widget.rect.y - 16))

        def _draw_row(item, row_y, label, remove_btn):
            row_rect = pygame.Rect(12, row_y - 4, editor_panel.rect.width - 24, 28)
            pygame.draw.rect(editor_panel.surface, COLOR_WIDGET_BG_HOVER, row_rect, border_radius=3)
            max_label_width = remove_btn.rect.x - 22
            fitted = label
            while fitted and font.size(fitted + "...")[0] > max_label_width:
                fitted = fitted[:-1]
            if fitted != label:
                fitted += "..."
            editor_panel.surface.blit(font.render(fitted, True, COLOR_TEXT), (18, row_y + 2))
            remove_btn.draw(editor_panel.surface, font, panel_mouse)

        # Vehicles
        editor_panel.surface.blit(font_bold.render(f"Vehicles ({len(editable.vehicles)})", True, COLOR_TEXT), (16, 4))
        add_vehicle_button.draw(editor_panel.surface, font, panel_mouse)
        _draw_label("Name", vf_name)
        _draw_label("Speed (mph)", vf_speed)
        vf_name.draw(editor_panel.surface, font)
        vf_speed.draw(editor_panel.surface, font)
        _draw_label("Fusion policy", vf_policy)
        _draw_label("Lane", vf_lane)
        vf_policy.draw_closed(editor_panel.surface, font)
        vf_lane.draw_closed(editor_panel.surface, font)
        vf_position.draw(editor_panel.surface, font)
        vf_position_text.draw(editor_panel.surface, font)
        for v, row_y in zip(editable.vehicles, vehicle_row_tops):
            label = f"{v.get('vehicle_id')}  {v.get('start_distance', 0):.0f}m  {v.get('start_speed_mph', 0):.0f}mph  lane{v.get('lane', 0)}  {v.get('fusion_policy')}"
            _draw_row(v, row_y, label, vehicle_remove_buttons[id(v)])

        # Attacks
        editor_panel.surface.blit(font_bold.render(f"Attacks ({len(editable.attacks)})", True, COLOR_TEXT), (16, attacks_header_y))
        add_attack_button.draw(editor_panel.surface, font, panel_mouse)
        _draw_label("Attack type", af_type)
        _draw_label("Target vehicle", af_target)
        af_type.draw_closed(editor_panel.surface, font)
        af_target.draw_closed(editor_panel.surface, font)
        _draw_label("Trigger before feature", af_trigger_feature)
        af_trigger_feature.draw_closed(editor_panel.surface, font)
        _draw_label("Distance (m)", af_trigger_distance)
        _draw_label("Duration (s)", af_duration)
        af_trigger_distance.draw(editor_panel.surface, font)
        af_duration.draw(editor_panel.surface, font)
        label1, label2 = _attack_extra_field_labels(af_type.value)
        if label1:
            _draw_label(label1, af_extra1)
            af_extra1.draw(editor_panel.surface, font)
        if label2:
            _draw_label(label2, af_extra2)
            af_extra2.draw(editor_panel.surface, font)
        for a, row_y in zip(editable.attacks, attack_row_tops):
            label = f"{a.get('type')} -> {a.get('target_vehicle')}  @{a.get('trigger_before_feature')}  {a.get('duration')}s"
            _draw_row(a, row_y, label, attack_remove_buttons[id(a)])

        # Hazards
        editor_panel.surface.blit(font_bold.render(f"Hazards ({len(editable.hazards)})", True, COLOR_TEXT), (16, hazards_header_y))
        add_hazard_button.draw(editor_panel.surface, font, panel_mouse)
        _draw_label("Hazard type", hf_type)
        _draw_label("Feature", hf_feature)
        hf_type.draw_closed(editor_panel.surface, font)
        hf_feature.draw_closed(editor_panel.surface, font)
        _draw_label("Start time (s)", hf_start_time)
        _draw_label("Duration (s)", hf_duration)
        hf_start_time.draw(editor_panel.surface, font)
        hf_duration.draw(editor_panel.surface, font)
        if arrival_hints:
            hint_y = arrival_hints_y
            for hint in arrival_hints:
                editor_panel.surface.blit(font.render(hint, True, COLOR_TEXT_DIM), (16, hint_y))
                hint_y += 16
        if hf_type.value == "obstacle_in_road":
            _draw_label("Lane", hf_lane)
            hf_lane.draw_closed(editor_panel.surface, font)
        elif hf_type.value == "pedestrian_crossing":
            _draw_label("Pedestrian type", hf_pedestrian_type)
            hf_pedestrian_type.draw_closed(editor_panel.surface, font)
        for h, row_y in zip(editable.hazards, hazard_row_tops):
            label = f"{h.get('type')} @{h.get('feature_id')}  t={h.get('start_time')}s  {h.get('duration')}s"
            _draw_row(h, row_y, label, hazard_remove_buttons[id(h)])

        # Track shape
        editor_panel.surface.blit(font_bold.render("Track shape", True, COLOR_TEXT), (16, track_header_y))
        _draw_label("Straight (m)", tf_straight_length)
        _draw_label("Radius (m)", tf_radius)
        tf_straight_length.draw(editor_panel.surface, font)
        tf_radius.draw(editor_panel.surface, font)
        add_feature_button.draw(editor_panel.surface, font, panel_mouse)
        _draw_label("Feature ID", ff_id)
        ff_id.draw(editor_panel.surface, font)
        _draw_label("Feature type", ff_type)
        ff_type.draw_closed(editor_panel.surface, font)
        ff_position.draw(editor_panel.surface, font)
        ff_position_text.draw(editor_panel.surface, font)
        if ff_type.value == "roundabout":
            _draw_label("Roundabout radius (m)", ff_radius)
            ff_radius.draw(editor_panel.surface, font)
        for f, row_y in zip(editable.track_features, feature_row_tops):
            label = f"{f.get('feature_id')}  {f.get('feature_type')}  @{f.get('position', 0):.0f}m"
            _draw_row(f, row_y, label, feature_remove_buttons[id(f)])

        # Spawner
        editor_panel.surface.blit(font_bold.render("Spawner", True, COLOR_TEXT), (16, spawner_header_y))
        toggle_spawner_button.draw(editor_panel.surface, font, panel_mouse)
        if editable.spawner:
            _draw_label("Spawns from feature", sf_feature)
            sf_feature.draw_closed(editor_panel.surface, font)
            _draw_label("Max concurrent", sf_max_concurrent)
            _draw_label("Speed (mph)", sf_speed)
            sf_max_concurrent.draw(editor_panel.surface, font)
            sf_speed.draw(editor_panel.surface, font)
            _draw_label("Fusion policy", sf_policy)
            sf_policy.draw_closed(editor_panel.surface, font)

        for dd in (vf_policy, vf_lane, af_type, af_target, af_trigger_feature, hf_type, hf_feature, hf_lane, hf_pedestrian_type, ff_type, sf_feature, sf_policy):
            dd.draw_open_overlay(editor_panel.surface, font, panel_mouse)

        editor_panel.blit_to(screen)

        # middle: live preview track
        old_to_screen = preview_camera.to_screen
        preview_camera.to_screen = lambda x, y, _f=old_to_screen, _o=preview_camera_offset_x: (lambda sx, sy: (sx + _o, sy))(*_f(x, y))
        geometry = build_static_geometry(preview_track)
        draw_track(screen, preview_camera, geometry)
        feature_form_touched = bool(ff_id.value.strip()) or ff_type.value != FEATURE_TYPES[0] or ff_position.value != 0.0 or ff_radius.value != "25"
        in_progress_feature = {"feature_id": ff_id.value, "feature_type": ff_type.value, "position": ff_position.value,
                                "radius": float(ff_radius.value) if ff_type.value == "roundabout" and ff_radius.value.replace(".", "", 1).isdigit() else None}
        _draw_feature_preview_markers(screen, preview_camera, preview_track, font, in_progress=(in_progress_feature if feature_form_touched else None))
        vehicle_form_touched = bool(vf_name.value.strip()) or vf_speed.value != "30" or vf_policy.value != FUSION_POLICIES[0] or vf_lane.value != "0" or vf_position.value != 0.0
        in_progress_vehicle = {"vehicle_id": vf_name.value, "start_distance": vf_position.value, "lane": int(vf_lane.value)}
        _draw_vehicle_preview_markers(screen, preview_camera, preview_track, editable.vehicles, font, in_progress=(in_progress_vehicle if vehicle_form_touched else None))
        title_surf = font_bold.render(editable.name, True, COLOR_TEXT)
        screen.blit(title_surf, (LEFT_WIDTH + 16, 16))

        # right panel: simple info
        pygame.draw.rect(screen, COLOR_SIDEBAR_BG, (WINDOW_WIDTH - RIGHT_WIDTH, 0, RIGHT_WIDTH, WINDOW_HEIGHT))
        rx = WINDOW_WIDTH - RIGHT_WIDTH + 16
        screen.blit(font_bold.render("Editing", True, COLOR_TEXT), (rx, 16))
        screen.blit(font.render(f"{len(editable.vehicles)} vehicle(s)", True, COLOR_TEXT_DIM), (rx, 46))
        screen.blit(font.render(f"{len(editable.attacks)} attack(s)", True, COLOR_TEXT_DIM), (rx, 64))
        screen.blit(font.render(f"{len(editable.hazards)} hazard(s)", True, COLOR_TEXT_DIM), (rx, 82))
        screen.blit(font.render(f"duration {editable.duration:.0f}s", True, COLOR_TEXT_DIM), (rx, 100))
        y = 140
        for line in ["ESC  quit", "Load/New/Save above", "Scroll the left panel", "for vehicles"]:
            screen.blit(font.render(line, True, COLOR_TEXT_DIM), (rx, y))
            y += 18

        scenario_dropdown.draw_open_overlay(screen, font)

        pygame.display.flip()


def run(scenario_path: str | None = None) -> None:
    """Entry point. With a scenario_path given, loads straight into the
    editor with that scenario open

    With no path, opens on an empty new scenario"""
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas,couriernew,monospace", 15)
    font_bold = pygame.font.SysFont("consolas,couriernew,monospace", 16, bold=True)

    try:
        run_editor(screen, clock, font, font_bold, scenario_path)
    except SystemExit:
        pass

    pygame.quit()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run(path)
