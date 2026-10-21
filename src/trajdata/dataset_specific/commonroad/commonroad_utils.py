import glob
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import List, cast, Any, Final, Tuple

import numpy as np
import pandas as pd
import shapely
import yaml
from scipy.signal import savgol_filter
from scipy.interpolate import CubicHermiteSpline
from commonroad.common.file_reader  import CommonRoadFileReader
try:
    from commonroad.scenario.scenario import Scenario
    from commonroad.scenario.obstacle import Obstacle, StaticObstacle, DynamicObstacle, EnvironmentObstacle, PhantomObstacle, ObstacleType, Prediction, TrajectoryPrediction, SetBasedPrediction
    from commonroad.geometry.shape import Shape, ShapeGroup, Rectangle, Circle, Polygon
    from commonroad.scenario.state import State, KSState, InitialState, PMState, KSTState, STState, STDState, MBState, LongitudinalState, LateralState, InputState, PMInputState, CustomState, ExtendedPMState
    from commonroad.scenario.lanelet import Lanelet, LaneletNetwork, LaneletType
    from commonroad.common.file_reader  import CommonRoadFileReader
    from commonroad.planning.planning_problem import PlanningProblem
    from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry
    from commonroad.scenario.traffic_sign_interpreter import TrafficSignInterpreter
except Exception:
    CommonRoadFileReader = None  # type: ignore

from trajdata.data_structures.agent import AgentType
from trajdata.caching.df_cache import DataFrameCache
from trajdata.data_structures.scene_metadata import Scene
from trajdata.maps import TrafficLightStatus, VectorMap
from trajdata.maps.vec_map_elements import (
    MapElementType,
    PedCrosswalk,
    PedWalkway,
    Polyline,
    RoadArea,
    RoadLane,
    # RoadLaneWithSpeedLimit
)
from trajdata.utils import map_utils

COMMONROAD_DT: Final[float] = 0.1

LANELET_SMOOTHING_SPACING_M: Final[float] = 0.5
LANELET_SMOOTHING_WINDOW_M: Final[float] = 4.0
LANELET_MAX_GEOMETRY_CORRECTION_M: Final[float] = 1.0


def _smooth_lanelet_geometry(lanelet_network: LaneletNetwork) -> dict:
    """Smooth road geometry jointly, without modifying the CommonRoad network.

    Close adjacent boundaries are shared (reversed for opposing traffic).
    Separated stretches stay independent and blend into nearby shared stretches.
    Successors share endpoint positions and forward tangents. Only metre-scale
    discrepancies are reconciled; malformed topology/geometry raises an error.
    """
    lanes = {
        lane.lanelet_id: lane for lane in lanelet_network.lanelets
        if translate_lanelet_type(lane.lanelet_type) == MapElementType.ROAD_LANE
    }
    geometry_names = ("center", "left", "right")
    originals = {}
    for lane_id, lane in lanes.items():
        for name in geometry_names:
            xy = np.asarray(getattr(lane, name + "_vertices"), dtype=np.float64)
            if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
                raise ValueError(f"Lanelet {lane_id} has invalid {name} geometry")
            xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-8]]
            if len(xy) < 2:
                raise ValueError(f"Lanelet {lane_id} has degenerate {name} geometry")
            originals[lane_id, name] = xy.copy()

    def relate(graph, first, second, direction):
        if second not in graph:
            raise ValueError(f"Lane geometry references missing road curve {second}")
        graph[first].append((second, direction))
        graph[second].append((first, direction))

    def components(graph):
        membership, groups = {}, {}
        for root in sorted(graph):
            if root in membership:
                continue
            membership[root] = (root, 1)
            groups[root] = []
            pending = [root]
            while pending:
                node = pending.pop()
                _, direction = membership[node]
                groups[root].append(node)
                for other, relative in graph[node]:
                    expected = (root, direction * relative)
                    if other in membership:
                        if membership[other] != expected:
                            raise ValueError(f"Inconsistent lane-geometry orientation at {other}")
                    else:
                        membership[other] = expected
                        pending.append(other)
        return membership, groups

    # Use CommonRoad's direction flags, including opposite-direction adjacency.
    boundary_graph = {key: [] for key in originals}
    for lane_id, lane in lanes.items():
        for side in ("left", "right"):
            adjacent = getattr(lane, "adj_" + side)
            if adjacent is None:
                continue
            same_direction = getattr(lane, "adj_" + side + "_same_direction")
            if not isinstance(same_direction, (bool, np.bool_)):
                raise ValueError(f"Lanelet {lane_id} has no traffic direction for adjacent lanelet {adjacent}")
            other_side = ("right" if side == "left" else "left") if same_direction else side
            relate(boundary_graph, (lane_id, side), (adjacent, other_side),
                   1 if same_direction else -1)
    boundary_members, boundary_groups = components(boundary_graph)

    curves, stations = {}, {}
    partial_boundary_groups = []
    for root, members in boundary_groups.items():
        oriented = [originals[key][::boundary_members[key][1]] for key in members]
        arc_lengths = [np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
                       for xy in oriented]
        count = max(5, int(np.ceil(max(s[-1] for s in arc_lengths) / LANELET_SMOOTHING_SPACING_M)) + 1)
        progress = np.linspace(0., 1., count)
        samples = np.stack([
            np.column_stack([np.interp(progress * s[-1], s, xy[:, axis]) for axis in range(2)])
            for xy, s in zip(oriented, arc_lengths)
        ])
        curve = samples.mean(axis=0)
        corrections = np.linalg.norm(samples - curve, axis=-1).max(axis=0)
        if np.max(corrections) > LANELET_MAX_GEOMETRY_CORRECTION_M:
            # Adjacency can describe only a partial shared boundary, e.g. where
            # lanes diverge. Keep separate curves rather than moving the far
            # stretches toward each other. Reconcile nearby samples after smoothing.
            partial_boundary_groups.append((members, corrections))
            for key, sample in zip(members, samples):
                direction = boundary_members[key][1]
                boundary_members[key] = (key, direction)
                s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(sample, axis=0), axis=1))]
                if np.any(np.diff(s) <= 1e-8):
                    raise ValueError(f"Boundary {key} collapses or reverses")
                curves[key], stations[key] = sample, s
            continue
        s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        if np.any(np.diff(s) <= 1e-8):
            raise ValueError(f"Shared boundary {root} collapses or reverses")
        curves[root], stations[root] = curve, s

    # Endpoints belong to canonical curves, so boundary sharing and successor
    # continuity are solved together rather than overwritten in separate passes.
    endpoint_graph = {(root, end): [] for root in curves for end in (0, 1)}
    for members, corrections in partial_boundary_groups:
        for end, index in ((0, 0), (1, -1)):
            if 2 * corrections[index] <= LANELET_MAX_GEOMETRY_CORRECTION_M:
                for key in members[1:]:
                    # Each separate curve is already oriented with the group.
                    relate(endpoint_graph, (members[0], end), (key, end), 1)
    for lane_id, lane in lanes.items():
        for successor in lane.successor:
            for name in geometry_names:
                first, first_direction = boundary_members[lane_id, name]
                if (successor, name) not in boundary_members:
                    raise ValueError(f"Lanelet {lane_id} references missing road successor {successor}")
                second, second_direction = boundary_members[successor, name]
                first_end = 1 if first_direction == 1 else 0
                second_start = 0 if second_direction == 1 else 1
                relate(endpoint_graph, (first, first_end), (second, second_start),
                       first_direction * second_direction)
    endpoint_members, endpoint_groups = components(endpoint_graph)
    positions, tangents = {}, {}
    for root, members in endpoint_groups.items():
        points, directions = [], []
        for key in members:
            curve, s = curves[key[0]], stations[key[0]]
            end = key[1]
            index = -1 if end else 0
            span = min(LANELET_SMOOTHING_WINDOW_M / 2, s[-1])
            probe = s[-1] - span if end else span
            nearby = np.array([np.interp(probe, s, curve[:, axis]) for axis in range(2)])
            tangent = curve[-1] - nearby if end else nearby - curve[0]
            norm = np.linalg.norm(tangent)
            if norm <= 1e-8:
                raise ValueError(f"Lane endpoint {key} has a degenerate tangent")
            points.append(curve[index])
            directions.append(endpoint_members[key][1] * tangent / norm)
        points = np.stack(points)
        if np.max(np.linalg.norm(points[:, None] - points[None], axis=-1)) > LANELET_MAX_GEOMETRY_CORRECTION_M:
            raise ValueError(f"Successor endpoints at {root} differ by more than 1 m")
        tangent = np.sum(directions, axis=0)
        norm = np.linalg.norm(tangent)
        if norm <= 1e-8:
            raise ValueError(f"Linked lane endpoint tangents cancel at {root}")
        positions[root], tangents[root] = points.mean(axis=0), tangent / norm

    # Keep center endpoints inside the reconciled cross-sections. Shared
    # boundaries can move longitudinally as well as laterally at a junction.
    for root, members in endpoint_groups.items():
        if root[0][1] != "center":
            continue
        midpoints = []
        for (lane_id, _), end in members:
            edges = []
            for side in ("left", "right"):
                boundary, direction = boundary_members[lane_id, side]
                boundary_end = end if direction == 1 else 1 - end
                endpoint_root, _ = endpoint_members[boundary, boundary_end]
                edges.append(positions[endpoint_root])
            midpoints.append(np.mean(edges, axis=0))
        positions[root] = np.mean(midpoints, axis=0)

    smoothed, smoothing_originals = {}, {}
    for root, curve in curves.items():
        s = stations[root]
        spacing = s[-1] / (len(s) - 1)
        window = int(round(LANELET_SMOOTHING_WINDOW_M / spacing)) | 1
        window = min(window, len(s) if len(s) % 2 else len(s) - 1)
        # Normalize arc-length stations before smoothing; irregular source
        # vertices must not alter the physical size of the smoothing window.
        sample_s = np.linspace(0., s[-1], len(s))
        xy = np.column_stack([np.interp(sample_s, s, curve[:, axis]) for axis in range(2)])
        if window >= 5:
            xy = savgol_filter(xy, window, 3, axis=0, mode="interp")
        span = min(LANELET_SMOOTHING_WINDOW_M, s[-1] / 3)
        desired = []
        for end, index in ((0, 0), (1, -1)):
            endpoint_root, direction = endpoint_members[root, end]
            distance = sample_s if end == 0 else sample_s[-1] - sample_s
            u = np.clip(distance / span, 0., 1.)
            weight = 1 - 10 * u**3 + 15 * u**4 - 6 * u**5
            xy += weight[:, None] * (positions[endpoint_root] - xy[index])
            desired.append(direction * tangents[endpoint_root])
        derivative = np.gradient(xy, sample_s, axis=0, edge_order=2)
        # Spread the tangent correction over metres of road rather than
        # creating another kink in the first/last sampled segment.
        for end, index in ((0, 0), (1, -1)):
            distance = sample_s if end == 0 else sample_s[-1] - sample_s
            u = np.clip(distance / span, 0., 1.)
            weight = distance * (1 - u)**3
            xy += (1 if end == 0 else -1) * weight[:, None] * (desired[end] - derivative[index])
        derivative = np.gradient(xy, sample_s, axis=0, edge_order=2)
        for end, index in ((0, 0), (1, -1)):
            endpoint_root, direction = endpoint_members[root, end]
            xy[index] = positions[endpoint_root]
            derivative[index] = direction * tangents[endpoint_root]
        spline = CubicHermiteSpline(sample_s, xy, derivative)
        target_s = np.linspace(0., s[-1], 2 * len(s) - 1)
        sampled = spline(target_s)
        sampled[0], sampled[-1] = xy[0], xy[-1]
        # Align discrete end chords too: Frenetix projects the sampled polyline,
        # rather than the underlying spline, and must see the common tangent.
        sampled[1] = sampled[0] + np.linalg.norm(sampled[1] - sampled[0]) * derivative[0]
        sampled[-2] = sampled[-1] - np.linalg.norm(sampled[-1] - sampled[-2]) * derivative[-1]
        original = np.column_stack([np.interp(target_s, s, curve[:, axis]) for axis in range(2)])
        if np.max(np.linalg.norm(sampled - original, axis=-1)) > LANELET_MAX_GEOMETRY_CORRECTION_M:
            raise ValueError(f"Smoothing lane curve {root} would move it by more than 1 m")
        if np.any(np.linalg.norm(np.diff(sampled, axis=0), axis=1) <= 1e-8):
            raise ValueError(f"Smoothing lane curve {root} creates duplicate vertices")
        smoothed[root] = sampled
        smoothing_originals[root] = original

    for members, _ in partial_boundary_groups:
        samples = [smoothed[key].copy() for key in members]
        lines = [shapely.LineString(sample) for sample in samples]
        for index, (key, sample) in enumerate(zip(members, samples)):
            tangent = np.gradient(sample, axis=0)
            norms = np.linalg.norm(tangent, axis=-1)
            if np.any(norms <= 1e-8):
                raise ValueError(f"Boundary {key} has a degenerate tangent")
            tangent /= norms[:, None]
            movement = np.zeros_like(sample)
            total_weight = np.ones(len(sample))
            for other_index, line in enumerate(lines):
                if other_index == index:
                    continue
                # Match by physical proximity, not percentage of curve length:
                # adjacent boundaries may cover different longitudinal extents.
                projected = shapely.get_coordinates(shapely.line_interpolate_point(
                    line, shapely.line_locate_point(line, shapely.points(sample)),
                ))
                delta = projected - sample
                correction = np.linalg.norm(delta, axis=-1) / 2
                transition = np.clip(2 * correction / LANELET_MAX_GEOMETRY_CORRECTION_M - 1, 0., 1.)
                weight = 1 - transition**2 * (3 - 2 * transition)
                # Keep the lane's longitudinal extent; do not pull short curves
                # toward another curve's endpoint or fold a curve back on itself.
                lateral = delta - np.sum(delta * tangent, axis=-1)[:, None] * tangent
                movement += weight[:, None] * lateral
                total_weight += weight
            reconciled = sample + movement / total_weight[:, None]
            # Preserve successor positions and their common discrete tangents.
            for end, neighbor in ((0, 1), (-1, -2)):
                reconciled[end] = sample[end]
                direction = sample[neighbor] - sample[end]
                direction /= np.linalg.norm(direction)
                reconciled[neighbor] = sample[end] + np.linalg.norm(
                    reconciled[neighbor] - sample[end]
                ) * direction
            if np.max(np.linalg.norm(reconciled - smoothing_originals[key], axis=-1)) > LANELET_MAX_GEOMETRY_CORRECTION_M:
                raise ValueError(f"Reconciling nearby boundary stretches at {key} requires more than 1 m correction")
            if np.any(np.linalg.norm(np.diff(reconciled, axis=0), axis=1) <= 1e-8):
                raise ValueError(f"Reconciling nearby boundary stretches at {key} creates duplicate vertices")
            smoothed[key] = reconciled

    result = {}
    for lane_id in lanes:
        result[lane_id] = {
            name: Polyline(smoothed[boundary_members[lane_id, name][0]][::boundary_members[lane_id, name][1]].copy())
            for name in geometry_names
        }
        geometry = result[lane_id]
        polygon = shapely.Polygon(np.vstack((geometry["left"].xy, geometry["right"].xy[::-1])))
        if not polygon.is_valid or polygon.area <= 0:
            raise ValueError(f"Smoothing lanelet {lane_id} creates an invalid lane polygon")
        if not polygon.buffer(1e-3).covers(shapely.LineString(geometry["center"].xy)):
            raise ValueError(f"Smoothed lanelet {lane_id} center leaves its boundaries")
    return result
# # GOAL_METADATA_FILENAME: Final[str] = "commonroad_goal.json"
# GOAL_METADATA_VERSION: Final[int] = 1


def _goal_time_interval(value: Any) -> list[int]:
    if hasattr(value, "start") and hasattr(value, "end"):
        start, end = float(value.start), float(value.end)
    elif isinstance(value, (int, np.integer)):
        start = end = float(value)
    else:
        raise TypeError(f"Unsupported CommonRoad goal time type: {type(value).__name__}")
    if not np.isfinite(start) or not np.isfinite(end) or end < start:
        raise ValueError(f"Invalid CommonRoad goal time interval: {start}, {end}")
    return [int(start), int(end)]


def _add_goal_area(vector_map: VectorMap, agent_id: str, geometry: shapely.Polygon) -> None:
    """Store one goal as a native map polygon, preserving holes and map extent."""
    if not isinstance(geometry, shapely.Polygon) or geometry.is_empty or not geometry.is_valid:
        raise ValueError(f"Goal for {agent_id!r} must be a valid nonempty polygon")
    if agent_id in vector_map.elements[MapElementType.ROAD_AREA]:
        raise ValueError(f"Goal ID {agent_id!r} collides with an existing RoadArea")
    exterior_poly = np.asarray(geometry.exterior.coords)
    # heading = map_utils.get_polyline_headings(np.asarray(geometry.exterior.coords))  # validate and add headings
    area = RoadArea(
        id=agent_id,
        exterior_polygon=Polyline(np.concatenate((exterior_poly, np.zeros((exterior_poly.shape[0], 1))), axis=-1)),
        interior_holes=[Polyline(np.asarray(ring.coords)) for ring in geometry.interiors],
    )
    vector_map.add_map_element(area)
    vector_map.extent[:3] = np.minimum(vector_map.extent[:3], area.exterior_polygon.xyz.min(axis=0))
    vector_map.extent[3:] = np.maximum(vector_map.extent[3:], area.exterior_polygon.xyz.max(axis=0))



def serialize_goal(scenario: Any, planning_problem: Any, vector_map: VectorMap) -> dict[str, Any]:
    """Store the first ego goal polygon/lane, or only time for a position-free goal."""
    # goals = []
    goal_state = planning_problem.goal.state_list[0]
    attributes = set(goal_state.attributes)
    has_position = False
    if "position" in attributes and goal_state.position is not None:
        shape = (
            goal_state.position.shapes[0]
            if isinstance(goal_state.position, ShapeGroup)
            else goal_state.position
        )
        # for shape in shapes[:1]:
        if isinstance(shape, np.ndarray):
            xy = np.asarray(shape, dtype=float).reshape(-1)
            if xy.size != 2 or not np.all(np.isfinite(xy)):
                raise ValueError("CommonRoad point goal must be finite x,y")
            geometry = shapely.Point(*xy)
        else:
            geometry = shape.shapely_object
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError("CommonRoad goal geometry is invalid")
        # agent_id = "ego"
        _add_goal_area(vector_map, "ego", geometry)
            
        if hasattr(planning_problem.goal, "lanelets_of_goal_position") and planning_problem.goal.lanelets_of_goal_position is not None:
            goal_lanelets = planning_problem.goal.lanelets_of_goal_position[0]
        else:
            goal_lanelets = scenario.lanelet_network.find_lanelet_by_position([shape.center])[0]
            # if type(goal_state.position) is not ShapeGroup:
                # goal_lanelets = scenario.lanelet_network.find_lanelet_by_position([goal_state.position.center])[0]
            # else:
                # goal_lanelets = scenario.lanelet_network.find_lanelet_by_position([goal_state.position.shapes[0].center])[0]
        # goal_lanelets[0]
        has_position = True
    interval = (
        _goal_time_interval(goal_state.time_step)
        if "time_step" in attributes and goal_state.time_step is not None
        else None
    )
    if not has_position:
        if interval is None:
            raise ValueError("CommonRoad goal state has neither position nor time_step")
        return {"time_interval": interval, "lanelet_id": None}
    if not goal_lanelets:
        raise ValueError(f"CommonRoad goal in scenario {scenario.scenario_id} has no matching lanelet")
    return {"time_interval": interval, "lanelet_id": int(goal_lanelets[0])}


# def write_goal_metadata(
#     cache_path: Path, env_name: str, scene_name: str, planning_problem: Any, source_dt_s: float
# ) -> dict[str, Any]:
#     metadata = serialize_goal(planning_problem, source_dt_s)
#     destination = DataFrameCache.scene_cache_dir(cache_path, env_name, scene_name) / GOAL_METADATA_FILENAME
#     destination.parent.mkdir(parents=True, exist_ok=True)
#     destination.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
#     return metadata


# def load_goal_metadata(scene_cache_dir: Path) -> dict[str, Any]:
#     source = Path(scene_cache_dir) / GOAL_METADATA_FILENAME
#     if not source.is_file():
#         raise FileNotFoundError(
#             f"CommonRoad goal metadata is missing at {source}; rebuild the scene cache."
#         )
#     metadata = json.loads(source.read_text())
#     if metadata.get("version") != GOAL_METADATA_VERSION:
#         raise RuntimeError(
#             f"Unsupported CommonRoad goal metadata version: {metadata.get('version')!r}"
#         )
#     return metadata

def _lanelet_goal_polygon(lanelet: Lanelet, position: np.ndarray) -> shapely.Polygon:
    """Follow both boundaries for five metres either side of the final station."""
    center = np.asarray(lanelet.center_vertices, dtype=float)
    left = np.asarray(lanelet.left_vertices, dtype=float)
    right = np.asarray(lanelet.right_vertices, dtype=float)
    distances = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(center, axis=0), axis=1))))
    if len(center) < 2 or np.any(np.diff(distances) <= 0.0):
        raise ValueError(f"Lanelet {lanelet.lanelet_id} has degenerate center geometry")
    if left.shape != center.shape or right.shape != center.shape:
        raise ValueError(f"Lanelet {lanelet.lanelet_id} has unaligned boundary vertices")
    station = shapely.LineString(center).project(shapely.Point(position))
    start, end = station - 5.0, station + 5.0
    stations = np.concatenate(([start], distances[(distances > start) & (distances < end)], [end]))

    def sample_boundary(vertices: np.ndarray) -> np.ndarray:
        points = np.column_stack([
            np.interp(stations, distances, vertices[:, axis]) for axis in range(2)
        ])
        for mask, endpoint, tangent, offset in (
            (stations < 0.0, vertices[0], vertices[1] - vertices[0], stations),
            (stations > distances[-1], vertices[-1], vertices[-1] - vertices[-2], stations - distances[-1]),
        ):
            if np.any(mask):
                norm = np.linalg.norm(tangent)
                if norm == 0.0:
                    raise ValueError(f"Lanelet {lanelet.lanelet_id} has a degenerate boundary tangent")
                points[mask] = endpoint + offset[mask, None] * tangent / norm
        return points

    polygon = shapely.Polygon(np.concatenate((sample_boundary(left), sample_boundary(right)[::-1])))
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0.0:
        raise ValueError(f"Lanelet {lanelet.lanelet_id} produces an invalid obstacle goal polygon")
    return polygon



def serialize_obstacle_goals(scenario: Scenario, obstacle: Obstacle, vector_map: VectorMap) -> dict[str, Any]:
    """Store a lane-width goal only when the final state lies inside the network."""
    if isinstance(obstacle, DynamicObstacle):
        if obstacle.prediction is None:
            final_state = obstacle.initial_state
        elif isinstance(obstacle.prediction, TrajectoryPrediction):
            final_state = obstacle.prediction.trajectory.final_state
        else:
            raise TypeError(f"Obstacle {obstacle.obstacle_id} has no final trajectory state")
    elif isinstance(obstacle, StaticObstacle):
        final_state = obstacle.initial_state
    else:
        raise TypeError(f"Obstacle {obstacle.obstacle_id} has no supported position state")

    # CommonRoad indexes the first candidate without handling an empty match.
    candidate_lanelets = scenario.lanelet_network.find_lanelet_by_position([final_state.position])[0]
    if not candidate_lanelets:
        return {}
    lanelet_id = scenario.lanelet_network.find_most_likely_lanelet_by_state([final_state])
    lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id[0])
    _add_goal_area(vector_map, str(obstacle.obstacle_id), _lanelet_goal_polygon(lanelet, final_state.position))
    return {"time_interval": [int(final_state.time_step), int(final_state.time_step)],  "lanelet_id": int(lanelet_id[0])}


def serialize_lane_metadata(scenario: Scenario) -> pd.DataFrame:
    """Serialize CommonRoad lanelet attributes omitted by the VectorMap proto."""
    country = SupportedTrafficSignCountry(scenario.scenario_id.country_id)
    interpreter = TrafficSignInterpreter(
        country=country,
        lanelet_network=scenario.lanelet_network,
    )
    metadata: dict[str, dict[str, Any]] = {}
    for lanelet in scenario.lanelet_network.lanelets:
        speed_limit = interpreter.speed_limit(frozenset({lanelet.lanelet_id}))
        metadata[str(lanelet.lanelet_id)] = {
            "speed_limit_mps": None if speed_limit is None else float(speed_limit),
            # "lanelet_types": sorted(value.value for value in lanelet.lanelet_type),
            # "user_one_way": sorted(value.value for value in lanelet.user_one_way),
            # "user_bidirectional": sorted(value.value for value in lanelet.user_bidirectional),
            # "line_marking_left": lanelet.line_marking_left_vertices.value,
            # "line_marking_right": lanelet.line_marking_right_vertices.value,
            # "adj_left_same_direction": lanelet.adj_left_same_direction,
            # "adj_right_same_direction": lanelet.adj_right_same_direction,
            # "traffic_sign_ids": sorted(int(value) for value in lanelet.traffic_signs),
            # "traffic_light_ids": sorted(int(value) for value in lanelet.traffic_lights),
            # "stop_line": stop_line,
        }
    metadata_df = pd.DataFrame.from_dict(
        metadata, orient="index", columns=["speed_limit_mps"],
    ).rename_axis("lane_id").astype({"speed_limit_mps": "float64"})
    return metadata_df

def cache_map_metadata(
    scenario: Scenario, planning_problem: PlanningProblem, vector_map: VectorMap
) -> pd.DataFrame:
    """Build lane attributes and map-level goal data omitted by the VectorMap proto.

    Map-level fields live in ``metadata.attrs["map_metadata"]``, matching the
    dataframe returned by ``DataFrameCache.load_map_metadata()``. Obstacles whose
    final state has no lanelet are omitted from goal metadata and goal polygons.
    """
    # country = SupportedTrafficSignCountry(scenario.scenario_id.country_id)
    # interpreter = TrafficSignInterpreter(
    #     country=country,
    #     lanelet_network=scenario.lanelet_network,
    # )
    # metadata: dict[str, dict[str, Any]] = {}
    # for lanelet in scenario.lanelet_network.lanelets:
    #     speed_limit = interpreter.speed_limit(frozenset({lanelet.lanelet_id}))
    #     # stop_line = None
    #     # if lanelet.stop_line is not None:
    #     #     stop_line = {
    #     #         "start": np.asarray(lanelet.stop_line.start, dtype=float).tolist(),
    #     #         "end": np.asarray(lanelet.stop_line.end, dtype=float).tolist(),
    #     #         "line_marking": lanelet.stop_line.line_marking.value,
    #     #         "traffic_sign_ref": sorted(
    #     #             int(value) for value in (lanelet.stop_line.traffic_sign_ref or set())
    #     #         ),
    #     #         "traffic_light_ref": sorted(
    #     #             int(value) for value in (lanelet.stop_line.traffic_light_ref or set())
    #     #         ),
    #     #     }
    #     metadata[str(lanelet.lanelet_id)] = {
    #         # "source": "commonroad",
    #         "speed_limit_mps": None if speed_limit is None else float(speed_limit),
    #         # "lanelet_types": sorted(value.value for value in lanelet.lanelet_type),
    #         # "user_one_way": sorted(value.value for value in lanelet.user_one_way),
    #         # "user_bidirectional": sorted(value.value for value in lanelet.user_bidirectional),
    #         # "line_marking_left": lanelet.line_marking_left_vertices.value,
    #         # "line_marking_right": lanelet.line_marking_right_vertices.value,
    #         # "adj_left_same_direction": lanelet.adj_left_same_direction,
    #         # "adj_right_same_direction": lanelet.adj_right_same_direction,
    #         # "traffic_sign_ids": sorted(int(value) for value in lanelet.traffic_signs),
    #         # "traffic_light_ids": sorted(int(value) for value in lanelet.traffic_lights),
    #         # "stop_line": stop_line,
    #     }
    metadata_df = serialize_lane_metadata(scenario)
    goals = {"ego": serialize_goal(scenario, planning_problem, vector_map)}
    for obstacle in scenario.obstacles:
        goal = serialize_obstacle_goals(scenario, obstacle, vector_map)
        if goal:
            goals[str(obstacle.obstacle_id)] = goal
    metadata_df.attrs["map_metadata"] = {"goals": goals}
    return metadata_df


class CommonRoadScenarioHandler:
    def __init__(self, data_dir: Path,) -> None:

        self.data_dir = Path(data_dir)
        # A stable order keeps raw_data_idx reproducible when a scenario is
        # added to the directory between two cache refreshes.
        self.scenario_files = sorted(self.data_dir.glob("*.xml"))
        # Validate dt when a selected scenario is loaded, not during discovery.

        self.num_scenarios = len(self.scenario_files)

        if self.num_scenarios==0:
            raise ValueError(f"No .xml files found in {data_dir}")
        
    def get_scenario_path(self, idx : int) -> Path:
        return self.scenario_files[idx]             #Thus it is INTENDED (keep in mind) that the data_idx of a scene = its idx in the list of dataset_obj.scenario_files
    
    def get_scenario_name(self, idx) -> str:
        return self.scenario_files[idx].stem

    def get_scenario_length(self, idx: int) -> int:         #Commonroad has no fixed length of scenario, so we will take max time_step of prediction to be = scenario length
        """Calculate number of timesteps in scenario"""
        try:
            scenario, pps = self.load_scenario(idx)
            planning_problem = list(pps.planning_problem_dict.values())[0]

            max_timestep = int(planning_problem.goal.state_list[0].time_step.end)
            return max_timestep
            # import commonroad_velocity_planner.fast_api as cvp_fast_api

            # global_trajectory = cvp_fast_api.global_trajectory_from_scenario_and_planning_problem(
            #     scenario=scenario,
            #     planning_problem=planning_problem,
            #     use_regulatory_elements=False,
            # )
            # start_idx = global_trajectory.get_closest_idx(np.array(planning_problem.initial_state.position))
            # velocities = np.asarray(global_trajectory.velocity_profile[start_idx:], dtype=np.float64)
            # interpoint_distance = np.asarray(global_trajectory.interpoint_distance[start_idx:], dtype=np.float64)
            # total_duration_s = float(np.sum(interpoint_distance / np.maximum(velocities, 0.01)))
            # ego_route_steps = int(np.floor(total_duration_s / COMMONROAD_DT)) + 1

            # return max(max_timestep, ego_route_steps)

        except Exception as e:
            print(f"Error calculating length for scenario {idx}: {e}")
            return 1  # Return minimum length to avoid crashes
    
    def load_scenario(self, idx: int) -> Tuple[Scenario, PlanningProblem]:
        path = self.get_scenario_path(idx)
        scenario, planning_problem_set = CommonRoadFileReader(path).open()
        # assert scenario.dt == COMMONROAD_DT, f"Scenario {path} has dt={scenario.dt}, expected {COMMONROAD_DT}"
        return scenario, planning_problem_set

def translate_agent_type(agent_type : ObstacleType):           #Types here might need to be reviewed. Example, where to put ObstacleType.TRAIN? Also, ObstacleType.PARKED_VEHICLE i probably Static, but ok. Also, MOTORCYCLE goes into AgentType.BICYCLE or AgentType.VEHICLE?
    if agent_type in {ObstacleType.CAR, ObstacleType.PRIORITY_VEHICLE, ObstacleType.PARKED_VEHICLE, ObstacleType.TAXI} :
        return AgentType.VEHICLE
    elif agent_type == ObstacleType.TRUCK:
        return AgentType.TRUCK
    elif agent_type == ObstacleType.BUS:
        return AgentType.BUS
    elif agent_type == ObstacleType.PEDESTRIAN:
        return AgentType.PEDESTRIAN
    elif agent_type == ObstacleType.BICYCLE:
        return AgentType.BICYCLE
    elif agent_type == ObstacleType.MOTORCYCLE:
        return AgentType.MOTORCYCLE
    elif agent_type == ObstacleType.BUILDING:
            return AgentType.BUILDING
    elif agent_type == ObstacleType.UNKNOWN:
        return AgentType.UNKNOWN
    return AgentType.UNKNOWN


# def pad_and_interpolate_array(data: np.ndarray, initial_ts_cr: int, final_ts_cr: int, len_scene_ts: int) -> np.ndarray:    #NOTE : "cr" ==> CommonRoad indexing. that is 1,2,3,4,....
    
#     data = pd.DataFrame(data).interpolate(limit_area="inside").to_numpy()
#     data = np.pad(array=data,
#            pad_width=((initial_ts_cr-1, len_scene_ts-final_ts_cr), (0,0)),
#            mode='constant',
#            constant_values=-1e8,
#            )
#     return data

def extract_vectorized(
    lanelet_network: LaneletNetwork, map_name: str #, country:str,  verbose: bool = False
) -> VectorMap:
    """
    Extract a vectorized representation of the lanelet network.
    """
    smoothed_geometry = _smooth_lanelet_geometry(lanelet_network)

    vec_map = VectorMap(map_id=map_name)
    max_pt = np.array([np.nan, np.nan, np.nan])
    min_pt = np.array([np.nan, np.nan, np.nan])
    # speed_limit_interpreter = TrafficSignInterpreter(country, lanelet_network)
    for _, lanelet in enumerate(lanelet_network.lanelets) :
        
        elem_type = translate_lanelet_type(lanelet.lanelet_type)

        if elem_type==MapElementType.PED_CROSSWALK:
            polygon=lanelet_to_polygon(lanelet)
            if polygon.points.size==0:
                continue
            crosswalk=PedCrosswalk(
                id=str(lanelet.lanelet_id),
                polygon=polygon,
                )
            max_pt = np.fmax(max_pt, crosswalk.polygon.xyz.max(axis=0))
            min_pt = np.fmin(min_pt, crosswalk.polygon.xyz.min(axis=0))
            vec_map.add_map_element(crosswalk)

        elif elem_type==MapElementType.PED_WALKWAY:
            polygon=lanelet_to_polygon(lanelet)
            if polygon.points.size==0:
                continue
            walkway = PedWalkway(
                id=str(lanelet.lanelet_id),
                polygon=polygon,
                )
            max_pt = np.fmax(max_pt, walkway.polygon.xyz.max(axis=0))
            min_pt = np.fmin(min_pt, walkway.polygon.xyz.min(axis=0))
            vec_map.add_map_element(walkway)
        
        elif elem_type==MapElementType.ROAD_AREA:
            polygon=lanelet_to_polygon(lanelet)
            if polygon.points.size==0:
                continue
            road_area = RoadArea(
                id=str(lanelet.lanelet_id),
                exterior_polygon=polygon,
                )
            max_pt = np.fmax(max_pt, road_area.center.xyz.max(axis=0))
            min_pt = np.fmin(min_pt, road_area.center.xyz.min(axis=0))

            print(f"ID: {road_area.id}")
            print(f"exterior_polygon.points.shape: {road_area.exterior_polygon.points.shape}")
            
            vec_map.add_map_element(road_area)

        elif elem_type==MapElementType.ROAD_LANE:
            adj_lanes_left = {str(lanelet.adj_left)} if lanelet.adj_left is not None else set()
            adj_lanes_right = {str(lanelet.adj_right)} if lanelet.adj_right is not None else set()
            next_lanes = {str(x) for x in lanelet.successor if x is not None}
            prev_lanes = {str(x) for x in lanelet.predecessor if x is not None}
            road_lane= RoadLane( #WithSpeedLimit(
                id=str(lanelet.lanelet_id),
                
                center=smoothed_geometry[lanelet.lanelet_id]["center"],
                left_edge=smoothed_geometry[lanelet.lanelet_id]["left"],
                right_edge=smoothed_geometry[lanelet.lanelet_id]["right"],
                adj_lanes_left=adj_lanes_left,
                adj_lanes_right=adj_lanes_right,
                next_lanes=next_lanes,
                prev_lanes=prev_lanes,                
            )
            #Calulate max_pt and min pt too while we're at it.
            max_pt = np.fmax(max_pt, road_lane.center.xyz.max(axis=0))
            min_pt = np.fmin(min_pt, road_lane.center.xyz.min(axis=0))
            if road_lane.left_edge:
                max_pt = np.fmax(max_pt, road_lane.left_edge.xyz.max(axis=0))
                min_pt = np.fmin(min_pt, road_lane.left_edge.xyz.min(axis=0))
            if road_lane.right_edge:
                max_pt = np.fmax(max_pt, road_lane.right_edge.xyz.max(axis=0))
                min_pt = np.fmin(min_pt, road_lane.right_edge.xyz.min(axis=0))

            vec_map.add_map_element(road_lane)      #add the element to vec_map

    vec_map.extent = np.concatenate((min_pt, max_pt))
    if MapElementType.ROAD_LANE in vec_map.elements:
        vec_map.lanes = cast(List[RoadLane], list(vec_map.elements[MapElementType.ROAD_LANE].values()))
    return vec_map


def translate_lanelet_type(lanelet_type: set[LaneletType]) -> MapElementType:
    if LaneletType.CROSSWALK in lanelet_type :
        return MapElementType.PED_CROSSWALK
    elif LaneletType.SIDEWALK in lanelet_type :
        return MapElementType.PED_WALKWAY
    elif LaneletType.PARKING in lanelet_type :
        return MapElementType.ROAD_AREA
    elif LaneletType.BICYCLE_LANE in lanelet_type:
        return MapElementType.PED_WALKWAY
    else : 
        return MapElementType.ROAD_LANE

def lanelet_to_polygon(lanelet: Lanelet) -> Polyline :
    """
    Converts a Lanelet object into a closed polygon Polyline
    """
    polygon_points = lanelet.polygon.vertices
    return Polyline(polygon_points)

# def check_obstacle_validity(dynamic_obstacle : DynamicObstacle) -> bool:

#     if (translate_agent_type(dynamic_obstacle.obstacle_type) == AgentType.UNKNOWN):
#         print(f"{dynamic_obstacle.obstacle_id} is of unrecognized Agent Type {dynamic_obstacle.obstacle_type}", flush=True)
#         return False
    
#     elif not isinstance(dynamic_obstacle.prediction, TrajectoryPrediction):
#         print(f"{dynamic_obstacle.obstacle_id} is of unsupported Prediction {type(dynamic_obstacle.prediction)}")
#         return False
    
#     elif not isinstance(dynamic_obstacle.obstacle_shape, Rectangle):
#         print(f"{dynamic_obstacle.obstacle_id} is of unsupported Shape {type(dynamic_obstacle.obstacle_shape)}")
#         return False
#     else : 
#         return True

# def check_state_validity(state: State) -> bool:
#     validity=False
#     allowed_states = (PMState, KSState, ExtendedPMState, CustomState)     #NOTE : KST states are subclasses of KSState only
#     if not (state.is_uncertain_position or state.is_uncertain_orientation) :
#         if isinstance(state, allowed_states):
#             validity=True

#     return validity
