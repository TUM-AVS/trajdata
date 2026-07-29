import glob
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Final, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline, splprep, splev
from commonroad.common.file_reader  import CommonRoadFileReader
try:
    from commonroad.scenario.scenario import Scenario
    from commonroad.scenario.obstacle import Obstacle, StaticObstacle, DynamicObstacle, EnvironmentObstacle, PhantomObstacle, ObstacleType, Prediction, TrajectoryPrediction, SetBasedPrediction
    from commonroad.geometry.shape import Shape, Rectangle, Circle, Polygon
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
GOAL_METADATA_FILENAME: Final[str] = "commonroad_goal.json"
GOAL_METADATA_VERSION: Final[int] = 1


def _goal_time_interval(value: Any) -> dict[str, float]:
    if hasattr(value, "start") and hasattr(value, "end"):
        start, end = float(value.start), float(value.end)
    elif isinstance(value, (int, np.integer)):
        start = end = float(value)
    else:
        raise TypeError(f"Unsupported CommonRoad goal time type: {type(value).__name__}")
    if not np.isfinite(start) or not np.isfinite(end) or end < start:
        raise ValueError(f"Invalid CommonRoad goal time interval: {start}, {end}")
    return {"start": start, "end": end}


def serialize_goal(planning_problem: Any, source_dt_s: float) -> dict[str, Any]:
    """Serialize CommonRoad's exact position/time termination alternatives."""
    if source_dt_s <= 0.0 or not np.isfinite(source_dt_s):
        raise ValueError(f"CommonRoad source dt must be positive, got {source_dt_s}")
    conditions = []
    for goal_state in planning_problem.goal.state_list:
        attributes = set(goal_state.attributes)
        position_wkb = None
        if "position" in attributes and goal_state.position is not None:
            shapes = getattr(goal_state.position, "shapes", [goal_state.position])
            position_wkb = []
            for shape in shapes:
                if isinstance(shape, np.ndarray):
                    xy = np.asarray(shape, dtype=float).reshape(-1)
                    if xy.size != 2 or not np.all(np.isfinite(xy)):
                        raise ValueError("CommonRoad point goal must be finite x,y")
                    import shapely
                    geometry = shapely.Point(*xy)
                else:
                    geometry = shape.shapely_object
                if geometry.is_empty or not geometry.is_valid:
                    raise ValueError("CommonRoad goal geometry is invalid")
                import shapely
                position_wkb.append(shapely.to_wkb(geometry, hex=True))
        interval = (
            _goal_time_interval(goal_state.time_step)
            if "time_step" in attributes and goal_state.time_step is not None
            else None
        )
        if position_wkb is None and interval is None:
            raise ValueError("CommonRoad goal state has neither position nor time_step")
        conditions.append({"position_wkb": position_wkb, "time_interval_steps": interval})
    if not conditions:
        raise ValueError("CommonRoad planning problem has no goal states")
    return {"version": GOAL_METADATA_VERSION, "source_dt_s": float(source_dt_s), "conditions": conditions}


def write_goal_metadata(
    cache_path: Path, env_name: str, scene_name: str, planning_problem: Any, source_dt_s: float
) -> dict[str, Any]:
    metadata = serialize_goal(planning_problem, source_dt_s)
    destination = DataFrameCache.scene_cache_dir(cache_path, env_name, scene_name) / GOAL_METADATA_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    return metadata


def load_goal_metadata(scene_cache_dir: Path) -> dict[str, Any]:
    source = Path(scene_cache_dir) / GOAL_METADATA_FILENAME
    if not source.is_file():
        raise FileNotFoundError(
            f"CommonRoad goal metadata is missing at {source}; rebuild the scene cache."
        )
    metadata = json.loads(source.read_text())
    if metadata.get("version") != GOAL_METADATA_VERSION:
        raise RuntimeError(
            f"Unsupported CommonRoad goal metadata version: {metadata.get('version')!r}"
        )
    return metadata


def commonroad_lane_metadata(scenario: Scenario) -> dict[str, dict[str, Any]]:
    """Extract CommonRoad lane attributes omitted by the VectorMap proto."""
    country = SupportedTrafficSignCountry(scenario.scenario_id.country_id)
    interpreter = TrafficSignInterpreter(
        country=country,
        lanelet_network=scenario.lanelet_network,
    )
    metadata: dict[str, dict[str, Any]] = {}
    for lanelet in scenario.lanelet_network.lanelets:
        speed_limit = interpreter.speed_limit(frozenset({lanelet.lanelet_id}))
        stop_line = None
        if lanelet.stop_line is not None:
            stop_line = {
                "start": np.asarray(lanelet.stop_line.start, dtype=float).tolist(),
                "end": np.asarray(lanelet.stop_line.end, dtype=float).tolist(),
                "line_marking": lanelet.stop_line.line_marking.value,
                "traffic_sign_ref": sorted(
                    int(value) for value in (lanelet.stop_line.traffic_sign_ref or set())
                ),
                "traffic_light_ref": sorted(
                    int(value) for value in (lanelet.stop_line.traffic_light_ref or set())
                ),
            }
        metadata[str(lanelet.lanelet_id)] = {
            "source": "commonroad",
            "speed_limit_mps": None if speed_limit is None else float(speed_limit),
            "lanelet_types": sorted(value.value for value in lanelet.lanelet_type),
            "user_one_way": sorted(value.value for value in lanelet.user_one_way),
            "user_bidirectional": sorted(value.value for value in lanelet.user_bidirectional),
            "line_marking_left": lanelet.line_marking_left_vertices.value,
            "line_marking_right": lanelet.line_marking_right_vertices.value,
            "adj_left_same_direction": lanelet.adj_left_same_direction,
            "adj_right_same_direction": lanelet.adj_right_same_direction,
            "traffic_sign_ids": sorted(int(value) for value in lanelet.traffic_signs),
            "traffic_light_ids": sorted(int(value) for value in lanelet.traffic_lights),
            "stop_line": stop_line,
        }
    return metadata


def write_vector_map_metadata(path: str | Path, metadata: dict[str, dict[str, Any]]) -> None:
    Path(path).write_text(json.dumps(metadata, indent=2, sort_keys=True))


def apply_vector_map_metadata(vector_map: VectorMap, path: str | Path) -> None:
    """Restore CommonRoad source-map metadata after a VectorMap proto reload."""
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Vector-map metadata sidecar not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    for lane in vector_map.lanes:
        lane_id = str(lane.id)
        if lane_id not in metadata:
            raise RuntimeError(
                f"Vector-map metadata sidecar has no entry for lane {lane_id!r}."
            )
        lane.map_metadata = metadata[lane_id]
        lane.speed_limit_mps = metadata[lane_id]["speed_limit_mps"]

class CommonRoadScenarios:
    def __init__(self, data_dir: Path,) -> None:

        self.data_dir = Path(data_dir)
        self.scenario_files = list(self.data_dir.glob("*.xml"))
        self.scenario_files = [f for f in self.scenario_files if CommonRoadFileReader(f).open()[0].dt == COMMONROAD_DT]          

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

            import commonroad_velocity_planner.fast_api as cvp_fast_api

            global_trajectory = cvp_fast_api.global_trajectory_from_scenario_and_planning_problem(
                scenario=scenario,
                planning_problem=planning_problem,
                use_regulatory_elements=False,
            )
            start_idx = global_trajectory.get_closest_idx(np.array(planning_problem.initial_state.position))
            velocities = np.asarray(global_trajectory.velocity_profile[start_idx:], dtype=np.float64)
            interpoint_distance = np.asarray(global_trajectory.interpoint_distance[start_idx:], dtype=np.float64)
            total_duration_s = float(np.sum(interpoint_distance / np.maximum(velocities, 0.01)))
            ego_route_steps = int(np.floor(total_duration_s / COMMONROAD_DT)) + 1

            return max(max_timestep, ego_route_steps)

        except Exception as e:
            print(f"Error calculating length for scenario {idx}: {e}")
            return 1  # Return minimum length to avoid crashes
    
    def load_scenario(self, idx: int) -> Tuple[Scenario, PlanningProblem]:
        path = self.get_scenario_path(idx)
        scenario, planning_problem_set = CommonRoadFileReader(path).open()
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
    elif agent_type == ObstacleType.UNKNOWN:
        return AgentType.UNKNOWN
    return AgentType.UNKNOWN


def pad_and_interpolate_array(data: np.ndarray, initial_ts_cr: int, final_ts_cr: int, len_scene_ts: int) -> np.ndarray:    #NOTE : "cr" ==> CommonRoad indexing. that is 1,2,3,4,....
    
    data = pd.DataFrame(data).interpolate(limit_area="inside").to_numpy()
    data = np.pad(array=data,
           pad_width=((initial_ts_cr-1, len_scene_ts-final_ts_cr), (0,0)),
           mode='constant',
           constant_values=-1e8,
           )
    return data

def extract_vectorized(
    lanelet_network: LaneletNetwork, country:str, map_name: str, verbose: bool = False
) -> VectorMap:
    def _savgol_interp(point_array):
        # smooth arrays
        line = Polyline(point_array).interpolate(max_dist=3)
        window = min(11, line.points.shape[0] if line.points.shape[0] % 2 == 1 else line.points.shape[0] - 1)
        if window < 5:
            return Polyline(point_array)
        x = savgol_filter(line.points[:,0], window,3)
        y = savgol_filter(line.points[:,1], window,3)
        smoothed = np.column_stack([x, y])

        # Force exact endpoint match to original geometry.
        smoothed[0] = point_array[0, :2]
        smoothed[-1] = point_array[-1, :2]
        return Polyline(smoothed)
    
    vec_map = VectorMap(map_id=map_name)
    max_pt = np.array([np.nan, np.nan, np.nan])
    min_pt = np.array([np.nan, np.nan, np.nan])
    speed_limit_interpreter = TrafficSignInterpreter(country, lanelet_network)
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
            speed_limit = speed_limit_interpreter.speed_limit((lanelet.lanelet_id,))
            road_lane= RoadLane( #WithSpeedLimit(
                id=str(lanelet.lanelet_id),
                
                center=_savgol_interp(lanelet.center_vertices),       
                left_edge=_savgol_interp(lanelet.left_vertices),
                right_edge=_savgol_interp(lanelet.right_vertices),
                adj_lanes_left=adj_lanes_left,
                adj_lanes_right=adj_lanes_right,
                next_lanes=next_lanes,
                prev_lanes=prev_lanes,                
            )
            road_lane.speed_limit = speed_limit
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
        vec_map.lanes = list(vec_map.elements[MapElementType.ROAD_LANE].values())
    #Now do something about bounding box

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
    polygon_points = lanelet.left_vertices.copy()

    polygon_points = np.concatenate([
        polygon_points,
        lanelet.right_vertices[::-1]        #Right boundary should be reversed
        ], axis=0)
    polygon_points = np.concatenate([
        polygon_points,
        polygon_points[0:1]
    ], axis=0)
    return Polyline(polygon_points)

def check_obstacle_validity(dynamic_obstacle : DynamicObstacle) -> bool:

    if (translate_agent_type(dynamic_obstacle.obstacle_type) == AgentType.UNKNOWN):
        print(f"{dynamic_obstacle.obstacle_id} is of unrecognized Agent Type {dynamic_obstacle.obstacle_type}", flush=True)
        return False
    
    elif not isinstance(dynamic_obstacle.prediction, TrajectoryPrediction):
        print(f"{dynamic_obstacle.obstacle_id} is of unsupported Prediction {type(dynamic_obstacle.prediction)}")
        return False
    
    elif not isinstance(dynamic_obstacle.obstacle_shape, Rectangle):
        print(f"{dynamic_obstacle.obstacle_id} is of unsupported Shape {type(dynamic_obstacle.obstacle_shape)}")
    
    else : 
        return True

def check_state_validity(state: State) -> bool:
    validity=False
    allowed_states = (PMState, KSState, ExtendedPMState, CustomState)     #NOTE : KST states are subclasses of KSState only
    if not (state.is_uncertain_position or state.is_uncertain_orientation) :
        if isinstance(state, allowed_states):
            validity=True

    return validity
