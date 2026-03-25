import glob
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, Final, Generator, Iterable, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import yaml

from commonroad.common.file_reader  import CommonRoadFileReader
try:
    from commonroad.scenario.scenario import Scenario
    from commonroad.scenario.obstacle import Obstacle, StaticObstacle, DynamicObstacle, EnvironmentObstacle, PhantomObstacle, ObstacleType, Prediction, TrajectoryPrediction, SetBasedPrediction
    from commonroad.geometry.shape import Shape, Rectangle, Circle, Polygon
    from commonroad.scenario.state import State, KSState, InitialState, PMState, KSTState, STState, STDState, MBState, LongitudinalState, LateralState, InputState, PMInputState, CustomState, ExtendedPMState
    from commonroad.scenario.lanelet import Lanelet, LaneletNetwork, LaneletType
    from commonroad.common.file_reader  import CommonRoadFileReader
    from commonroad.planning.planning_problem import PlanningProblem
except Exception:
    CommonRoadFileReader = None  # type: ignore

from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.scene_metadata import Scene
from trajdata.maps import TrafficLightStatus, VectorMap
from trajdata.maps.vec_map_elements import (
    MapElementType,
    PedCrosswalk,
    PedWalkway,
    Polyline,
    RoadArea,
    RoadLane,
)
from trajdata.utils import map_utils

COMMONROAD_DT: Final[float] = 0.05

class CommonRoadScenarios:
    def __init__(self, data_dir: Path,) -> None:

        self.data_dir = Path(data_dir)
        self.scenario_files = list(self.data_dir.glob("*.xml"))
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
            scenario, _ = self.load_scenario(idx)
            
            # Find the maximum timestep across all dynamic obstacles
            max_timestep = 1
            """
            if scenario.dynamic_obstacles==[]:    #NOTE : No longer required hopefully, but this was for Handling case for no dynamic obstacles
                max_timestep = 100
                return max_timestep
            """
            
            for obstacle in scenario.dynamic_obstacles:
                final_timestep = obstacle.prediction.trajectory.final_state.time_step       #Final recorded time step of the scenario.
                if final_timestep > max_timestep:
                    max_timestep = final_timestep
            # No need to add 1 because timesteps are 1-indexed (timestep 1, 2, ... max_timestep)
            return max_timestep
            
        except Exception as e:
            print(f"Error calculating length for scenario {idx}: {e}")
            return 1  # Return minimum length to avoid crashes
    
    def load_scenario(self, idx: int) -> Tuple[Scenario, PlanningProblem]:
        path = self.get_scenario_path(idx)
        scenario, planning_problem_set = CommonRoadFileReader(path).open()
        return scenario, planning_problem_set

def translate_agent_type(agent_type : ObstacleType):           #Types here might need to be reviewed. Example, where to put ObstacleType.TRAIN? Also, ObstacleType.PARKED_VEHICLE i probably Static, but ok. Also, MOTORCYCLE goes into AgentType.BICYCLE or AgentType.VEHICLE?
    if agent_type in {ObstacleType.CAR, ObstacleType.TRUCK, ObstacleType.PRIORITY_VEHICLE, ObstacleType.PARKED_VEHICLE, ObstacleType.TAXI, ObstacleType.BUS } :
        return AgentType.VEHICLE
    elif agent_type == ObstacleType.PEDESTRIAN:
        return AgentType.PEDESTRIAN
    elif agent_type in {ObstacleType.BICYCLE, ObstacleType.MOTORCYCLE}:
        return AgentType.BICYCLE
    elif agent_type == ObstacleType.UNKNOWN:
        return AgentType.UNKNOWN
    return AgentType.UNKNOWN


def pad_and_interpolate_array(data: np.ndarray, initial_ts_cr: int, final_ts_cr: int, len_scene_ts: int) -> np.ndarray:    #NOTE : "cr" ==> CommonRoad indexing. that is 1,2,3,4,....
    
    data = pd.DataFrame(data).interpolate(limit_area="inside").to_numpy()
    data = np.pad(array=data,
           pad_width=((initial_ts_cr-1, len_scene_ts-final_ts_cr), (0,0)),
           mode='constant',
           constant_values=np.nan,
           )
    return data

def extract_vectorized(
    lanelet_network: LaneletNetwork, map_name: str, verbose: bool = False
) -> VectorMap:
    
    vec_map = VectorMap(map_id=map_name)
    max_pt = np.array([np.nan, np.nan, np.nan])
    min_pt = np.array([np.nan, np.nan, np.nan])

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
            max_pt = np.fmax(max_pt, crosswalk.center.xyz.max(axis=0))
            min_pt = np.fmin(min_pt, crosswalk.center.xyz.min(axis=0))
            vec_map.add_map_element(crosswalk)

        elif elem_type==MapElementType.PED_WALKWAY:
            polygon=lanelet_to_polygon(lanelet)
            if polygon.points.size==0:
                continue
            walkway = PedWalkway(
                id=str(lanelet.lanelet_id),
                polygon=polygon,
                )
            max_pt = np.fmax(max_pt, walkway.center.xyz.max(axis=0))
            min_pt = np.fmin(min_pt, walkway.center.xyz.min(axis=0))
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
            road_lane= RoadLane(
                id=str(lanelet.lanelet_id),
                center=Polyline(lanelet.center_vertices),        #lanelet.left_vertices has only x and y dimensions btw, but not an issue. Polyline handles it during initialization.
                left_edge=Polyline(lanelet.left_vertices),
                right_edge=Polyline(lanelet.right_vertices),
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
    #Now do something about bounding box

    return vec_map


def translate_lanelet_type(lanelet_type: Set[LaneletType]) -> MapElementType:
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