from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from collections import defaultdict
from functools import partial
import warnings

import numpy as np
import pandas as pd
import tqdm

from trajdata.dataset_specific.commonroad import commonroad_utils

# Try to import CommonRoad reader; if not present, users will need to install commonroad-io
try:
    from commonroad.scenario.scenario import Scenario
    from commonroad.scenario.obstacle import (
        Obstacle,
        StaticObstacle,
        DynamicObstacle,
        EnvironmentObstacle,
        PhantomObstacle,
        Prediction,
        TrajectoryPrediction,
        SetBasedPrediction,
    )
    from commonroad.scenario.trajectory import Trajectory
    from commonroad.scenario.state import (
        State,
        KSState,
        InitialState,
        PMState,
        KSTState,
        STState,
        STDState,
        MBState,
        LongitudinalState,
        LateralState,
        InputState,
        PMInputState,
        CustomState,
    )
    from commonroad.common.file_reader import CommonRoadFileReader
    from commonroad.planning.planning_problem import PlanningProblem
except Exception:
    CommonRoadFileReader = None  # type: ignore

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures.agent import (
    AgentMetadata,
    AgentType,
    FixedExtent,
    VariableExtent,
)
from trajdata.data_structures.environment import EnvMetadata
from trajdata.data_structures.scene_metadata import Scene, SceneMetadata
from trajdata.data_structures.scene_tag import SceneTag
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.maps.vec_map import VectorMap
from trajdata.utils import arr_utils
from trajdata.utils.parallel_utils import parallel_apply

from trajdata.dataset_specific.commonroad import CommonRoadDataset
from trajdata.dataset_specific.commonroad.commonroad_utils import (
    CommonRoadScenarios,
    translate_agent_type,
    pad_and_interpolate_array,
    check_obstacle_validity,
    check_state_validity,
)
from trajdata.dataset_specific.scene_records import CommonRoadSceneRecord

env_name = "commonroad_dev"
data_dir = "/home/yogeshwar10/avs/commonroad/cr_scenarios/alienware_scenarios"
object = CommonRoadDataset(
    env_name=env_name, data_dir=data_dir, parallelizable=False, has_maps=False
)

object.load_dataset_obj(verbose=True)
scenario_idx: int = (
    0  # The ones with no dynamic obstacles were ZAM_Over-1_1, ZAM_Urban-3_3, ZAM_Urban-2_1, ZAM_Urban-3_1
)
len_scene_ts = object.dataset_obj.get_scenario_length(scenario_idx)
scenario, _ = object.dataset_obj.load_scenario(scenario_idx)
print(f"Scneario name : {scenario.scenario_id}")
print(f"Scenario length : {len_scene_ts}")
print(f"Number of dynamic obstacles : {scenario.dynamic_obstacles.__len__()}")
print(
    f"Entering Scene : {scenario.scenario_id},  Data idx : {scenario_idx},  No. of dynamic obstacles : {len(scenario.dynamic_obstacles)}, No. of timesteps : {len_scene_ts}",
    flush=True,
)


"""
#For checking extract_vectorized()

lanelet_network = scenario.lanelet_network
vec_map = commonroad_utils.extract_vectorized(
    lanelet_network=lanelet_network, map_name="hello:hello", verbose=False
)

# print(lanelet_network.lanelets[1].left_vertices.shape)
print(lanelet_network.lanelets[0].predecessor)
# print(vec_map.elements[MapElementType.ROAD_LANE][str(102647)])
"""

# For Checking get_agent_info()

"""
for _, dynamic_obstacle in enumerate(scenario.dynamic_obstacles):

    prediction = dynamic_obstacle.prediction
    print(
        f"Obstacle ID : {dynamic_obstacle.obstacle_id}, Initial time step : {prediction.initial_time_step}, Final Time step : {prediction.final_time_step}"
    )

    # print(prediction.trajectory.state_list)

"""
obstacle_idx = 0
dynamic_obstacle = scenario.dynamic_obstacles[obstacle_idx]
prediction = dynamic_obstacle.prediction

agent_type: AgentType = translate_agent_type(dynamic_obstacle.obstacle_type)
agent_id: int = dynamic_obstacle.obstacle_id

translations = []
velocities = []
yaws = []

trajectory: Trajectory = prediction.trajectory
for state in trajectory.state_list:
    translations.append(
        (state.position[0], state.position[1], 0)
    )  # NOTE : Check if commonroad uses z=0 or z=h/2
    velocities.append((state.velocity, state.velocity_y))
    yaws.append(state.orientation)

curr_agent_data = np.concatenate(  # Check validity of concantenation after implemmenations too pls
    (
        translations,
        velocities,
        np.expand_dims(
            yaws, axis=1
        ),  # just changes shape from (T,) to (T,1). thus makes it 2D array from a List (which is always 1D, as lists dont have concept of matrices), for concatenation.
        # sizes,
    ),
    axis=1,
)


curr_agent_data = pad_and_interpolate_array(
    curr_agent_data,
    trajectory.initial_time_step,
    trajectory.final_state.time_step,
    len_scene_ts,
)  # To fill "Internal" missing values. Note that the size our data is len_timesteps only, for each column. We will drop columns in the last aftter converting to dataframe.


print(f"curr_agent_data shape is \n{curr_agent_data.shape}")

agent_ids = []
agent_ids.append(agent_id)
all_agent_data = []
all_agent_data.append(curr_agent_data)

agent_ids = np.repeat(agent_ids, len_scene_ts)
traj_cols = ["x", "y", "z", "vx", "vy", "heading"]
# extent_cols = ["length", "width", "height"]
agent_frame_ids = np.resize(
    np.arange(len_scene_ts),
    len(agent_ids),  # As length of agent_ids will be no. of VALID obstacles*scene_ts
)

all_agent_data_df = pd.DataFrame(
    np.concatenate(all_agent_data),
    columns=traj_cols,  # +extent_cols,
    index=[agent_ids, agent_frame_ids],
)
print(f"Dynamic Obstacles in scene is {len(scenario.dynamic_obstacles)}")
# print(f"all_agent_data_df is \n{all_agent_data_df}")


# Testing a section of code alone :

all_agent_data = [
    np.full((len_scene_ts, 6), np.nan),
]
agent_ids = np.repeat(np.nan, len_scene_ts)
agent_frame_ids = np.arange(len_scene_ts)
all_agent_data_df = pd.DataFrame(
    np.concatenate(all_agent_data),
    columns=traj_cols,  # +extent_cols,
    index=[agent_ids, agent_frame_ids],
)

# print(f"agent_ids is \n{agent_ids.shape}\n{agent_ids}", flush=True)
# print(f"agent_frame_ids is \n{agent_frame_ids.shape}\n{agent_frame_ids}", flush=True)
# print(f"all_agent_data is \n{all_agent_data.__len__()}\n{all_agent_data}", flush=True)
# print(f"all_agent_data[0] is \n{all_agent_data[0]}", flush=True)

all_agent_data_df[["ax", "ay"]] = (
    arr_utils.agent_aware_diff(all_agent_data_df[["vx", "vy"]].to_numpy(), agent_ids)
    / commonroad_utils.COMMONROAD_DT
)

print(all_agent_data_df)
