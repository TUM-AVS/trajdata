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
    from commonroad.scenario.obstacle import Obstacle, StaticObstacle, DynamicObstacle, EnvironmentObstacle, PhantomObstacle, Prediction, TrajectoryPrediction, SetBasedPrediction
    from commonroad.scenario.trajectory import Trajectory
    from commonroad.scenario.state import State, KSState, InitialState, PMState, KSTState, STState, STDState, MBState, LongitudinalState, LateralState, InputState, PMInputState, CustomState, ExtendedPMState
    from commonroad.common.file_reader  import CommonRoadFileReader
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

from trajdata.dataset_specific.commonroad.commonroad_utils import CommonRoadScenarios, translate_agent_type, pad_and_interpolate_array, check_obstacle_validity, check_state_validity
from trajdata.dataset_specific.scene_records import CommonRoadSceneRecord

def const_lambda(const_val: Any) -> Any:
    return const_val

class CommonRoadDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str)-> EnvMetadata:
        dataset_parts = [(env_name,)]                    #As we have no parts (categories) as such. Lets just fill it with env_name then.
        scene_split_map = defaultdict(partial(const_lambda, const_val = "commonroad"))         #As we have no splits within our parts either
        return EnvMetadata(
            name = env_name,
            data_dir=data_dir,
            dt = commonroad_utils.COMMONROAD_DT,
            parts=dataset_parts,
            scene_split_map=scene_split_map,
        )

    def load_dataset_obj(self, verbose = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset ", flush = True)
        dataset_name = "commonroad_train"
        self.dataset_obj = CommonRoadScenarios(data_dir = self.metadata.data_dir)         #For handling reading of .xml files

        if verbose:
            print(f"Found {self.dataset_obj.num_scenarios} scenarios", flush=True)

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,                                                    #Set of Labels to search in its "tags" for fast categorical filtering, e.g {"intersection", "daytime"}, etc
        scene_desc_contains: Optional[List[str]],                               #Look for these in the scene description text (if any)
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        
        all_scenes_list: List[CommonRoadSceneRecord] = list()                   #Will be cached at the end, for saving computation in future.
        scenes_list: List[SceneMetadata] = list()

        for idx in range(self.dataset_obj.num_scenarios):
            scene_name: str = self.dataset_obj.get_scenario_name(idx)           #identifier for the scene
            scene_split: str = self.metadata.scene_split_map[scene_name]        #will remain "train" only for us right now.
            scene_length: int = self.dataset_obj.get_scenario_length(idx)       #Commonroad has different lengths for each scenario

            scene_record = CommonRoadSceneRecord(
                    name=scene_name,
                    length=str(scene_length), 
                    data_idx=idx
                )
            all_scenes_list.append(scene_record)

            matches_split = scene_split in scene_tag  # Does the split match what user wants?
            matches_description = True  # Assume True if no description filter, else check in next snippet
            
            if scene_desc_contains is not None:
                matches_description = False
                # Check if any of the required strings are in the scene name
                for required_text in scene_desc_contains:
                    if required_text.lower() in scene_name.lower():
                        matches_description = True
                        break
            if matches_split and matches_description:                           #If matches all criteria, then append its metadata to scenes_list
                scene_metadata = SceneMetadata(
                    env_name = self.metadata.name,
                    name = scene_name,
                    dt = self.metadata.dt,                                      #NOTE : TO CHECK if dt is same for each scenario in commonroad or not. And to figure out trajdata behaviour as well if it isn't
                    raw_data_idx = idx 
                )
                scenes_list.append(scene_metadata)

        self.cache_all_scenes_list(env_cache, all_scenes_list)                  #Cache the basic info about all the scenarios we parsed through
        return scenes_list
            
    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        _, name, _, data_idx = scene_info
        scene_name: str = scene_info
        scene_name: str = name
        scene_split: str = self.metadata.scene_split_map[scene_name]
        scene_length: int = self.dataset_obj.get_scenario_length(data_idx)

        return Scene(
            env_metadata=self.metadata,
            name=name,
            location=f"{self.name}_{data_idx}",            #NOTE : Can add this later on.
            data_split=scene_split,
            length_timesteps=scene_length,
            raw_data_idx=data_idx, 
            data_access_info=None,                          
        )
    
    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        all_scenes_list : List[CommonRoadSceneRecord] = env_cache.load_env_scenes_list(self.name)

        scenes_list: List[SceneMetadata] = list()

        for scene_record in all_scenes_list :
            scene_name, scene_length, data_idx = scene_record
            scene_split: str = self.metadata.scene_split_map[scene_name]

            if scene_split in scene_tag and scene_desc_contains is None :
                scene_metadata = Scene(                                         #Called scene_metadata as Scene class holds info about the scene, not the literal data inside it. And ig SceneMetadata class is an even more lightweight version, probably for caching and referring to Scene.
                    env_metadata=self.metadata,
                    name=scene_name,
                    location=f"{self.name}_{data_idx}",                         #NOTE : Can be set as something else if Commonroad associates scenes with locations. Not sure what effect/significance this paramater has on the code itself however. 
                    data_split=scene_split, 
                    length_timesteps=scene_length,
                    raw_data_idx=data_idx,
                data_access_info=None                                           #Waymo says that this isn't used if everything is already cached or somehting. Still something to think about.
                )
                scenes_list.append(scene_metadata)
        return scenes_list

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        
        agent_list: List[AgentMetadata] = []
        agent_presence: List[List[AgentMetadata]] = [
            [] for _ in range(scene.length_timesteps)
        ]

        scenario: Scenario
        scenario, _ = self.dataset_obj.load_scenario(scene.raw_data_idx)

        #Used this print statement for debugging which scenes were problematic. Now they're removed though.
        #print(f"Entering Scene : {scene.name},  Data idx : {scene.raw_data_idx},  No. of dynamic obstacles : {len(scenario.dynamic_obstacles)}, No. of timesteps : {scene.length_timesteps}", flush=True)

        agent_ids = []
        all_agent_data = []
        agents_to_remove = []
        ego_id = None               #NOTE : Not given any ego_id right now.

        for index, dynamic_obstacle in enumerate(scenario.dynamic_obstacles):
            if not check_obstacle_validity(dynamic_obstacle) :      #Ensuring that "obstacle" is valid
                continue
            
            agent_type: AgentType = translate_agent_type(dynamic_obstacle.obstacle_type)
            agent_id: int = dynamic_obstacle.obstacle_id
            agent_ids.append(agent_id)

            prediction : TrajectoryPrediction = dynamic_obstacle.prediction            #Represents the states at all timesteps, for current particular agent

            translations = []
            velocities = []
            #sizes = []     #Commonroad has Fixed Extents, so no need to store sizes in the cached dataframe. Waymo had it cuz it would vary slightly due to sensor error etc.
            yaws = []
            
            trajectory: Trajectory = prediction.trajectory
            for state in trajectory.state_list:            #Key issue : This structure/code assumes that each "prediction" has exactly length = len_timesteps (of the given scene), and someone it is not active, then that state is represented by Null. Else, may have to write function to expand the length accordingly till t=0 (before the array) and t=lem_timestep(after the array). "state" here == at time=t

                if check_state_validity(state):           #ensuring that is is instance of PMState or KSState
                    state : Union[PMState, KSState]
                    translations.append(
                        (state.position[0], state.position[1], 0)
                        )                                        #NOTE : Check if commonroad uses z=0 or z=h/2
                    velocities.append((state.velocity, state.velocity_y))
                    yaws.append(state.orientation)
                    #sizes.append((dynamic_obstacle.obstacle_shape.length, dynamic_obstacle.obstacle_shape.width, 0))
                    
                else:
                    print(f"Invalid State encountered of type {type(state)} in agent {dynamic_obstacle.obstacle_id} of scene {scene.name} ; continuing...", flush=True)
                    translations.append((np.nan, np.nan, np.nan))
                    velocities.append((np.nan, np.nan))
                    yaws.append(np.nan)
                    #sizes.append((np.nan, np.nan, np.nan))
                    
                
            curr_agent_data = np.concatenate(                   #Check validity of concantenation after implemmenations too pls
                (
                    translations, 
                    velocities, 
                    np.expand_dims(yaws, axis=1),               #just changes shape from (T,) to (T,1). thus makes it 2D array from a List (which is always 1D, as lists dont have concept of matrices), for concatenation.
                    #sizes,
                ),
                axis=1,
            )

            curr_agent_data = pad_and_interpolate_array(curr_agent_data, trajectory.initial_time_step, trajectory.final_state.time_step, scene.length_timesteps)            #To fill "Internal" missing values. Note that the size our data is len_timesteps only, for each column. We will drop columns in the last aftter converting to dataframe.

            all_agent_data.append(curr_agent_data)
            first_timestep = pd.Series(curr_agent_data[:, 0]).first_valid_index()
            last_timestep = pd.Series(curr_agent_data[:, 0]).last_valid_index()
            if first_timestep is None or last_timestep is None :
                first_timestep=0
                last_timestep=0
            
            agent_name = str(agent_id)
            #insert something to recognize ego vehicle separately (thru its ID maybe) and then give name = "ego"

            extent = FixedExtent(dynamic_obstacle.obstacle_shape.length, dynamic_obstacle.obstacle_shape.width, 0)     #Using the length, width, height of the given agent. In commonroad they are fixed values.
            agent_info = AgentMetadata(
                name=agent_name,
                agent_type=agent_type,
                first_timestep=first_timestep,
                last_timestep=last_timestep,
                extent=extent,
            )

            if last_timestep-first_timestep>0 :
                agent_list.append(agent_info)
                for timestep in range(first_timestep, last_timestep+1):         #NOTE : trajdata uses indexing of timesteps from 0 to n-1, but commonroad uses timesteps from 1 to n. 
                                                                                # But it isnt causing any issue over here, as commonroad's timesteps isn't really being used anywhere except for length_scene, where it eitherways is handled properly. 
                                                                                # Probably will still need that while expanding the length of dynamic_obstacle's timesteps to match that of the scene. 
                    agent_presence[timestep].append(agent_info)         #agent_presence = List of timesteps. Each index pe gonna list all agents that are active at that timestep.
            else :
                agents_to_remove.append(agent_id)                       #Will drop these agents if they appeaared for just 1 (or 0) timestep. Will do it in the end after creating dataframe etc.

        traj_cols = ["x", "y", "z", "vx", "vy", "heading"]

        """
        if all_agent_data==[]:                      #Handling case for empty all_agent_data. ==> i.e when the scene doesn't have any dynamic obstacle only, so it never entered the loop at all.
            all_agent_data = [np.full((scene.length_timesteps, 6), np.nan)]
            agent_ids = np.repeat(np.nan, scene.length_timesteps)
            agent_frame_ids = np.arange(scene.length_timesteps)
            all_agent_data_df = pd.DataFrame(
                np.concatenate(all_agent_data), 
                columns = traj_cols, #+extent_cols,
                index = [agent_ids, agent_frame_ids],
            )
            mask = pd.isna(all_agent_data_df).all(axis=1, bool_only=False)
        """    
            
    
        agent_ids = np.repeat(agent_ids, scene.length_timesteps)    
        #extent_cols = ["length", "width", "height"]    
        agent_frame_ids = np.resize(
            np.arange(scene.length_timesteps),
            len(agent_ids),                         #As length of agent_ids will be no. of VALID obstacles*scene_ts
        )

        all_agent_data_df = pd.DataFrame(
            np.concatenate(all_agent_data), 
            columns = traj_cols, #+extent_cols,
            index = [agent_ids, agent_frame_ids],
        )
        mask = pd.notna(all_agent_data_df).all(axis=1, bool_only=False)
        all_agent_data_df=all_agent_data_df.loc[mask]           #removing rows with ANY missing value.


        all_agent_data_df.index.names = ["agent_id", "scene_ts"]
        all_agent_data_df.sort_index(inplace=True)
        all_agent_data_df.reset_index(level=1, inplace=True)    #Removed scene_ts from indices to operate with it etc.
            
        #print(f"agent_ids is \n{agent_ids.shape}\n{agent_ids}", flush=True)
        #print(f"agent_frame_ids is \n{agent_frame_ids.shape}\n{agent_frame_ids}", flush=True)
        #print(f"all_agent_data is \n{all_agent_data.__len__()}\n{all_agent_data}", flush=True)
        #print(f"all_agent_data[0] is \n{all_agent_data[0]}", flush=True)

        try : 
            all_agent_data_df[["ax", "ay"]] = (
                arr_utils.agent_aware_diff(
                    all_agent_data_df[["vx", "vy"]].to_numpy(), agent_ids[mask]
                )
                / commonroad_utils.COMMONROAD_DT
            )

        except IndexError as e :
            print(e, flush=True)
            print(f"All agent data : {all_agent_data.__len__()} \n{all_agent_data}", flush=True)
            print(f"All agent data df : {all_agent_data_df.shape} \n{all_agent_data_df}", flush=True)
            print(f"This error happened in Scene : {scene.name},  Data idx : {scene.raw_data_idx},  No. of dynamic obstacles : {len(scenario.dynamic_obstacles)}, No. of timesteps : {scene.length_timesteps}", flush=True)
            

        final_cols = [
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "ax",
            "ay",
            "heading",
        ] #+ extent_cols

        # Removing agents with only one detection.
        all_agent_data_df.drop(index=agents_to_remove, inplace=True)

        # Changing the agent_id dtype to str
        all_agent_data_df.reset_index(inplace=True)                                 #Again, this time removed agent_ids from index to change type to str etc.
        all_agent_data_df["agent_id"] = all_agent_data_df["agent_id"].astype(str)
        all_agent_data_df.set_index(["agent_id", "scene_ts"], inplace=True)         #set them as indices again after the cleaning etc.
        all_agent_data_df.rename(
            index={str(ego_id): "ego"}, inplace=True, level="agent_id"              #set ego_id as "ego" instead, convinient for usage later on and maybe even important for trajdata
        )
        
        cache_class.save_agent_data(
            all_agent_data_df.loc[:, final_cols],
            cache_path,
            scene,
        )

        #---Insert code here for traffic data caching if u want later---

        return agent_list, agent_presence
        
    def cache_map(
        self,
        data_idx: int,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ):
        scenario: Scenario
        scenario, _ = self.dataset_obj.load_scenario(data_idx)
        vector_map: VectorMap = commonroad_utils.extract_vectorized(
            lanelet_network=scenario.lanelet_network,
            map_name=f"{self.name}:{self.name}_{data_idx}",
        )
        map_cache_class.finalize_and_cache_map(cache_path, vector_map, map_params)

    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        """
        Get static, scene-level info from the source dataset, caching it
        to cache_path. (Primarily this is info needed to construct VectorMap)
        
        Resolution is in pixels per meter.
        """

        num_workers: int = map_params.get("num_workers", 0)
        if num_workers > 1:
            parallel_apply(
                partial(
                    self.cache_map,
                    cache_path=cache_path,
                    map_cache_class=map_cache_class,
                    map_params=map_params,
                ),
                range(self.dataset_obj.num_scenarios),
                num_workers=num_workers,            #NOTE : Can add a line that updates self.metadata's locations list (present in class EnvMetadata.map_locations). Then we can access list of all maps in an environment from dataset.envs[i].metadata.map_locations (easy iteration over the list)
            )
        else:
            for i in tqdm.trange(self.dataset_obj.num_scenarios):
                self.cache_map(i, cache_path, map_cache_class, map_params)
