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
        # Each configured directory is an independent trajdata environment.  Use
        # that name consistently for the scene tag, cached scene list, and map.
        scene_split_map = defaultdict(partial(const_lambda, const_val=env_name))
        map_locations = [f.stem for f in Path(data_dir).glob("*.xml")]
        """Get timeStepSize for all scenario files"""
        return EnvMetadata(
            name = env_name,
            data_dir=data_dir,
            dt = float(self.dataset_options.get("desired_dt", commonroad_utils.COMMONROAD_DT)),
            parts=dataset_parts,
            scene_split_map=scene_split_map,
            map_locations=map_locations
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
            location=scene_name,          
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
                    location=scene_name,                         #NOTE : Can be set as something else if Commonroad associates scenes with locations. Not sure what effect/significance this paramater has on the code itself however. 
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
        scenario, planning_problem_set = self.dataset_obj.load_scenario(scene.raw_data_idx)
        planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
        #Used this print statement for debugging which scenes were problematic. Now they're removed though.
        #print(f"Entering Scene : {scene.name},  Data idx : {scene.raw_data_idx},  No. of dynamic obstacles : {len(scenario.dynamic_obstacles)}, No. of timesteps : {scene.length_timesteps}", flush=True)

        agent_ids = []
        all_agent_data = []
        agents_to_remove = []
        # ego_id = None               #NOTE : Not given any ego_id right now.

        for index, dynamic_obstacle in enumerate(scenario.dynamic_obstacles):
            if not check_obstacle_validity(dynamic_obstacle) :      #Ensuring that "obstacle" is valid
                continue
            
            agent_type: AgentType = translate_agent_type(dynamic_obstacle.obstacle_type)
            agent_id: int = dynamic_obstacle.obstacle_id
            agent_ids.append(agent_id)
            ini_state = dynamic_obstacle.initial_state
            prediction : TrajectoryPrediction = dynamic_obstacle.prediction            #Represents the states at all timesteps, for current particular agent

            translations = [(ini_state.position[0], ini_state.position[1], 0)]         #Initializing the list with the initial state. Will be useful for padding and interpolation later on, in case of missing values at the beginning of the trajectory. Also, we will be dropping all rows with missing values in the end, so if we don't add this initial state here, then we might end up dropping the whole trajectory of an agent just because it is missing at t=0, which is a common case in commonroad as many agents appear after t=0.
            velocities = [(np.cos(ini_state.orientation) * ini_state.velocity, np.sin(ini_state.orientation) * ini_state.velocity)]       #Same as above, initializing with the initial state velocity. Also, we can compute it from the initial state itself, so no problem of missing value at t=0 for velocity. But still adding it here for consistency and to avoid issues in padding and interpolation later on.
            #sizes = []     #Commonroad has Fixed Extents, so no need to store sizes in the cached dataframe. Waymo had it cuz it would vary slightly due to sensor error etc.
            yaws = [ini_state.orientation]       #Same as above, initializing with the initial state yaw. Also, we can compute it from the initial state itself, so no problem of missing value at t=0 for yaw. But still adding it here for consistency and to avoid issues in padding and interpolation later on.
            
            trajectory: Trajectory = prediction.trajectory
            for state in trajectory.state_list:            #Key issue : This structure/code assumes that each "prediction" has exactly length = len_timesteps (of the given scene), and someone it is not active, then that state is represented by Null. Else, may have to write function to expand the length accordingly till t=0 (before the array) and t=lem_timestep(after the array). "state" here == at time=t

                if check_state_validity(state):           #ensuring that is is instance of PMState or KSState
                    state : Union[PMState, KSState]
                    translations.append(
                        (state.position[0], state.position[1], 0)
                        )                
                    vx = np.cos(state.orientation) * state.velocity   
                    vy = np.sin(state.orientation) * state.velocity         
                    velocities.append((vx, vy))
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
            # if first_timestep is None or last_timestep is None :
            #     first_timestep=0
            #     last_timestep=0
            
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

            # if last_timestep-first_timestep>0 :
            agent_list.append(agent_info)
            for timestep in range(first_timestep, last_timestep):         
                agent_presence[timestep].append(agent_info)         #agent_presence = List of timesteps. Each index pe gonna list all agents that are active at that timestep.
            # else :
            #     agents_to_remove.append(agent_id)                       #Will drop these agents if they appeaared for just 1 (or 0) timestep. Will do it in the end after creating dataframe etc.
        
        #### EGO 
        import commonroad_velocity_planner.fast_api  as cvp_fast_api
        from scipy.interpolate import interp1d, PchipInterpolator

        global_trajectory = cvp_fast_api.global_trajectory_from_scenario_and_planning_problem(
                scenario=scenario, 
                planning_problem=planning_problem, 
                use_regulatory_elements=False
            )
        idx = global_trajectory.get_closest_idx(np.array(planning_problem.initial_state.position)) 

        vs = np.asarray(global_trajectory.velocity_profile[idx:], dtype=np.float64)
        interpoint_distance = np.asarray(global_trajectory.interpoint_distance[idx:], dtype=np.float64)
        time_deltas = interpoint_distance / np.maximum(vs, 0.01)
        time_at_points = np.concatenate([[0.0], np.cumsum(time_deltas)])[:-1]

        positions_x = np.asarray(global_trajectory.reference_path[idx:, 0], dtype=np.float64)
        positions_y = np.asarray(global_trajectory.reference_path[idx:, 1], dtype=np.float64)
        headings = np.unwrap(np.asarray(global_trajectory.path_orientation[idx:], dtype=np.float64))

        interp_x = PchipInterpolator(time_at_points, positions_x, extrapolate=False)
        interp_y = PchipInterpolator(time_at_points, positions_y, extrapolate=False)
        interp_heading = PchipInterpolator(time_at_points, headings, extrapolate=False)
        interp_velocity = PchipInterpolator(time_at_points, vs, extrapolate=False)

        full_duration = float(time_at_points[-1]) if time_at_points.size > 0 else 0.0
        scene_length_timesteps = max(scene.length_timesteps, int(np.floor(full_duration / scenario.dt)) + 1)
        time_samples = np.arange(scene_length_timesteps + 1, dtype=np.float64) * scenario.dt
        clamped_time_samples = np.minimum(time_samples, full_duration)

        sampled_positions = np.column_stack([
            interp_x(clamped_time_samples),
            interp_y(clamped_time_samples),
        ])
        sampled_headings = interp_heading(clamped_time_samples)
        sampled_velocities = interp_velocity(clamped_time_samples)
        # Compute velocity components from heading and speed
        vx = sampled_velocities * np.cos(sampled_headings)
        vy = sampled_velocities * np.sin(sampled_headings)

        # pos_interp_x = interp1d(s_values, global_trajectory.reference_path[:, 0], kind='cubic', fill_value='extrapolate')
        # pos_interp_y = interp1d(s_values, global_trajectory.reference_path[:, 1], kind='cubic', fill_value='extrapolate')
        # heading_interp = interp1d(s_values, global_trajectory.path_orientation, kind='cubic', fill_value='extrapolate')
        # velocity_interp = interp1d(s_values, global_trajectory.velocity_profile, kind='cubic', fill_value='extrapolate')

        # positions = []
        # headings = []
        # velocities = []
        # for ts in range(scene.length_timesteps+1):          #Adding +1 to include the last timestep as well, as range is exclusive of the end value. This is important for us as we want to have the reference trajectory values for all timesteps of the scene, including the last one.
        #     t = ts * scenario.dt
        #     s = t * global_trajectory.average_velocity  # arc length from time
        #     positions.append((pos_interp_x(s), pos_interp_y(s),0))
        #     heading = heading_interp(s)
        #     headings.append(heading)
        #     v = velocity_interp(s)
        #     vx = np.cos(heading) * v  
        #     vy = np.sin(heading) * v         
        #     velocities.append((vx, vy))

        curr_agent_data = np.concatenate(                   #Check validity of concantenation after implemmenations too pls
            (
                sampled_positions, 
                np.expand_dims(np.zeros_like(sampled_positions[:, 0]), axis=1),  
                np.column_stack([vx, vy]), 
                np.expand_dims(sampled_headings, axis=1),               #just changes shape from (T,) to (T,1). thus makes it 2D array from a List (which is always 1D, as lists dont have concept of matrices), for concatenation.
                #sizes,
            ),
            axis=1,
        )

        # curr_agent_data = pad_and_interpolate_array(curr_agent_data, trajectory.initial_time_step, trajectory.final_state.time_step, scene.length_timesteps)            #To fill "Internal" missing values. Note that the size our data is len_timesteps only, for each column. We will drop columns in the last aftter converting to dataframe.

        all_agent_data.append(curr_agent_data)
        first_timestep = pd.Series(curr_agent_data[:, 0]).first_valid_index()
        last_timestep = pd.Series(curr_agent_data[:, 0]).last_valid_index()
        # if first_timestep is None or last_timestep is None :
        #     first_timestep=0
        #     last_timestep=0
        
        agent_name = "ego"
        agent_ids.append(agent_name)
        #insert something to recognize ego vehicle separately (thru its ID maybe) and then give name = "ego"

        extent = FixedExtent(4, 2, 0)     # Dummy Size of Ego
        agent_info = AgentMetadata(
            name=agent_name,
            agent_type=AgentType.VEHICLE,
            first_timestep=first_timestep,
            last_timestep=last_timestep,
            extent=extent,
        )

        # if last_timestep-first_timestep>0 :
        agent_list.append(agent_info)
        for timestep in range(first_timestep, last_timestep):         
            try:
                agent_presence[timestep].append(agent_info) 
            except IndexError:
                agent_presence.append([agent_info])

        ######## 
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
            
    
        agent_ids_ext = np.repeat(agent_ids, len(agent_presence)+1) # scene.length_timesteps+1)    
        #extent_cols = ["length", "width", "height"]    
        agent_frame_ids = np.resize(
            np.arange(len(agent_presence)+1), #scene.length_timesteps+1),
            len(agent_ids_ext),                         #As length of agent_ids will be no. of VALID obstacles*scene_ts
        )

        all_agent_data_df = pd.DataFrame(
            np.concatenate(all_agent_data), 
            columns = traj_cols, #+extent_cols,
            index = [agent_ids_ext, agent_frame_ids],
        )
        # mask = pd.notna(all_agent_data_df).all(axis=1, bool_only=False)
        # all_agent_data_df=all_agent_data_df.loc[mask]           #removing rows with ANY missing value.


        all_agent_data_df.index.names = ["agent_id", "scene_ts"]
        all_agent_data_df.sort_index(inplace=True)
        all_agent_data_df.reset_index(level=1, inplace=True)    #Removed scene_ts from indices to operate with it etc.
            

        # try : 
        all_agent_data_df[["ax", "ay"]] = (
            arr_utils.agent_aware_diff(
                all_agent_data_df[["vx", "vy"]].to_numpy(), agent_ids#[mask]
            )
            / commonroad_utils.COMMONROAD_DT
        )
        #TODO : ax, ay to large at first timestep

        # except IndexError as e :
        #     print(e, flush=True)
        #     print(f"All agent data : {all_agent_data.__len__()} \n{all_agent_data}", flush=True)
        #     print(f"All agent data df : {all_agent_data_df.shape} \n{all_agent_data_df}", flush=True)
        #     print(f"This error happened in Scene : {scene.name},  Data idx : {scene.raw_data_idx},  No. of dynamic obstacles : {len(scenario.dynamic_obstacles)}, No. of timesteps : {scene.length_timesteps}", flush=True)
            

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

        # Changing the agent_id dtype to str and renaming ego
        all_agent_data_df.reset_index(inplace=True)
        all_agent_data_df['agent_id'] = all_agent_data_df['agent_id'].astype(str)
                                        #   .replace(str(ego_id), 'ego'))
        all_agent_data_df.set_index(['agent_id', 'scene_ts'], inplace=True)
        
        cache_class.save_agent_data(
            all_agent_data_df.loc[:, final_cols],
            cache_path,
            scene,
        )

        #---Insert code here for traffic data caching if u want later---
        # TODO Traffic lights!
        if len(scenario.lanelet_network.traffic_lights) > 0:
            print(f"Scene {scene.name} has traffic lights, which are currently not handled in caching. Consider implementing this if you want to use this data for traffic light related research questions.", flush=True)
            pass

        return agent_list, agent_presence
        
    def cache_map(
        self,
        data_idx: int,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ):
        scenario: Scenario
        scenario, planning_problem_set = self.dataset_obj.load_scenario(data_idx)
        planning_problem = next(iter(planning_problem_set.planning_problem_dict.values()))
        map_name = self.dataset_obj.get_scenario_name(data_idx)
        vector_map: VectorMap = commonroad_utils.extract_vectorized(
            lanelet_network=scenario.lanelet_network, country = scenario.scenario_id.country_name,
            map_name=f"{self.name}:{map_name}",
        )
        resolution = float(map_params["px_per_m"])
        maps_path = map_cache_class.get_map_paths(cache_path, self.name, map_name, resolution)[0]
        maps_path.mkdir(parents=True, exist_ok=True)
        commonroad_utils.write_vector_map_metadata(
            maps_path / f"{map_name}.metadata.json",
            commonroad_utils.commonroad_lane_metadata(scenario),
        )
        commonroad_utils.write_goal_metadata(
            cache_path,
            self.name,
            map_name,
            planning_problem,
            float(scenario.dt),
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
