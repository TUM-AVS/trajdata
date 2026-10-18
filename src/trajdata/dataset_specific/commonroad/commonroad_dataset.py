import hashlib
from pathlib import Path
from typing import cast, Any, Dict, List, Optional, Tuple, Type, Union
from collections import defaultdict
from functools import partial
import warnings

import numpy as np
import pandas as pd
import tqdm
from scipy.interpolate import interp1d, PchipInterpolator
from trajdata.dataset_specific.commonroad import commonroad_utils

# Try to import CommonRoad reader; if not present, users will need to install commonroad-io
try:
    from commonroad.scenario.scenario import Scenario
    # from commonroad.scenario.obstacle import Obstacle, StaticObstacle, DynamicObstacle, EnvironmentObstacle, PhantomObstacle, Prediction, TrajectoryPrediction, SetBasedPrediction
    # from commonroad.scenario.trajectory import Trajectory
    # from commonroad.scenario.state import State, KSState, InitialState, PMState, KSTState, STState, STDState, MBState, LongitudinalState, LateralState, InputState, PMInputState, CustomState, ExtendedPMState
    # from commonroad.common.file_reader  import CommonRoadFileReader
    # from commonroad.planning.planning_problem import PlanningProblem
    # import commonroad_velocity_planner.fast_api  as crvp_fast_api
    from vehiclemodels.vehicle_parameters import setup_vehicle_parameters
    from commonroad.geometry import shape

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

from trajdata.dataset_specific.commonroad.commonroad_utils import CommonRoadScenarioHandler, translate_agent_type #, pad_and_interpolate_array # , check_obstacle_validity, check_state_validity
from trajdata.dataset_specific.scene_records import CommonRoadSceneRecord

def const_lambda(const_val: Any) -> Any:
    return const_val

class CommonRoadDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str)-> EnvMetadata:
        dataset_parts = [(env_name,)]                   
        scene_split_map = defaultdict(partial(const_lambda, const_val=env_name))     
        map_locations = [f.stem for f in Path(data_dir).glob("*.xml") ]
        map_locations = cast(Tuple[str], tuple(map_locations))
        """Get timeStepSize for all scenario files"""
        return EnvMetadata(
            name = env_name,
            data_dir=data_dir,
            dt = commonroad_utils.COMMONROAD_DT,
            parts=dataset_parts,
            scene_split_map=scene_split_map,
            map_locations=map_locations,
        )

    def load_dataset_obj(self, verbose = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset ", flush = True)

        self.dataset_obj = CommonRoadScenarioHandler(data_dir = self.metadata.data_dir)      

        if verbose:
            print(f"Found {self.dataset_obj.num_scenarios} scenarios", flush=True)

    def _all_scene_records(self) -> List[CommonRoadSceneRecord]:
        """Inventory filenames without parsing scenario XMLs for their lengths."""
        if self.dataset_obj is None:
            raise RuntimeError("CommonRoad source inventory requires a loaded dataset.")

        return [
            CommonRoadSceneRecord(
                name=self.dataset_obj.get_scenario_name(idx),
                length=None,
                data_idx=idx,
            )
            for idx in range(self.dataset_obj.num_scenarios)
        ]

    def _source_fingerprints(self) -> Dict[str, str]:
        """Return SHA-256 fingerprints for the XML files currently discovered."""
        if self.dataset_obj is None:
            raise RuntimeError("CommonRoad source fingerprints require a loaded dataset.")

        fingerprints: Dict[str, str] = {}
        for idx in range(self.dataset_obj.num_scenarios):
            digest = hashlib.sha256()
            with self.dataset_obj.get_scenario_path(idx).open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            fingerprints[self.dataset_obj.get_scenario_name(idx)] = digest.hexdigest()
        return fingerprints

    def refresh_cached_scenes(
        self,
        env_cache: EnvCache,
        cache_class: Type[SceneCache],
        verbose: bool = False,
    ) -> bool:
        """Refresh added and modified CommonRoad XML sources incrementally.

        RawDataset owns the shared fingerprint comparison and single-scene
        cache invalidation; this adapter supplies CommonRoad source hashes.
        """
        self.load_dataset_obj(verbose=verbose)
        current_records = self._all_scene_records()
        modified_scenes = self.refresh_source_fingerprints(
            env_cache, cache_class, self._source_fingerprints()
        )
        cached_records = env_cache.load_env_scenes_list(self.name)

        if current_records == cached_records and not modified_scenes:
            self.del_dataset_obj()
            return False

        env_cache.save_env_scenes_list(self.name, current_records)
        self.del_dataset_obj()
        return True

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,    #Set of Labels to search in its "tags" for fast categorical filtering, e.g {"intersection", "daytime"}, etc
        scene_desc_contains: Optional[List[str]],  
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        
        all_scenes_list = self._all_scene_records()
        scenes_list: List[SceneMetadata] = list()

        for scene_record in all_scenes_list:
            scene_name = scene_record.name
            scene_split: str = self.metadata.scene_split_map[scene_name]       

            matches_split = scene_split in scene_tag  # Does the split match what user wants?
            matches_description = True 
            
            if scene_desc_contains is not None:
                matches_description = False
                # Check if any of the required strings are in the scene name
                for required_text in scene_desc_contains:
                    if required_text.lower() in scene_name.lower():
                        matches_description = True
                        break
            if matches_split and matches_description:
                scene_metadata = SceneMetadata(
                    env_name = self.metadata.name,
                    name = scene_name,
                    dt = self.metadata.dt,                                      
                    raw_data_idx = scene_record.data_idx
                )
                scenes_list.append(scene_metadata)

        self.cache_all_scenes_list(env_cache, all_scenes_list)  
        env_cache.save_source_fingerprints(self.name, self._source_fingerprints())
        return scenes_list
            
    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        _, name, _, data_idx = scene_info
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
        records: List[CommonRoadSceneRecord] = env_cache.load_env_scenes_list(self.name)
        scenes: List[Scene] = []
        for record in records:
            matches_description = scene_desc_contains is None or any(
                text.lower() in record.name.lower() for text in scene_desc_contains
            )
            if self.metadata.scene_split_map[record.name] not in scene_tag or not matches_description:
                continue
            # Inventory lengths may be unknown. Cached Scene objects already
            # contain the real length; only uncached selected scenes need parsing.
            if env_cache.scene_is_cached(self.name, record.name, self.metadata.dt):
                scene = env_cache.load_scene(self.name, record.name, self.metadata.dt)
                scene.raw_data_idx = record.data_idx
            else:
                if self.dataset_obj is None:
                    self.load_dataset_obj()
                scene = self.get_scene(SceneMetadata(
                    self.name, record.name, self.metadata.dt, record.data_idx,
                ))
            scenes.append(scene)
        return scenes

 


    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        """Cache CommonRoad states as records, padded to the scene timeline."""
        scenario, planning_problem_set = self.dataset_obj.load_scenario(
            scene.raw_data_idx
        )
        planning_problem = next(
            iter(planning_problem_set.planning_problem_dict.values())
        )

        agent_list: List[AgentMetadata] = []
        agent_presence: List[List[AgentMetadata]] = [
            [] for _ in range(scene.length_timesteps + 1)
        ]
        df_records: List[Dict[str, Any]] = []
        # dynamic obstacles
        for obstacle in scenario.dynamic_obstacles:
            first_timestep = obstacle.initial_state.time_step
            last_timestep = min(obstacle.prediction.final_time_step, scene.length_timesteps)
            if not 0 <= first_timestep <= last_timestep <= scene.length_timesteps:
                raise ValueError(
                    f"Obstacle {obstacle.obstacle_id} has lifetime "
                    f"[{first_timestep}, {last_timestep}] outside scene "
                    f"[0, {scene.length_timesteps}]."
                )

            agent_info = AgentMetadata(
                name=str(obstacle.obstacle_id),
                agent_type=translate_agent_type(obstacle.obstacle_type),
                first_timestep=first_timestep,
                last_timestep=last_timestep,
                extent=FixedExtent(
                    obstacle.obstacle_shape.length,
                    obstacle.obstacle_shape.width,
                    0.0,
                ),
            )
            agent_list.append(agent_info)
            #TODO : ax, ay to large at first timestep -> does lowvelaugmentation solve this issue?
            for scene_ts in range(first_timestep, last_timestep + 1):
                state = obstacle.state_at_time(scene_ts)

                agent_presence[scene_ts].append(agent_info)
                theta =arr_utils.angle_wrap(state.orientation)
                df_records.append(
                    {
                        "agent_id": agent_info.name,
                        "scene_ts": scene_ts,
                        "x": state.position[0],
                        "y": state.position[1],
                        "z": 0.0,
                        "vx": np.cos(theta) * state.velocity,
                        "vy": np.sin(theta) * state.velocity,
                        "heading": theta,
                    }
                )
        # static obstacles
        for static_obstacle in scenario.static_obstacles:
            initial_state = static_obstacle.initial_state
            agent_info = AgentMetadata(
                name=str(static_obstacle.obstacle_id),
                agent_type=translate_agent_type(static_obstacle.obstacle_type),
                first_timestep=0,
                last_timestep=scene.length_timesteps,
                extent=FixedExtent(static_obstacle.obstacle_shape.length, static_obstacle.obstacle_shape.width, 0.0),
            )
            agent_list.append(agent_info)
            for scene_ts in range(scene.length_timesteps + 1):
                agent_presence[scene_ts].append(agent_info)
                df_records.append(
                    {
                        "agent_id": agent_info.name,
                        "scene_ts": scene_ts,
                        "x": float(initial_state.position[0]),
                        "y": float(initial_state.position[1]),
                        "z": 0.0,
                        "vx": 0.0,
                        "vy": 0.0,
                        "heading": arr_utils.angle_wrap(initial_state.orientation),
                    }
                )

        # ego agent
        ego_state = planning_problem.initial_state
        ego_first_timestep = ego_state.time_step
        if not 0 <= ego_first_timestep <= scene.length_timesteps:
            raise ValueError(
                f"Ego initial timestep {ego_first_timestep} is outside scene "
                f"[0, {scene.length_timesteps}]."
            )
        vehicle_model = setup_vehicle_parameters(2)  # dummy/default vehicle model (BMW320), replaced by used vehicle model during simulation
        ego_info = AgentMetadata(
            name="ego",
            agent_type=AgentType.VEHICLE,
            first_timestep=ego_first_timestep,
            last_timestep=scene.length_timesteps,
            extent=FixedExtent(vehicle_model.l, vehicle_model.w, 0.0),
        )
        agent_list.append(ego_info)
        for scene_ts in range(ego_first_timestep, scene.length_timesteps + 1):
            agent_presence[scene_ts].append(ego_info)
        
        # goal_state= planning_problem.goal.state_list[0]
        # goal_interval = range(goal_state.time_step.start, goal_state.time_step.end + 1) 
        # goal_position = goal_state.position.shapes[0].center
        # goal_velocity = goal_state.velocity.end
        theta = arr_utils.angle_wrap(ego_state.orientation)
        df_records.append(
            {
                "agent_id": ego_info.name,
                "scene_ts": ego_first_timestep,
                "x": ego_state.position[0],
                "y": ego_state.position[1],
                "z": 0.0,
                "vx": np.cos(theta) * ego_state.velocity,
                "vy": np.sin(theta) * ego_state.velocity,
                "heading": theta,
            }
        )
        # df_records.extend(
        #     {
        #         "agent_id": ego_info.name,
        #         "scene_ts": scene_ts,
        #         "x": goal_position[0],
        #         "y": goal_position[1],
        #         "z": 0.0,
        #         "vx": np.cos(theta) * goal_velocity,
        #         "vy": np.sin(theta) * goal_velocity,
        #         "heading": theta,
        #     }
        #     for scene_ts in goal_interval
        # )

        # Compute accelerations and cache the agent data
        agent_data = pd.DataFrame.from_records(df_records).set_index(
            ["agent_id", "scene_ts"]
        )
        agent_data.sort_index(inplace=True)
        agent_data[["ax", "ay"]] = np.nan
        for _, agent_states in agent_data.groupby(level="agent_id", sort=False):
            if len(agent_states) > 1:
                agent_data.loc[agent_states.index, ["ax", "ay"]] = (
                    arr_utils.agent_aware_diff(
                        agent_states[["vx", "vy"]].to_numpy(),
                        agent_states.index.get_level_values("agent_id").to_numpy(),
                    )
                    / scene.dt
                )

        # The ego has no previous cached velocity at its initial timestep.
        agent_data.loc[(ego_info.name, ego_first_timestep), ["ax", "ay"]] = 0.0

        full_index = pd.MultiIndex.from_product(
            [
                agent_data.index.get_level_values("agent_id").unique(),
                range(scene.length_timesteps + 1),
            ],
            names=["agent_id", "scene_ts"],
        )
        agent_data = agent_data.reindex(full_index)
        cache_class.save_agent_data(
            agent_data[["x", "y", "z", "vx", "vy", "ax", "ay", "heading"]],
            cache_path,
            scene,
        )

        if scenario.lanelet_network.traffic_lights:
            raise NotImplementedError(
                "CommonRoad traffic-light caching is not implemented."
            )

        return agent_list, agent_presence

    def cache_map(
        self,
        data_idx: int,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ):

        scenario, planning_problem_set = self.dataset_obj.load_scenario(data_idx)
        planning_problem = next(iter(planning_problem_set.planning_problem_dict.values()))
        scenario_name = self.dataset_obj.get_scenario_name(data_idx)
        vector_map: VectorMap = commonroad_utils.extract_vectorized(scenario.lanelet_network, 
        map_name=f"{self.name}:{scenario_name}") #, country = scenario.scenario_id.country_name,
            # ",
        # )
        metadata_df = commonroad_utils.cache_map_metadata(scenario, planning_problem, vector_map)
        map_cache_class.finalize_and_cache_map(cache_path, vector_map, map_params)
        # resolution = float(map_params["px_per_m"])
        # cache_dir = map_cache_class.scene_cache_dir(cache_path, self.name, scenario_name)
        # cache_dir.mkdir(parents=True, exist_ok=True)

        map_cache_class.save_map_metadata(
            metadata_df, cache_path, self.name, scenario_name,
        )
        # commonroad_utils.write_vector_map_metadata(
        #     maps_path / f"{scenario_name}.metadata.json",
        #     commonroad_utils.cache_map_metadata(scenario),
        # )
        # commonroad_utils.write_goal_metadata(
        #     cache_path,
        #     self.name,
        #     scenario_name,
        #     planning_problem,
        #     float(scenario.dt),
        # )
        

    def cache_maps_for_scenes(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
        scenes: List[Union[Scene, SceneMetadata]],
        *,
        rebuild: bool = False,
    ) -> None:
        """Cache selected maps; rebuild clears the environment maps directory."""
        selected_names = {scene.name for scene in scenes}
        indices_by_name = {
            self.dataset_obj.get_scenario_name(idx): idx
            for idx in range(self.dataset_obj.num_scenarios)
        }
        indices = [indices_by_name[name] for name in sorted(selected_names)]
        self.cache_maps(cache_path, map_cache_class, map_params,
                        scene_indices=indices, rebuild=rebuild)

    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
        *,
        scene_indices: Optional[List[int]] = None,
        rebuild: bool = False,
    ) -> None:
        """
        Get static, scene-level info from the source dataset, caching it
        to cache_path. (Primarily this is info needed to construct VectorMap)
        
        Resolution is in pixels per meter.
        """

        maps_dir = map_cache_class.get_maps_path(cache_path, self.name)
        if rebuild and maps_dir.exists():
            import shutil
            shutil.rmtree(maps_dir)

        resolution = float(map_params["px_per_m"])
        missing_map_indices = [
            idx
            for idx in (range(self.dataset_obj.num_scenarios) if scene_indices is None else scene_indices)
            if not map_cache_class.is_map_cached(
                cache_path,
                self.name,
                self.dataset_obj.get_scenario_name(idx),
                resolution,
            )
        ]

        num_workers: int = map_params.get("num_workers", 0)
        if num_workers > 1:
            parallel_apply(
                partial(
                    self.cache_map,
                    cache_path=cache_path,
                    map_cache_class=map_cache_class,
                    map_params=map_params,
                ),
                missing_map_indices,
                num_workers=num_workers,            #NOTE : Can add a line that updates self.metadata's locations list (present in class EnvMetadata.map_locations). Then we can access list of all maps in an environment from dataset.envs[i].metadata.map_locations (easy iteration over the list)
            )
        else:
            for i in tqdm.tqdm(missing_map_indices):
                self.cache_map(i, cache_path, map_cache_class, map_params)
