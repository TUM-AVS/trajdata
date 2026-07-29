from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import dill
from nuplan.common.maps.nuplan_map import map_factory
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap
from tqdm import tqdm

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures.agent import (
    Agent,
    AgentMetadata,
    AgentType,
    FixedExtent,
    VariableExtent,
)
from trajdata.data_structures.environment import EnvMetadata
from trajdata.data_structures.scene_metadata import Scene, SceneMetadata
from trajdata.data_structures.scene_tag import SceneTag
from trajdata.dataset_specific.nuplan import nuplan_utils
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import NuPlanSceneRecord
from trajdata.maps.vec_map import VectorMap
from trajdata.maps.vec_map_elements import MapElementType
from trajdata.utils import map_utils
from trajdata.utils import arr_utils


class NuplanDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        all_log_splits: Dict[str, List[str]] = nuplan_utils.create_splits_logs()

        nup_log_splits: Dict[str, List[str]]
        if env_name.startswith("nuplan_mini"):
            nup_log_splits = {
                k: all_log_splits[k[5:]]
                for k in ["mini_train", "mini_val", "mini_test"]
            }

            # nuScenes possibilities are the Cartesian product of these
            dataset_parts: List[Tuple[str, ...]] = [
                ("mini_train", "mini_val", "mini_test"),
                nuplan_utils.NUPLAN_LOCATIONS,
            ]
        elif env_name.startswith("nuplan"):
            split_str = env_name.split("_")[-1]
            nup_log_splits = {split_str: all_log_splits[split_str]}

            # nuScenes possibilities are the Cartesian product of these
            dataset_parts: List[Tuple[str, ...]] = [
                (split_str,),
                nuplan_utils.NUPLAN_LOCATIONS,
            ]
        else:
            raise ValueError(f"Unknown nuPlan environment name: {env_name}")

        # Inverting the dict from above, associating every log with its data split.
        nup_log_split_map: Dict[str, str] = {
            v_elem: k for k, v in nup_log_splits.items() for v_elem in v
        }

        requested_locations = self.dataset_options.get("map_locations")
        if requested_locations is not None:
            requested_locations = tuple(str(value) for value in requested_locations)
            unknown = set(requested_locations) - set(nuplan_utils.NUPLAN_LOCATIONS)
            if unknown:
                raise ValueError(f"Unknown nuPlan map locations: {sorted(unknown)}")
        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=nuplan_utils.NUPLAN_DT,
            parts=dataset_parts,
            scene_split_map=nup_log_split_map,
            # The location names should match the map names used in
            # the unified data cache.
            map_locations=(
                nuplan_utils.NUPLAN_LOCATIONS
                if requested_locations is None
                else requested_locations
            ),
        )

    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)

        configured_subfolder = self.dataset_options.get("split_subfolder")
        if configured_subfolder is not None:
            subfolder = str(configured_subfolder)
        elif self.name.startswith("nuplan_mini"):
            subfolder = "mini"
        elif self.name.startswith("nuplan"):
            subfolder = "trainval"

        self.dataset_obj = nuplan_utils.NuPlanObject(self.metadata.data_dir, subfolder)

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[NuPlanSceneRecord] = list()

        default_split = "mini_train" if "mini" in self.metadata.name else "train"

        scenes_list: List[SceneMetadata] = list()
        for idx, scene_record in enumerate(self.dataset_obj.scenes):
            scene_name: str = scene_record["name"]
            originating_log: str = scene_name.split("=")[0]
            # scene_desc: str = scene_record["description"].lower()
            scene_location: str = scene_record["location"]
            scene_split: str = self.metadata.scene_split_map.get(
                originating_log, default_split
            )
            scene_length: int = scene_record["num_timesteps"]

            if scene_length == 1:
                # nuPlan has scenes with only a single frame of data which we
                # can't do much with in terms of prediction/planning/etc. As a
                # result, we skip it.
                # As an example, nuplan_mini scene e958b276c7a65197
                # from log 2021.06.14.19.22.11_veh-38_01480_01860.
                continue

            # Saving all scene records for later caching.
            all_scenes_list.append(
                NuPlanSceneRecord(
                    scene_name,
                    scene_location,
                    scene_length,
                    scene_split,
                    # scene_desc,
                    idx,
                )
            )

            if (
                scene_location in scene_tag
                and scene_split in scene_tag
                and scene_desc_contains is None
            ):
                # if scene_desc_contains is not None and not any(
                #     desc_query in scene_desc for desc_query in scene_desc_contains
                # ):
                #     continue

                scene_metadata = SceneMetadata(
                    env_name=self.metadata.name,
                    name=scene_name,
                    dt=self.metadata.dt,
                    raw_data_idx=idx,
                )
                scenes_list.append(scene_metadata)

        self.cache_all_scenes_list(env_cache, all_scenes_list)
        return scenes_list

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        all_scenes_list: List[NuPlanSceneRecord] = env_cache.load_env_scenes_list(
            self.name
        )

        scenes_list: List[SceneMetadata] = list()
        for scene_record in all_scenes_list:
            (
                scene_name,
                scene_location,
                scene_length,
                scene_split,
                # scene_desc,
                data_idx,
            ) = scene_record

            if (
                scene_location in scene_tag
                and scene_split in scene_tag
                and scene_desc_contains is None
            ):
                # if scene_desc_contains is not None and not any(
                #     desc_query in scene_desc for desc_query in scene_desc_contains
                # ):
                #     continue

                scene_metadata = Scene(
                    self.metadata,
                    scene_name,
                    scene_location,
                    scene_split,
                    scene_length,
                    data_idx,
                    None,  # This isn't used if everything is already cached.
                    # scene_desc,
                )
                scenes_list.append(scene_metadata)

        return scenes_list

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        _, _, _, data_idx = scene_info
        default_split = "mini_train" if "mini" in self.metadata.name else "train"

        scene_record: Dict[str, str] = self.dataset_obj.scenes[data_idx]

        scene_name: str = scene_record["name"]
        originating_log: str = scene_name.split("=")[0]
        # scene_desc: str = scene_record["description"].lower()
        scene_location: str = scene_record["location"]
        scene_split: str = self.metadata.scene_split_map.get(
            originating_log, default_split
        )
        scene_length: int = scene_record["num_timesteps"]

        return Scene(
            self.metadata,
            scene_name,
            scene_location,
            scene_split,
            scene_length,
            data_idx,
            scene_record,
            # scene_desc,
        )

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        # instantiate VectorMap from map_api if necessary
        self.dataset_obj.open_db(scene.name.split("=")[0] + ".db")

        ego_agent_info: AgentMetadata = AgentMetadata(
            name="ego",
            agent_type=AgentType.VEHICLE,
            first_timestep=0,
            last_timestep=scene.length_timesteps - 1,
            # From https://github.com/motional/nuplan-devkit/blob/761cdbd52d699560629c79ba1b10b29c18ebc068/nuplan/common/actor_state/vehicle_parameters.py#L125
            extent=FixedExtent(length=4.049 + 1.127, width=1.1485 * 2.0, height=1.777),
        )

        agent_list: List[AgentMetadata] = [ego_agent_info]
        agent_presence: List[List[AgentMetadata]] = [
            [ego_agent_info] for _ in range(scene.length_timesteps)
        ]

        all_frames: pd.DataFrame = self.dataset_obj.get_scene_frames(scene)

        ego_df = (
            all_frames[
                ["ego_x", "ego_y", "ego_z", "ego_vx", "ego_vy", "ego_ax", "ego_ay"]
            ]
            .rename(columns=lambda name: name[4:])
            .reset_index(drop=True)
        )
        ego_df["heading"] = arr_utils.quaternion_to_yaw(
            all_frames[["ego_qw", "ego_qx", "ego_qy", "ego_qz"]].values
        )
        ego_df["scene_ts"] = np.arange(len(ego_df))
        ego_df["agent_id"] = "ego"

        lpc_tokens: List[bytearray] = all_frames.index.tolist()
        agents_df: pd.DataFrame = self.dataset_obj.get_detected_agents(lpc_tokens)
        tls_df: pd.DataFrame = self.dataset_obj.get_traffic_light_status(lpc_tokens)

        self.dataset_obj.close_db()

        agents_df["scene_ts"] = agents_df["lidar_pc_token"].map(
            {lpc_token: scene_ts for scene_ts, lpc_token in enumerate(lpc_tokens)}
        )
        agents_df["agent_id"] = agents_df["track_token"].apply(lambda x: x.hex())

        # Recording agent metadata for later.
        agent_metadata_dict: Dict[str, Dict[str, Any]] = dict()
        for agent_id, agent_data in agents_df.groupby("agent_id").first().iterrows():
            if agent_id not in agent_metadata_dict:
                agent_metadata_dict[agent_id] = {
                    "type": nuplan_utils.nuplan_type_to_unified_type(
                        agent_data["category_name"]
                    ),
                    "width": agent_data["width"],
                    "length": agent_data["length"],
                    "height": agent_data["height"],
                }

        agents_df = agents_df.drop(
            columns=[
                "lidar_pc_token",
                "track_token",
                "category_name",
                "width",
                "length",
                "height",
            ],
        ).rename(columns={"yaw": "heading"})

        # Sorting the agents' combined DataFrame here.
        agents_df.set_index(["agent_id", "scene_ts"], inplace=True)
        agents_df.sort_index(inplace=True)
        agents_df.reset_index(level=1, inplace=True)

        one_detection_agents: List[str] = list()
        for agent_id in agent_metadata_dict:
            agent_metadata_entry = agent_metadata_dict[agent_id]

            agent_specific_df = agents_df.loc[agent_id]
            if len(agent_specific_df.shape) <= 1 or agent_specific_df.shape[0] <= 1:
                # Removing agents that are only observed once.
                one_detection_agents.append(agent_id)
                continue

            first_timestep: int = agent_specific_df.iat[0, 0].item()
            last_timestep: int = agent_specific_df.iat[-1, 0].item()
            agent_info: AgentMetadata = AgentMetadata(
                name=agent_id,
                agent_type=agent_metadata_entry["type"],
                first_timestep=first_timestep,
                last_timestep=last_timestep,
                extent=FixedExtent(
                    length=agent_metadata_entry["length"],
                    width=agent_metadata_entry["width"],
                    height=agent_metadata_entry["height"],
                ),
            )

            agent_list.append(agent_info)
            for timestep in range(
                agent_info.first_timestep, agent_info.last_timestep + 1
            ):
                agent_presence[timestep].append(agent_info)

        # Removing agents with only one detection.
        agents_df.drop(index=one_detection_agents, inplace=True)

        ### Calculating agent accelerations
        agent_ids: np.ndarray = agents_df.index.get_level_values(0).to_numpy()
        if len(agent_ids) > 0:
            agents_df[["ax", "ay"]] = (
                arr_utils.agent_aware_diff(
                    agents_df[["vx", "vy"]].to_numpy(), agent_ids
                )
                / nuplan_utils.NUPLAN_DT
            )
        else:
            agents_df[["ax", "ay"]] = agents_df[["vx", "vy"]]

        # for agent_id, frames in agents_df.groupby("agent_id")["scene_ts"]:
        #     if frames.shape[0] <= 1:
        #         raise ValueError("nuPlan can have one-detection agents :(")

        #     start_frame: int = frames.iat[0].item()
        #     last_frame: int = frames.iat[-1].item()

        #     if frames.shape[0] < last_frame - start_frame + 1:
        #         raise ValueError("nuPlan indeed can have missing frames :(")

        overall_agents_df = pd.concat([ego_df, agents_df.reset_index()]).set_index(
            ["agent_id", "scene_ts"]
        )
        metadata_path = cache_path / scene.env_name / "maps" / f"{scene.location}.metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"nuPlan map metadata is missing at {metadata_path}; map caching must precede scene caching."
            )
        map_metadata = __import__("json").loads(metadata_path.read_text())
        origins = {
            tuple(value["coordinate_origin_xy"])
            for value in map_metadata.values()
            if value.get("coordinate_frame") == "map_local_v1"
        }
        if len(origins) != 1:
            raise RuntimeError(f"nuPlan map metadata has no unique local origin: {metadata_path}")
        origin_xy = np.asarray(next(iter(origins)), dtype=float)
        overall_agents_df["x"] = overall_agents_df["x"].astype(float) - origin_xy[0]
        overall_agents_df["y"] = overall_agents_df["y"].astype(float) - origin_xy[1]
        cache_class.save_agent_data(overall_agents_df, cache_path, scene)

        goal_area = self.dataset_options.get("goal_area")
        if goal_area is not None:
            if "ego" not in overall_agents_df.index.get_level_values("agent_id"):
                raise RuntimeError("nuPlan scene cache has no ego trajectory for goal creation.")
            final_ego = overall_agents_df.xs("ego", level="agent_id").iloc[-1]
            scene_cache_dir = cache_class.scene_cache_dir(cache_path, scene.env_name, scene.name)
            vector_path = cache_path / scene.env_name / "maps" / f"{scene.location}.pb"
            if not vector_path.is_file():
                raise FileNotFoundError(f"nuPlan vector map is missing at {vector_path}")
            vector_map = VectorMap.from_proto(map_utils.load_vector_map(vector_path))
            position_geometry = nuplan_utils.create_goal_geometry(
                vector_map,
                {
                    "x": float(final_ego["x"]),
                    "y": float(final_ego["y"]),
                    "heading": float(final_ego["heading"]),
                },
                float(goal_area["area_length_m"]),
            )
            nuplan_utils.write_goal_metadata(
                scene_cache_dir,
                {
                    "x": float(final_ego["x"]),
                    "y": float(final_ego["y"]),
                    "heading": float(final_ego["heading"]),
                },
                goal_area,
                int(float(goal_area["minimum_progress_ratio"]) * scene.length_timesteps),
                position_geometry,
            )

        # similar process to clean up and traffic light data
        tls_df["scene_ts"] = tls_df["lidar_pc_token"].map(
            {lpc_token: scene_ts for scene_ts, lpc_token in enumerate(lpc_tokens)}
        )
        tls_df = tls_df.drop(columns=["lidar_pc_token"]).set_index(
            ["lane_id", "scene_ts"]
        )

        cache_class.save_traffic_light_data(tls_df, cache_path, scene)

        return agent_list, agent_presence

    def cache_map(
        self,
        map_name: str,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        map_root = self.dataset_options.get("map_root")
        if map_root is None:
            map_root = self.metadata.data_dir.parent / "maps"
        nuplan_map: NuPlanMap = map_factory.get_maps_api(
            map_root=str(map_root),
            map_version=nuplan_utils.NUPLAN_MAP_VERSION,
            map_name=nuplan_utils.NUPLAN_FULL_MAP_NAME_DICT[map_name],
        )

        # Loading all layer geometries.
        nuplan_map.initialize_all_layers()

        # This df has the normal lane_connectors with additional boundary information,
        # which we want to use, however the default index is not the lane_connector_fid,
        # although it is a 1:1 mapping so we instead create another index with the
        # lane_connector_fids as the key and the resulting integer indices as the value.
        lane_connector_fids: pd.Series = nuplan_map._vector_map[
            "gen_lane_connectors_scaled_width_polygons"
        ]["lane_connector_fid"]
        lane_connector_idxs: pd.Series = pd.Series(
            index=lane_connector_fids, data=range(len(lane_connector_fids))
        )

        vector_map = VectorMap(map_id=f"{self.name}:{map_name}")
        nuplan_utils.populate_vector_map(vector_map, nuplan_map, lane_connector_idxs)
        vector_map.lanes = list(vector_map.elements[MapElementType.ROAD_LANE].values())
        if not vector_map.lanes:
            raise RuntimeError(f"nuPlan map {map_name!r} conversion produced no road lanes.")
        origin_xy = np.floor(
            np.min(np.concatenate([lane.center.xy for lane in vector_map.lanes]), axis=0) / 1000.0
        ) * 1000.0
        vector_map.extent[[0, 1]] -= origin_xy
        vector_map.extent[[3, 4]] -= origin_xy
        for element in vector_map.iter_elems():
            polylines = (
                [element.center, element.left_edge, element.right_edge]
                if element.elem_type == MapElementType.ROAD_LANE
                else [element.exterior_polygon, *element.interior_holes]
                if element.elem_type == MapElementType.ROAD_AREA
                else [element.polygon]
            )
            for polyline in polylines:
                if polyline is not None:
                    polyline.points[..., :2] -= origin_xy
        metadata = {}
        for lane in vector_map.lanes:
            map_object = nuplan_map.get_map_object(str(lane.id), SemanticMapLayer.LANE)
            object_type = "lane"
            if map_object is None:
                map_object = nuplan_map.get_map_object(str(lane.id), SemanticMapLayer.LANE_CONNECTOR)
                object_type = "lane_connector"
            if map_object is None:
                raise RuntimeError(f"nuPlan map {map_name!r} has no lane object for {lane.id!r}.")
            metadata[str(lane.id)] = {
                "source": "nuplan",
                "coordinate_frame": "map_local_v1",
                "coordinate_origin_xy": origin_xy.tolist(),
                "speed_limit_mps": map_object.speed_limit_mps,
                "map_object_type": object_type,
                "roadblock_id": map_object.get_roadblock_id(),
                "has_traffic_lights": bool(map_object.has_traffic_lights()),
                "stop_lines": [
                    {"id": str(stop.id), "polygon": (np.asarray(stop.polygon.exterior.coords, dtype=float) - origin_xy).tolist()}
                    for stop in map_object.stop_lines
                ],
                "incoming_edge_ids": sorted(edge.id for edge in map_object.incoming_edges),
                "outgoing_edge_ids": sorted(edge.id for edge in map_object.outgoing_edges),
            }
        vector_map.compute_search_indices()
        maps_path, vector_path, kdtrees_path, rtrees_path, _, _ = map_cache_class.get_map_paths(
            cache_path, vector_map.env_name, vector_map.map_name, float(map_params["px_per_m"])
        )
        maps_path.mkdir(parents=True, exist_ok=True)
        vector_path.write_bytes(vector_map.to_proto().SerializeToString())
        with kdtrees_path.open("wb") as file:
            dill.dump(vector_map.search_kdtrees, file)
        with rtrees_path.open("wb") as file:
            dill.dump(vector_map.search_rtrees, file)
        nuplan_utils.write_vector_map_metadata(maps_path / f"{map_name}.metadata.json", metadata)

    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        """
        Stores rasterized maps to disk for later retrieval.
        """
        for map_name in tqdm(
            self.metadata.map_locations,
            desc=f"Caching {self.name} Maps at {map_params['px_per_m']:.2f} px/m",
            position=0,
        ):
            self.cache_map(map_name, cache_path, map_cache_class, map_params)
