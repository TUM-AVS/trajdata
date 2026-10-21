import warnings
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Type, Union

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures import (
    AgentMetadata,
    EnvMetadata,
    Scene,
    SceneMetadata,
    SceneTag,
)


class RawDataset:
    def __init__(
        self,
        env_name: str,
        data_dir: str,
        parallelizable: bool,
        has_maps: bool,
        dataset_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.dataset_options = {} if dataset_options is None else dict(dataset_options)
        metadata = self.compute_metadata(env_name, data_dir)

        self.metadata = metadata
        self.name = metadata.name
        self.scene_tags = metadata.scene_tags
        self.dataset_obj = None

        self.parallelizable = parallelizable
        self.has_maps = has_maps

    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        raise NotImplementedError()

    def get_matching_scene_tags(self, query: Set[str]) -> List[SceneTag]:
        return [scene_tag for scene_tag in self.scene_tags if scene_tag.contains(query)]

    def load_dataset_obj(self, verbose: bool = False) -> None:
        raise NotImplementedError()

    def del_dataset_obj(self) -> None:
        del self.dataset_obj
        self.dataset_obj = None

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        raise NotImplementedError()

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        raise NotImplementedError()

    def cache_all_scenes_list(
        self, env_cache: EnvCache, all_scenes_list: List[NamedTuple]
    ) -> None:
        env_cache.save_env_scenes_list(self.name, all_scenes_list)

    def refresh_cached_scenes(
        self,
        env_cache: EnvCache,
        cache_class: Type[SceneCache],
        verbose: bool = False,
    ) -> bool:
        """Refresh cached scene discovery metadata when this adapter supports it.

        Returning ``True`` tells :class:`UnifiedDataset` that the set of
        scenes may have changed, so its derived data index must be rebuilt.
        Adapters that cannot cheaply or safely rediscover their source data
        retain the existing cache-only behavior.
        """
        warnings.warn(
            f"Incremental scene refresh is not implemented for dataset "
            f"{self.name!r}; using its existing cached scene list.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False

    def refresh_source_fingerprints(
        self,
        env_cache: EnvCache,
        cache_class: Type[SceneCache],
        current_fingerprints: Dict[str, str],
    ) -> Set[str]:
        """Invalidate only caches whose adapter-reported source changed.

        The comparison and invalidation policy is shared by all adapters.
        Concrete adapters only provide their ``scene_name -> fingerprint`` map.
        An environment without a previous manifest establishes a baseline and
        therefore does not invalidate its existing cache on that first run.
        """
        previous_fingerprints = env_cache.load_source_fingerprints(self.name)
        if previous_fingerprints is None:
            env_cache.save_source_fingerprints(self.name, current_fingerprints)
            return set()

        modified_scenes = {
            scene_name
            for scene_name, fingerprint in current_fingerprints.items()
            if previous_fingerprints.get(scene_name) not in {None, fingerprint}
        }
        for scene_name in modified_scenes:
            env_cache.remove_scene_cache(self.name, scene_name)
            cache_class.remove_map_cache(env_cache.path, self.name, scene_name)

        env_cache.save_source_fingerprints(self.name, current_fingerprints)
        return modified_scenes

    def get_matching_scenes(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
        rebuild_cache: bool,
    ) -> Union[List[Scene], List[SceneMetadata]]:
        if self.dataset_obj is None and not rebuild_cache:
            return self._get_matching_scenes_from_cache(
                scene_tag, scene_desc_contains, env_cache
            )
        else:
            return self._get_matching_scenes_from_obj(
                scene_tag, scene_desc_contains, env_cache
            )

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        raise NotImplementedError()

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        """
        Get frame-level information from source dataset, caching it
        to cache_path.

        Always called after cache_maps, can load map if needed
        to associate map information to positions.
        """
        raise NotImplementedError()

    def cache_maps_for_scenes(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
        scenes: List[Union[Scene, SceneMetadata]],
        *,
        rebuild: bool = False,
    ) -> None:
        """Compatibility for adapters with environment-wide map caching.

        Scene-specific adapters override this to cache only selected maps.
        """
        maps_dir = map_cache_class.get_maps_path(cache_path, self.name)
        if rebuild and maps_dir.exists():
            import shutil
            shutil.rmtree(maps_dir)
        self.cache_maps(cache_path, map_cache_class, map_params)

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
        raise NotImplementedError()
