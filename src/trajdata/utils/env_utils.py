from typing import Any, Dict, List, Optional

from trajdata.dataset_specific import RawDataset


def get_raw_dataset(
    dataset_name: str,
    data_dir: str,
    dataset_options: Optional[Dict[str, Any]] = None,
) -> RawDataset:
    if "nusc" in dataset_name:
        from trajdata.dataset_specific.nusc import NuscDataset

        return NuscDataset(dataset_name, data_dir, parallelizable=False, has_maps=True)

    if "vod" in dataset_name:
        from trajdata.dataset_specific.vod import VODDataset

        return VODDataset(dataset_name, data_dir, parallelizable=True, has_maps=True)

    if "lyft" in dataset_name:
        from trajdata.dataset_specific.lyft import LyftDataset

        return LyftDataset(dataset_name, data_dir, parallelizable=True, has_maps=True)

    if "eupeds" in dataset_name:
        from trajdata.dataset_specific.eth_ucy_peds import EUPedsDataset

        return EUPedsDataset(
            dataset_name, data_dir, parallelizable=True, has_maps=False
        )

    if "sdd" in dataset_name:
        from trajdata.dataset_specific.sdd_peds import SDDPedsDataset

        return SDDPedsDataset(
            dataset_name, data_dir, parallelizable=True, has_maps=False
        )

    if "nuplan" in dataset_name:
        from trajdata.dataset_specific.nuplan import NuplanDataset

        return NuplanDataset(dataset_name, data_dir, parallelizable=True, has_maps=True, dataset_options=dataset_options)

    if "waymo" in dataset_name:
        from trajdata.dataset_specific.waymo import WaymoDataset

        return WaymoDataset(dataset_name, data_dir, parallelizable=True, has_maps=True)

    if "interaction" in dataset_name:
        from trajdata.dataset_specific.interaction import InteractionDataset

        return InteractionDataset(
            dataset_name, data_dir, parallelizable=True, has_maps=True
        )

    if "av2" in dataset_name:
        from trajdata.dataset_specific.argoverse2 import Av2Dataset

        return Av2Dataset(dataset_name, data_dir, parallelizable=True, has_maps=True)
        
    if "commonroad" in dataset_name:
        from trajdata.dataset_specific.commonroad import CommonRoadDataset

        return CommonRoadDataset(dataset_name, data_dir, parallelizable=False, has_maps=True, dataset_options=dataset_options)
    raise ValueError(f"Dataset with name '{dataset_name}' is not supported")


def get_raw_datasets(
    data_dirs: Dict[str, str],
    dataset_options: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[RawDataset]:
    raw_datasets: List[RawDataset] = list()

    for dataset_name, data_dir in data_dirs.items():
        options = None if dataset_options is None else dataset_options.get(dataset_name)
        raw_datasets.append(get_raw_dataset(dataset_name, data_dir, options))

    return raw_datasets
