import pandas as pd

from trajdata.data_structures.batch import AgentBatch, SceneBatch
from trajdata.data_structures.scene_metadata import Scene


class Augmentation:
    def __init__(self) -> None:
        raise NotImplementedError()


class DatasetAugmentation(Augmentation):
    def apply(
        self, scene_data_df: pd.DataFrame, scene: Scene | None = None
    ) -> pd.DataFrame | None:
        """Optionally update scene metadata and return a replacement DataFrame.

        Existing in-place augmentations may omit scene and return None.
        """
        raise NotImplementedError()


class BatchAugmentation(Augmentation):
    def apply_agent(self, agent_batch: AgentBatch) -> None:
        raise NotImplementedError()

    def apply_scene(self, scene_batch: SceneBatch) -> None:
        raise NotImplementedError()
