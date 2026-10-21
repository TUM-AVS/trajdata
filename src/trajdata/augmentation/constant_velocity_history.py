"""Constant-velocity pre-history added once when a scene cache is loaded."""

from decimal import Decimal
from math import floor, isfinite

import numpy as np
import pandas as pd

from trajdata.augmentation.augmentation import DatasetAugmentation
from trajdata.data_structures.scene_metadata import Scene


class ConstantVelocityHistory(DatasetAugmentation):
    """Pad the beginning of a recording without shifting its real timestamps.

    Only agents present at timestep zero receive synthetic negative-timestep
    history. Later arrivals retain their real lifetimes and presence.
    """

    def __init__(self, max_history_sec: float) -> None:
        self.max_history_sec = float(max_history_sec)

    def apply(self, scene_data_df: pd.DataFrame, scene: Scene | None = None) -> pd.DataFrame:
        steps = floor(Decimal(str(self.max_history_sec)) / Decimal(str(scene.dt)))
        if steps == 0:
            return scene_data_df

        prefixes = []
        padded_agents = []
        for agent in scene.agents:
            if agent.first_timestep > 0:
                # Later arrivals have no recording-start history to pad.
                continue
            if agent.first_timestep != 0:
                raise NotImplementedError("ConstantVelocityHistory only supports agents present at timestep 0.")
            initial = scene_data_df.loc[(agent.name, 0)]
            timesteps = np.arange(-steps, 0)
            prefix = pd.DataFrame(
                np.repeat(initial.to_numpy()[None, :], steps, axis=0),
                columns=scene_data_df.columns,
                index=pd.MultiIndex.from_product([[agent.name], timesteps], names=scene_data_df.index.names),
            )
            prefix["x"] = initial["x"] + timesteps * scene.dt * initial["vx"]
            prefix["y"] = initial["y"] + timesteps * scene.dt * initial["vy"]
            prefix[["ax", "ay"]] = 0.0
            prefixes.append(prefix)
            padded_agents.append(agent)

        if not prefixes:
            return scene_data_df
        padded = pd.concat([scene_data_df, *prefixes], verify_integrity=True).sort_index()
        for agent in padded_agents:
            agent.first_timestep = -steps
        # Presence at non-negative timesteps, last_timestep and scene length
        # stay unchanged: only historical context precedes the recording.
        return padded
