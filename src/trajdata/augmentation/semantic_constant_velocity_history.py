"""Road-following synthetic history for trajdata scene caches."""

from pathlib import Path

import numpy as np
import pandas as pd
from trajdata.augmentation.constant_velocity_history import ConstantVelocityHistory
from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.scene_metadata import Scene
from trajdata.maps import MapAPI, VectorMap
from trajdata.maps.vec_map_elements import Polyline
from trajdata.utils import arr_utils


class SemanticConstantVelocityHistory(ConstantVelocityHistory):
    """Extend vehicle histories backward along lanes at constant speed.

    The original augmentation supplies negative timestamps, stationary and
    non-road-agent histories, unmatched road-agent histories, and metadata updates.
    Moving road agents with matching lanes follow
    predecessor centerlines, extending the first lane's tangent at map ends.
    At merges, the predecessor with the smallest heading change is selected;
    lane IDs break ties. The initial position offset is retained.
    Synthetic velocities rotate at constant magnitude; their derivatives supply
    acceleration on curves. Real states, including timestep zero, stay unchanged.
    """

    ROAD_TYPES = {
        AgentType.VEHICLE,
        AgentType.TRUCK,
        AgentType.BUS,
        AgentType.BICYCLE,
        AgentType.MOTORCYCLE,
    }

    def __init__(self, max_history_sec: float, cache_path: str):
        super().__init__(max_history_sec)
        self.a_map_api = MapAPI(Path(cache_path), keep_in_memory=True)

    def apply(self, scene_data_df: pd.DataFrame, scene: Scene | None = None) -> pd.DataFrame:
        padded = super().apply(scene_data_df, scene)
        if padded is scene_data_df:
            return padded

        vec_map = None
        for agent in scene.agents:
            if agent.first_timestep >= 0:
                # Only agents padded by the parent augmentation need lane history.
                continue
            initial = scene_data_df.loc[(agent.name, 0)]
            speed = np.linalg.norm(initial[["vx", "vy"]].to_numpy())
            if agent.type not in self.ROAD_TYPES or speed == 0.0:
                continue
            if vec_map is None:
                vec_map = self.a_map_api.get_map(f"{scene.env_name}:{scene.location}")
            steps = -agent.first_timestep
            lane_history = self._lane_history(vec_map, initial, steps, scene.dt)
            if lane_history is None:
                # Keep the ordinary constant-velocity padding created above.
                continue
            positions, headings = lane_history
            angles = arr_utils.angle_wrap(headings - headings[-1])
            velocities = arr_utils.transform_coords_2d_np(
                np.broadcast_to(initial[["vx", "vy"]].to_numpy(), (steps + 1, 2)),
                angle=angles,
            )
            accelerations = arr_utils.agent_aware_diff(velocities, np.repeat(agent.name, steps + 1)) / scene.dt
            index = pd.MultiIndex.from_product([[agent.name], range(-steps, 0)])
            padded.loc[index, ["x", "y"]] = positions[:-1]
            padded.loc[index, ["vx", "vy"]] = velocities[:-1]
            padded.loc[index, ["ax", "ay"]] = accelerations[:-1]
            padded.loc[index, "heading"] = arr_utils.angle_wrap(initial["heading"] + angles[:-1])
        return padded

    @staticmethod
    def _lane_history(
        vec_map: VectorMap,
        initial: pd.Series,
        steps: int,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Sample lane-following history, or return None when no lane matches."""
        position = initial[["x", "y", "z", "heading"]].to_numpy()
        candidates = vec_map.get_current_lane(position)
        if not candidates:
            return None
        projected = [(lane, lane.center.project_onto(position[None])[0]) for lane in candidates]
        lane, projection = min(
            projected,
            key=lambda item: (
                np.linalg.norm(item[1][:3] - position[:3]),
                abs(arr_utils.angle_wrap(item[1][3] - position[3])),
                item[0].id,
            ),
        )
        lengths = np.linalg.norm(np.diff(lane.center.xyz, axis=0), axis=-1)
        lane_length = lengths.sum()
        distances = np.linalg.norm(lane.center.xyz - projection[:3], axis=-1)
        segment = int(np.argmin(distances[:-1] + distances[1:] - lengths))
        progress = lengths[:segment].sum() + distances[segment]
        required = np.linalg.norm(initial[["vx", "vy"]].to_numpy()) * steps * dt

        path = [lane]
        available = progress
        while available < required and path[0].prev_lanes:
            predecessors = [vec_map.get_road_lane(lane_id) for lane_id in path[0].prev_lanes]
            predecessor = min(
                predecessors,
                key=lambda candidate: (
                    abs(arr_utils.angle_wrap(candidate.center.h[-1] - path[0].center.h[0])),
                    candidate.id,
                ),
            )
            length = np.linalg.norm(np.diff(predecessor.center.xyz, axis=0), axis=-1).sum()
            if not np.isfinite(length) or length <= 0.0:
                raise ValueError(f"Predecessor lane {predecessor.id} has no valid length.")
            path.insert(0, predecessor)
            available += length

        # Use the same centerline trimming and Polyline interpolation as
        # SemanticCVPrediction, traversing predecessors instead of successors.
        centers = np.concatenate([part.center.xyzh for part in path]).copy()
        lengths = np.linalg.norm(np.diff(centers[:, :3], axis=0), axis=-1)
        keep = np.r_[True, lengths > 1e-9]
        centers = centers[keep]
        cumulative = np.r_[0.0, np.cumsum(lengths[keep[1:]])]
        centers[:, 3] = np.unwrap(centers[:, 3])
        end = cumulative[-1] - (lane_length - progress)
        start = end - required
        endpoints = np.array(
            [[np.interp(distance, cumulative, centers[:, column]) for column in range(4)] for distance in (start, end)]
        )
        if start < 0.0:
            endpoints[0, :2] += start * arr_utils.rotation_matrix(centers[0, 3])[:, 0]
        interior = centers[(cumulative > start) & (cumulative < end)]
        points = np.vstack((endpoints[0], interior, endpoints[1]))
        if len(points) == steps + 1:
            # trajdata otherwise skips resampling an already matching count.
            points = np.insert(points, 1, (points[0] + points[1]) / 2, axis=0)
        sampled = Polyline(points).interpolate(num_pts=steps + 1)
        return sampled.xy + position[:2] - projection[:2], sampled.h
