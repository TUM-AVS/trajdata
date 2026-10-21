from collections import defaultdict
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.data_structures import Scene
from trajdata.data_structures.state import StateArray, StateTensor
from trajdata.visualization import plot_agent_batch, plot_agent_batch_interactive
from trajdata.visualization.interactive_animation import (
    InteractiveAnimation,
    animate_agent_batch_interactive,
)
from trajdata import MapAPI, VectorMap
from trajdata.maps.vec_map_elements import MapElementType
from trajdata.utils import map_utils

from trajdata.maps.vec_map_elements import (
    MapElement,
    MapElementType,
    PedCrosswalk,
    PedWalkway,
    Polyline,
    RoadArea,
    RoadLane,
)


def main():
    dataset = UnifiedDataset(
        desired_data=["commonroad"],
        centric="agent",
        desired_dt=0.1,
        # history_sec=(3.2, 3.2),
        # future_sec=(4.8, 4.8),
        state_format="x,y,z,xd,yd,xdd,ydd,h",
        agent_interaction_distances=defaultdict(lambda: 30.0),
        incl_robot_future=False,
        incl_raster_map=True,
        raster_map_params={
            "px_per_m": 2,
            "map_size_px": 224,
            "offset_frac_xy": (-0.5, 0.0),
        },
        num_workers=4,
        verbose=True,
        data_dirs={  # Remember to change this to match your filesystem!
            "commonroad": "/home/yogeshwar10/avs/commonroad/cr_scenarios/alienware_scenarios"
        },
        rebuild_cache=False,
        incl_vector_map=True,
    )

    print(f"# Data Samples: {len(dataset):,}")

    dataloader = DataLoader(  # NOTE : Does dataloader basically just extract data from the dataset into AgentBatches? if so, what is an AgentBatch really?
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=dataset.get_collate_fn(),
        num_workers=0,
    )

    # batchElement has properties that correspond to agent states
    ego_state = dataset[0].curr_agent_state_np.copy()
    print("ego_state : \n", ego_state)

    batch: AgentBatch = next(iter(dataloader))
    ego_state_t: StateTensor = batch.curr_agent_state
    print("ego_state_t :\n", ego_state_t)

    # plot_agent_batch(batch, batch_idx=0)
    # plot_agent_batch_interactive(batch, batch_idx=0, cache_path=dataset.cache_path)

    # animation = InteractiveAnimation(
    #     animate_agent_batch_interactive,
    #     batch=batch,
    #     batch_idx=0,
    #     cache_path=dataset.cache_path,
    # )
    # animation.show()

    cache_path = Path("~/.unified_data_cache").expanduser()
    map_api = MapAPI(cache_path)

    # scene: Scene
    # for scene in dataset.scenes():
    #     env_name = "commonroad"
    #     map_name = scene.location
    #     vec_map = map_api.get_map(f"{env_name}:{map_name}", incl_road_areas=True)
    #     print(
    #         f"Number of road_area elements in {map_name} is {len(vec_map.elements[MapElementType.ROAD_AREA]) + len(vec_map.elements[MapElementType.PED_CROSSWALK]) + len(vec_map.elements[MapElementType.PED_WALKWAY]) }",
    #         flush=True,
    #     )

    env_name = "commonroad"
    location = "commonroad_1883"
    start = time.perf_counter()
    vec_map: VectorMap = map_api.get_map(f"{env_name}:{location}", incl_road_areas=True)
    end = time.perf_counter()
    print(f"Map loading took {(end - start)*1000:.2f} ms")

    # fig, ax = plt.subplots()

    # print(f"Rasterizing Map...")
    # start = time.perf_counter()
    # map_img, raster_from_world = vec_map.rasterize(
    #     resolution=2,
    #     return_tf_mat=True,
    #     incl_centerlines=False,
    #     area_color=(255, 255, 255),
    #     edge_color=(0, 0, 0),
    #     scene_ts=100,
    # )
    # end = time.perf_counter()
    # print(f"Map rasterization took {(end - start)*1000:.2f} ms")

    # ax.imshow(map_img, alpha=0.5, origin="lower")

    # lane_idx = np.random.randint(0, len(vec_map.lanes))
    # print("num of lanes is ", len(vec_map.lanes))
    # print("lane idx is ", lane_idx)
    # print("lane succesors are : ", vec_map.lanes[lane_idx].next_lanes)

    # for lane_idx in range(len(vec_map.lanes)):
    #     id = vec_map.lanes[lane_idx].id
    #     print("id is ", id)
    #     # print(vec_map.get_road_lane(id))
    # for lane in vec_map.lanes:
    #     if any(l in (None, "None") for l in lane.reachable_lanes):
    #         print(
    #             f"⚠️ Lane {lane.id} has invalid reachable lanes: {lane.reachable_lanes}"
    #         )

    # print(f"Visualizing random lane index {lane_idx}...")
    # start = time.perf_counter()
    # vec_map.visualize_lane_graph(
    #     origin_lane=lane_idx,
    #     num_hops=10,
    #     raster_from_world=raster_from_world,
    #     ax=ax,
    # )
    # end = time.perf_counter()
    # print(f"Lane visualization took {(end - start)*1000:.2f} ms")

    # point = vec_map.lanes[lane_idx].center.xyz[0, :]

    # point_raster = map_utils.transform_points(
    #     point[None, :], transf_mat=raster_from_world
    # )
    # ax.scatter(point_raster[:, 0], point_raster[:, 1])

    lane: RoadLane = vec_map.lanes[np.random.randint(0, len(vec_map.lanes))]

    ### Lane Interpolation (max_dist)
    start = time.perf_counter()
    interpolated: Polyline = lane.center.interpolate(max_dist=0.01)
    end = time.perf_counter()
    print(f"interpolate (max_dist) took {(end - start)*1000:.2f} ms")

    fig, ax = plt.subplots()
    ax.scatter(
        lane.center.points[:, 0], lane.center.points[:, 1], label="Original", s=80
    )
    ax.quiver(
        lane.center.points[:, 0],
        lane.center.points[:, 1],
        np.cos(lane.center.points[:, -1]),
        np.sin(lane.center.points[:, -1]),
    )

    ax.scatter(
        interpolated.points[:, 0], interpolated.points[:, 1], label="Interpolated"
    )
    ax.quiver(
        interpolated.points[:, 0],
        interpolated.points[:, 1],
        np.cos(interpolated.points[:, -1]),
        np.sin(interpolated.points[:, -1]),
    )

    ax.legend(loc="best")
    ax.axis("equal")

    ### Lane Interpolation (num_pts)
    start = time.perf_counter()
    interpolated: Polyline = lane.center.interpolate(num_pts=10)
    end = time.perf_counter()
    print(f"interpolate (num_pts) took {(end - start)*1000:.2f} ms")

    fig, ax = plt.subplots()
    ax.scatter(
        lane.center.points[:, 0], lane.center.points[:, 1], label="Original", s=80
    )
    ax.quiver(
        lane.center.points[:, 0],
        lane.center.points[:, 1],
        np.cos(lane.center.points[:, -1]),
        np.sin(lane.center.points[:, -1]),
    )

    ax.scatter(
        interpolated.points[:, 0], interpolated.points[:, 1], label="Interpolated"
    )
    ax.quiver(
        interpolated.points[:, 0],
        interpolated.points[:, 1],
        np.cos(interpolated.points[:, -1]),
        np.sin(interpolated.points[:, -1]),
    )

    ax.legend(loc="best")
    ax.axis("equal")

    ### Projection onto Lane
    num_pts = 2
    orig_pts = lane.center.midpoint + np.concatenate(
        [
            np.random.uniform(-3, 3, size=(num_pts, 2)),  # x,y offsets
            np.zeros(shape=(num_pts, 1)),  # no z offsets
            np.random.uniform(-np.pi, np.pi, size=(num_pts, 1)),  # headings
        ],
        axis=-1,
    )
    start = time.perf_counter()
    proj_pts = lane.center.project_onto(orig_pts)
    end = time.perf_counter()
    print(f"project_onto ({num_pts} points) took {(end - start)*1000:.2f} ms")

    fig, ax = plt.subplots()
    ax.plot(lane.center.points[:, 0], lane.center.points[:, 1], label="Lane")
    ax.scatter(orig_pts[:, 0], orig_pts[:, 1], label="Original")
    ax.quiver(
        orig_pts[:, 0],
        orig_pts[:, 1],
        np.cos(orig_pts[:, -1]),
        np.sin(orig_pts[:, -1]),
    )

    ax.scatter(proj_pts[:, 0], proj_pts[:, 1], label="Projected")
    ax.quiver(
        proj_pts[:, 0],
        proj_pts[:, 1],
        np.cos(proj_pts[:, -1]),
        np.sin(proj_pts[:, -1]),
    )

    ax.legend(loc="best")
    ax.axis("equal")

    # ### Lane Graph Visualization (with rasterized map in background)
    # fig, ax = plt.subplots()
    # map_img, raster_from_world = vec_map.rasterize(
    #     resolution=2,
    #     return_tf_mat=True,
    #     incl_centerlines=False,
    #     area_color=(255, 255, 255),
    #     edge_color=(0, 0, 0),
    #     scene_ts=100,
    # )
    # ax.imshow(map_img, alpha=0.5, origin="lower")
    # vec_map.visualize_lane_graph(
    #     origin_lane=np.random.randint(0, len(vec_map.lanes)),
    #     num_hops=5,
    #     raster_from_world=raster_from_world,
    #     ax=ax,
    # )
    # ax.axis("equal")
    # ax.grid(None)

    # ### Closest Lane Query (with rasterized map in background)
    # # vec_map.extent is [min_x, min_y, min_z, max_x, max_y, max_z]
    # min_x, min_y, _, max_x, max_y, _ = vec_map.extent

    # mean_pt: np.ndarray = np.array(
    #     [
    #         np.random.uniform(min_x, max_x),
    #         np.random.uniform(min_y, max_y),
    #         0,
    #     ]
    # )

    # start = time.perf_counter()
    # lane: RoadLane = vec_map.get_closest_lane(mean_pt)
    # end = time.perf_counter()
    # print(f"get_closest_lane took {(end - start)*1000:.2f} ms")

    # fig, ax = plt.subplots()
    # map_img, raster_from_world = vec_map.rasterize(
    #     resolution=2,
    #     return_tf_mat=True,
    #     incl_centerlines=False,
    #     area_color=(255, 255, 255),
    #     edge_color=(0, 0, 0),
    # )
    # ax.imshow(map_img, alpha=0.5, origin="lower")
    # query_pt_map: np.ndarray = map_utils.transform_points(
    #     mean_pt[None, :2], raster_from_world
    # )[0]
    # ax.scatter(*query_pt_map, label="Query Point")
    # vec_map.visualize_lane_graph(
    #     origin_lane=lane, num_hops=0, raster_from_world=raster_from_world, ax=ax
    # )
    # ax.axis("equal")
    # ax.grid(None)

    # ### Lanes Within Range Query (with rasterized map in background)
    # radius: float = 20.0

    # # vec_map.extent is [min_x, min_y, min_z, max_x, max_y, max_z]
    # min_x, min_y, _, max_x, max_y, _ = vec_map.extent

    # mean_pt: np.ndarray = np.array(
    #     [
    #         np.random.uniform(min_x, max_x),
    #         np.random.uniform(min_y, max_y),
    #         0,
    #     ]
    # )

    # start = time.perf_counter()
    # lanes: List[RoadLane] = vec_map.get_lanes_within(mean_pt, radius)
    # end = time.perf_counter()
    # print(f"get_lanes_within took {(end - start)*1000:.2f} ms")

    # fig, ax = plt.subplots()
    # img_resolution: float = 2
    # map_img, raster_from_world = vec_map.rasterize(
    #     resolution=img_resolution,
    #     return_tf_mat=True,
    #     incl_centerlines=False,
    #     area_color=(255, 255, 255),
    #     edge_color=(0, 0, 0),
    # )
    # ax.imshow(map_img, alpha=0.5, origin="lower")

    # query_pt_map: np.ndarray = map_utils.transform_points(
    #     mean_pt[None, :2], raster_from_world
    # )[0]
    # ax.scatter(*query_pt_map, label="Query Point")
    # circle2 = plt.Circle(
    #     (query_pt_map[0], query_pt_map[1]),
    #     radius * img_resolution,
    #     color="b",
    #     fill=False,
    # )
    # ax.add_patch(circle2)

    # for l in lanes:
    #     vec_map.visualize_lane_graph(
    #         origin_lane=l,
    #         num_hops=0,
    #         raster_from_world=raster_from_world,
    #         ax=ax,
    #         legend=False,
    #     )

    # ax.axis("equal")
    # ax.grid(None)
    # ax.legend(loc="best", frameon=True)

    plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
