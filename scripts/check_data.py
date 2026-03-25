# #!/usr/bin/env python3
# import sys
# from pathlib import Path
# from trajdata import MapAPI, UnifiedDataset
# from trajdata.caching.env_cache import EnvCache

# print("=" * 60)
# print("TRAJDATA DATA DIAGNOSTICS")
# print("=" * 60)

# cache_path = Path("~/.unified_data_cache").expanduser()
# print(f"\nCache path: {cache_path}")
# print(f"Cache exists: {cache_path.exists()}")

# # Try to load the dataset
# print("\n" + "=" * 60)
# print("LOADING DATASET")
# print("=" * 60)

# try:
#     dataset = UnifiedDataset(
#         desired_data=["commonroad"],
#         data_dirs={
#             "commonroad": "/home/yogeshwar10/avs/commonroad/cr_scenarios/alienware_scenarios"
#         },
#         verbose=True,
#         rebuild_cache=False,
#         num_workers=0,
#     )
#     print(f"\n✓ Dataset loaded successfully!")
#     print(f"  Total samples: {len(dataset)}")

#     # Try to get one sample
#     if len(dataset) > 0:
#         print(f"\nTesting first sample...")
#         sample = dataset[0]
#         print(f"✓ Sample loaded:")
#         print(f"  Scene ID: {sample.scene_id}")
#         print(f"  Agent name: {sample.agent_name}")

#         # Correct attribute names for AgentBatchElement
#         print(f"  Agent history available: {hasattr(sample, 'agent_hist_np')}")
#         print(f"  Agent future available: {hasattr(sample, 'agent_fut_np')}")

#         # List all attributes
#         print(f"\n  Available attributes:")
#         for attr in dir(sample):
#             if not attr.startswith("_"):
#                 print(f"    - {attr}")

# except Exception as e:
#     print(f"\n✗ Dataset loading failed: {e}")
#     import traceback

#     traceback.print_exc()

# # Check maps
# print("\n" + "=" * 60)
# print("CHECKING MAPS")
# print("=" * 60)

# try:
#     map_api = MapAPI(cache_path)
#     env_cache = EnvCache(cache_path)

#     scenes_list = env_cache.load_env_scenes_list("commonroad")
#     print(f"\nFound {len(scenes_list)} CommonRoad scenes")

#     # Show first few scene names
#     print("\nFirst 5 scenes:")
#     for i, scene in enumerate(scenes_list[:5]):
#         print(f"  {i+1}. Scene name: {scene.name}")
#         # Check what attributes are available
#         print(
#             f"      Available attributes: {[a for a in dir(scene) if not a.startswith('_')]}"
#         )

#     # Try to load a specific map that we know exists
#     # CommonRoad maps are named after the scenario location
#     if len(scenes_list) > 0:
#         test_scene = scenes_list[0]
#         print(f"\nTrying to load map for scene: {test_scene.name}")

#         # For CommonRoad, the map name is derived from the scenario name
#         # Extract the location part (everything before the last underscore and number)
#         # e.g., "USA_US101-28_1_T-1" -> map might be "USA_US101-28"

#         # Let's check what maps are actually cached
#         maps_dir = cache_path / "commonroad" / "maps"
#         if maps_dir.exists():
#             print(f"\nCached maps in {maps_dir}:")
#             map_files = sorted(maps_dir.glob("*.dill"))[:10]
#             for mf in map_files:
#                 print(f"  - {mf.stem}")

#             # Try to load the first map
#             if len(map_files) > 0:
#                 # Extract map name from file (remove .dill extension)
#                 first_map_file = map_files[0].stem
#                 # Map files are named like: commonroad_SCENARIONAME_2.0ppm
#                 # We need to extract just the scenario name part
#                 map_name_parts = first_map_file.split("_")
#                 # Try different variations
#                 possible_map_names = [
#                     f"commonroad:{first_map_file.replace('_2.0ppm', '').replace('commonroad_', '')}",
#                     f"commonroad:{first_map_file}",
#                 ]

#                 print(f"\nTrying to load maps:")
#                 for map_name in possible_map_names:
#                     try:
#                         print(f"  Attempting: {map_name}")
#                         vec_map = map_api.get_map(map_name)
#                         print(f"  ✓ Map loaded successfully!")
#                         print(f"    Extent: {vec_map.extent}")
#                         print(f"    Lanes: {len(vec_map.lanes)}")
#                         print(f"    Road areas: {len(vec_map.areas)}")

#                         if len(vec_map.lanes) > 0:
#                             print(f"\n  First lane details:")
#                             first_lane = vec_map.lanes[0]
#                             print(f"    Lane ID: {first_lane.id}")
#                             print(f"    Lane type: {type(first_lane).__name__}")
#                             print(
#                                 f"    Center points shape: {first_lane.center.points.shape}"
#                             )
#                             break
#                         else:
#                             print(f"    ⚠ Map has NO LANES")
#                     except Exception as e:
#                         print(f"    ✗ Failed: {e}")
#         else:
#             print(f"\n✗ No maps directory found at {maps_dir}")

# except Exception as e:
#     print(f"\n✗ Error: {e}")
#     import traceback

#     traceback.print_exc()

# print("\n" + "=" * 60)
# print("DIAGNOSTICS COMPLETE")
# print("=" * 60)

import time
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np

from trajdata import MapAPI, VectorMap
from trajdata.maps.vec_map import Polyline, RoadLane
from trajdata.utils import map_utils


def main():
    cache_path = Path("~/.unified_data_cache").expanduser()
    map_api = MapAPI(cache_path)

    # List available .pb map files
    maps_dir = cache_path / "commonroad" / "maps"
    if not maps_dir.exists():
        print("ERROR: No maps directory found!")
        return

    # Get all .pb files (vector maps)
    pb_files = sorted(maps_dir.glob("commonroad_*.pb"))
    print(f"Found {len(pb_files)} vector map files\n")

    if len(pb_files) == 0:
        print("ERROR: No .pb map files found!")
        return

    # Show first 10 maps
    print("First 10 available maps:")
    for i, pb_file in enumerate(pb_files[:10]):
        # Extract just the number from filename like "commonroad_0.pb"
        map_id = pb_file.stem.replace("commonroad_", "")
        print(f"  {i+1}. commonroad:{map_id}")

    # Use the first map
    first_map_id = pb_files[0].stem.replace("commonroad_", "")
    map_name = f"commonroad:commonroad_{first_map_id}"

    print(f"\nLoading map: {map_name}")

    try:
        vec_map: VectorMap = map_api.get_map(map_name)
        print(f"✓ Map loaded successfully!")
        print(f"  Extent: {vec_map.extent}")
        print(f"  Lanes: {len(vec_map.lanes)}")
        # print(f"  Road areas: {len(vec_map.areas)}")
    except Exception as e:
        print(f"✗ Failed to load map: {e}")
        import traceback

        traceback.print_exc()
        return

    # Check if map has lanes
    if len(vec_map.lanes) == 0:
        print("\nERROR: This map has no lanes!")
        print("Trying next map...")

        # Try up to 10 maps to find one with lanes
        for pb_file in pb_files[1:11]:
            map_id = pb_file.stem.replace("commonroad_", "")
            map_name = f"commonroad:{map_id}"
            print(f"  Trying: {map_name}")

            try:
                vec_map = map_api.get_map(map_name)
                if len(vec_map.lanes) > 0:
                    print(f"  ✓ Found map with {len(vec_map.lanes)} lanes!")
                    break
            except:
                continue

        if len(vec_map.lanes) == 0:
            print("ERROR: Could not find any map with lanes!")
            return

    print(f"\nUsing map with {len(vec_map.lanes)} lanes")

    ### Select a random lane
    lane_idx = np.random.randint(0, len(vec_map.lanes))
    lane: RoadLane = vec_map.lanes[lane_idx]
    print(f"Selected lane {lane_idx}: {lane.id}")

    ### Lane Interpolation (max_dist)
    print("\n--- Lane Interpolation Tests ---")
    start = time.perf_counter()
    interpolated: Polyline = lane.center.interpolate(max_dist=0.01)
    end = time.perf_counter()
    print(f"interpolate (max_dist=0.01) took {(end - start)*1000:.2f} ms")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        lane.center.points[:, 0],
        lane.center.points[:, 1],
        label="Original",
        s=80,
        c="blue",
        zorder=3,
    )
    ax.quiver(
        lane.center.points[:, 0],
        lane.center.points[:, 1],
        np.cos(lane.center.points[:, -1]),
        np.sin(lane.center.points[:, -1]),
        scale=50,
        width=0.003,
        color="blue",
    )

    ax.scatter(
        interpolated.points[:, 0],
        interpolated.points[:, 1],
        label="Interpolated",
        s=20,
        c="red",
        alpha=0.5,
    )
    ax.quiver(
        interpolated.points[:, 0],
        interpolated.points[:, 1],
        np.cos(interpolated.points[:, -1]),
        np.sin(interpolated.points[:, -1]),
        scale=50,
        width=0.002,
        color="red",
        alpha=0.5,
    )

    ax.legend(loc="best")
    ax.axis("equal")
    ax.set_title("Lane Interpolation (max_dist=0.01)")
    ax.grid(True, alpha=0.3)

    ### Lane Interpolation (num_pts)
    start = time.perf_counter()
    interpolated: Polyline = lane.center.interpolate(num_pts=10)
    end = time.perf_counter()
    print(f"interpolate (num_pts=10) took {(end - start)*1000:.2f} ms")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        lane.center.points[:, 0],
        lane.center.points[:, 1],
        label="Original",
        s=80,
        c="blue",
        zorder=3,
    )
    ax.quiver(
        lane.center.points[:, 0],
        lane.center.points[:, 1],
        np.cos(lane.center.points[:, -1]),
        np.sin(lane.center.points[:, -1]),
        scale=50,
        width=0.003,
        color="blue",
    )

    ax.scatter(
        interpolated.points[:, 0],
        interpolated.points[:, 1],
        label="Interpolated (10 pts)",
        s=80,
        c="red",
        marker="x",
    )
    ax.quiver(
        interpolated.points[:, 0],
        interpolated.points[:, 1],
        np.cos(interpolated.points[:, -1]),
        np.sin(interpolated.points[:, -1]),
        scale=50,
        width=0.003,
        color="red",
    )

    ax.legend(loc="best")
    ax.axis("equal")
    ax.set_title("Lane Interpolation (num_pts=10)")
    ax.grid(True, alpha=0.3)

    ### Projection onto Lane
    print("\n--- Lane Projection Test ---")
    num_pts = 15
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

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        lane.center.points[:, 0],
        lane.center.points[:, 1],
        "b-",
        linewidth=2,
        label="Lane Centerline",
    )

    ax.scatter(
        orig_pts[:, 0],
        orig_pts[:, 1],
        s=80,
        c="green",
        marker="o",
        label="Original Points",
        zorder=3,
    )
    ax.quiver(
        orig_pts[:, 0],
        orig_pts[:, 1],
        np.cos(orig_pts[:, -1]),
        np.sin(orig_pts[:, -1]),
        scale=30,
        width=0.003,
        color="green",
    )

    ax.scatter(
        proj_pts[:, 0],
        proj_pts[:, 1],
        s=80,
        c="red",
        marker="x",
        label="Projected Points",
        zorder=3,
    )
    ax.quiver(
        proj_pts[:, 0],
        proj_pts[:, 1],
        np.cos(proj_pts[:, -1]),
        np.sin(proj_pts[:, -1]),
        scale=30,
        width=0.003,
        color="red",
    )

    # Draw lines connecting original to projected
    for i in range(len(orig_pts)):
        ax.plot(
            [orig_pts[i, 0], proj_pts[i, 0]],
            [orig_pts[i, 1], proj_pts[i, 1]],
            "k--",
            alpha=0.3,
            linewidth=0.5,
        )

    ax.legend(loc="best")
    ax.axis("equal")
    ax.set_title("Projection onto Lane Centerline")
    ax.grid(True, alpha=0.3)

    ### Lane Graph Visualization
    print("\n--- Lane Graph Visualization ---")
    fig, ax = plt.subplots(figsize=(12, 10))

    try:
        map_img, raster_from_world = vec_map.rasterize(
            resolution=2,
            return_tf_mat=True,
            incl_centerlines=False,
            area_color=(255, 255, 255),
            edge_color=(0, 0, 0),
        )
        ax.imshow(map_img, alpha=0.5, origin="lower")

        # Select a random lane for visualization
        random_lane: RoadLane = vec_map.lanes[np.random.randint(0, len(vec_map.lanes))]
        print(f"Visualizing lane graph from lane: {random_lane.id}")

        vec_map.visualize_lane_graph(
            origin_lane=random_lane,
            num_hops=5,
            raster_from_world=raster_from_world,
            ax=ax,
        )
        ax.axis("equal")
        ax.grid(None)
        ax.set_title(f"Lane Graph (5 hops from {random_lane.id})")
        print("✓ Lane graph visualization successful")

    except Exception as e:
        print(f"✗ Lane graph visualization failed: {e}")
        import traceback

        traceback.print_exc()

    ### Closest Lane Query
    print("\n--- Closest Lane Query ---")
    min_x, min_y, _, max_x, max_y, _ = vec_map.extent
    mean_pt: np.ndarray = np.array(
        [
            np.random.uniform(min_x, max_x),
            np.random.uniform(min_y, max_y),
            0,
        ]
    )

    start = time.perf_counter()
    closest_lane: RoadLane = vec_map.get_closest_lane(mean_pt)
    end = time.perf_counter()
    print(f"get_closest_lane took {(end - start)*1000:.2f} ms")
    print(f"Closest lane: {closest_lane.id}")

    fig, ax = plt.subplots(figsize=(12, 10))
    map_img, raster_from_world = vec_map.rasterize(
        resolution=2,
        return_tf_mat=True,
        incl_centerlines=False,
        area_color=(255, 255, 255),
        edge_color=(0, 0, 0),
    )
    ax.imshow(map_img, alpha=0.5, origin="lower")

    query_pt_map: np.ndarray = map_utils.transform_points(
        mean_pt[None, :2], raster_from_world
    )[0]
    ax.scatter(
        *query_pt_map,
        s=200,
        c="red",
        marker="*",
        label="Query Point",
        zorder=10,
        edgecolor="black",
        linewidth=2,
    )

    vec_map.visualize_lane_graph(
        origin_lane=closest_lane,
        num_hops=0,
        raster_from_world=raster_from_world,
        ax=ax,
    )
    ax.axis("equal")
    ax.grid(None)
    ax.set_title(f"Closest Lane to Query Point: {closest_lane.id}")
    ax.legend()

    ### Lanes Within Range Query
    print("\n--- Lanes Within Range Query ---")
    radius: float = 20.0
    mean_pt: np.ndarray = np.array(
        [
            np.random.uniform(min_x, max_x),
            np.random.uniform(min_y, max_y),
            0,
        ]
    )

    start = time.perf_counter()
    lanes: List[RoadLane] = vec_map.get_lanes_within(mean_pt, radius)
    end = time.perf_counter()
    print(f"get_lanes_within (radius={radius}m) took {(end - start)*1000:.2f} ms")
    print(f"Found {len(lanes)} lanes within {radius}m")

    fig, ax = plt.subplots(figsize=(12, 10))
    img_resolution: float = 2
    map_img, raster_from_world = vec_map.rasterize(
        resolution=img_resolution,
        return_tf_mat=True,
        incl_centerlines=False,
        area_color=(255, 255, 255),
        edge_color=(0, 0, 0),
    )
    ax.imshow(map_img, alpha=0.5, origin="lower")

    query_pt_map: np.ndarray = map_utils.transform_points(
        mean_pt[None, :2], raster_from_world
    )[0]
    ax.scatter(
        *query_pt_map,
        s=200,
        c="red",
        marker="*",
        label=f"Query Point",
        zorder=10,
        edgecolor="black",
        linewidth=2,
    )

    circle = plt.Circle(
        (query_pt_map[0], query_pt_map[1]),
        radius * img_resolution,
        color="red",
        fill=False,
        linewidth=2,
        linestyle="--",
        label=f"{radius}m radius",
    )
    ax.add_patch(circle)

    for l in lanes:
        vec_map.visualize_lane_graph(
            origin_lane=l,
            num_hops=0,
            raster_from_world=raster_from_world,
            ax=ax,
            legend=False,
        )

    ax.axis("equal")
    ax.grid(None)
    ax.legend(loc="best", frameon=True)
    ax.set_title(f"Lanes Within {radius}m (found {len(lanes)} lanes)")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
