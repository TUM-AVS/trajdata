#!/usr/bin/python -u
from typing import Dict
import numpy as np

import os, logging, time
from pathlib import Path
from trajdata import MapAPI, VectorMap
from torch.utils.data import DataLoader
from trajdata import AgentBatch, UnifiedDataset
from trajdata.data_structures.scene_metadata import Scene  # Just for type annotations
from trajdata.data_structures.state import StateArray
from trajdata.data_structures.state import NP_STATE_TYPES
from trajdata.simulation import SimulationScene

import sys

print("starting script now, printing", flush=True)

# See below for a list of already-supported datasets and splits.
dataset = UnifiedDataset(
    desired_data=["commonroad"],
    data_dirs={  # Remember to change this to match your filesystem!
        "commonroad": "/home/yogeshwar10/avs/commonroad/cr_scenarios/alienware_scenarios"
    },
    verbose=True,
    rebuild_cache=False,
    save_index=False,
    num_workers=0,
    incl_vector_map=True,
)
print("executed load dataset line", flush=True)


# print("Testing dataset indices...")
# valid_indices = []
# for idx in range(len(dataset)):
#     try:
#         _ = dataset[idx]  # Try to fetch
#         valid_indices.append(idx)
#     except KeyError as e:
#         print(f"Index {idx} failed: {e}")
#     except Exception as e:
#         print(f"Index {idx} failed: {type(e).__name__}: {e}")

#     if idx % 100 == 0:
#         print(f"Tested {idx}/{len(dataset)} indices...")

# print(f"\nValid indices: {len(valid_indices)}/{len(dataset)}")


dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=dataset.get_collate_fn(),
    num_workers=6,  # os.cpu_count(), # This can be set to 0 for single-threaded loading, if desired.
)

print("executed dataloader line", flush=True)

print(dataloader.batch_size)
print(dataloader.__len__())
print(dataloader)


cache_path = Path("~/.unified_data_cache").expanduser()
map_api = MapAPI(cache_path)

vector_map: VectorMap = map_api.get_map("commonroad:commonroad_787")
print("loaded VectorMap")

desired_scene: Scene = dataset.get_scene(scene_idx=78)
sim_scene = SimulationScene(
    env_name="commonroad",
    scene_name="commonroad",
    scene=desired_scene,
    dataset=dataset,
    init_timestep=0,
    freeze_agents=True,
)

STATE_FORMAT_KEY = "x,y,h"
obs: AgentBatch = sim_scene.reset()
for t in range(1, sim_scene.scene.length_timesteps):
    new_xyh_dict: Dict[str, np.ndarray] = dict()

    # Everything inside the forloop just sets
    # agents' next states to their current ones.
    for idx, agent_name in enumerate(obs.agent_name):
        curr_yaw = obs.curr_agent_state[idx, -1]
        curr_pos = obs.curr_agent_state[idx, :2]

        next_state = np.zeros((3,))
        next_state[:2] = curr_pos
        next_state[2] = curr_yaw

        state_array_object = StateArray.from_array(next_state, format=STATE_FORMAT_KEY)
        new_xyh_dict[agent_name] = state_array_object
        print(f"  Timestep {t}")
        print(f"  Agent: {agent_name}")
        print(
            f"    Current state (x,y,h): {curr_pos[0]:.2f}, {curr_pos[1]:.2f}, {curr_yaw:.2f}"
        )

    obs = sim_scene.step(new_xyh_dict)

# for batch_idx, batch in enumerate(dataloader):
#     try:
#         print(f"Batch {batch_idx}")
#         # Your training code here
#     except KeyError as e:
#         print(f"Skipping batch {batch_idx} due to KeyError: {e}")
#         continue
#     except Exception as e:
#         print(f"Skipping batch {batch_idx} due to error: {type(e).__name__}: {e}")
#         continue
