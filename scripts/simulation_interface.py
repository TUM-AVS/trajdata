from typing import Dict # Just for type annotations

import numpy as np
from trajdata.data_structures.state import StateArray
from trajdata.data_structures.state import NP_STATE_TYPES

from trajdata import AgentBatch, UnifiedDataset
from trajdata.data_structures.scene_metadata import Scene # Just for type annotations
from trajdata.simulation import SimulationScene



# See below for a list of already-supported datasets and splits.
dataset = UnifiedDataset(
    desired_data=["nuplan_mini"],
    data_dirs={  # Remember to change this to match your filesystem!
        "nuplan_mini": "/home/yogeshwar10/nuplan/dataset/nuplan-v1.1"
    },
    rebuild_cache=True,
)

desired_scene: Scene = dataset.get_scene(scene_idx=0)
sim_scene = SimulationScene(
    env_name="nusc_mini_sim",
    scene_name="sim_scene",
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
        print(f"  Agent: {agent_name}")
        print(f"    Current state (x,y,h): {curr_pos[0]:.2f}, {curr_pos[1]:.2f}, {curr_yaw:.2f}")


    obs = sim_scene.step(new_xyh_dict)