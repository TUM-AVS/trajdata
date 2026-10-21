from typing import Dict  # Just for type annotations

import numpy as np
from tqdm import trange

from trajdata import AgentBatch, UnifiedDataset
from trajdata.data_structures.scene_metadata import Scene
from trajdata.data_structures.state import StateArray  # Just for type annotations
from trajdata.simulation import SimulationScene


def your_ego_controller(obs, ego_idx) -> StateArray:
    curr_yaw = obs.curr_agent_state[ego_idx].heading.item()
    curr_pos = obs.curr_agent_state[ego_idx].position.numpy()

    next_state = np.zeros((4,))
    next_state[:2] = curr_pos
    next_state[-1] = curr_yaw

    return StateArray.from_array(next_state, "x,y,z,h")


dataset = UnifiedDataset(
    desired_data=["commonroad"],
    data_dirs={
        "commonroad": "/home/yogeshwar10/avs/commonroad/cr_scenarios/alienware_scenarios"
    },
    verbose=True,
    rebuild_cache=False,
    save_index=False,
    num_workers=0,
    incl_vector_map=True,
)

desired_scene: Scene = dataset.get_scene(scene_idx=0)
sim_scene = SimulationScene(
    env_name="commonroad",
    scene_name=desired_scene.name,
    scene=desired_scene,
    dataset=dataset,
    init_timestep=0,
    freeze_agents=True,
)

obs: AgentBatch = sim_scene.reset()
# for t in trange(1, sim_scene.scene.length_timesteps):
#     new_xyzh_dict: Dict[str, StateArray] = dict()

#     # Everything inside the forloop just sets
#     # agents' next states to their current ones.
#     for idx, agent_name in enumerate(obs.agent_name):
#         curr_yaw = obs.curr_agent_state[idx].heading.item()
#         curr_pos = obs.curr_agent_state[idx].position.numpy()

#         next_state = np.zeros((4,))
#         next_state[:2] = curr_pos
#         next_state[-1] = curr_yaw
#         new_xyzh_dict[agent_name] = StateArray.from_array(next_state, "x,y,z,h")

#     obs = sim_scene.step(new_xyzh_dict)
#     print(obs.curr_agent_state[0].heading.item())


ego_idx = 2
ego_agent_name = "VEHICLE/30"
# for t in range(1, sim_scene.scene.length_timesteps):
#     new_xyzh_dict = {}

#     # Only provide action for ego
#     next_state = your_ego_controller(obs, ego_idx)
#     new_xyzh_dict[ego_agent_name] = next_state

#     # All other agents automatically follow their recorded paths
#     obs = sim_scene.step(new_xyzh_dict)


print(f"Scene length: {sim_scene.scene.length_timesteps}")
print(f"Num agents: {len(sim_scene.scene.agents)}")
cache = sim_scene.cache
index_dict = cache.index_dict
for k in list(index_dict.keys())[:20]:
    print(k)
