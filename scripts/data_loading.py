import os
from torch.utils.data import DataLoader
from trajdata import AgentBatch, UnifiedDataset


# See below for a list of already-supported datasets and splits.
dataset = UnifiedDataset(
    desired_data=["nuplan_mini"],
    data_dirs={  # Remember to change this to match your filesystem!
        "nuplan_mini": "/home/yogeshwar10/nuplan/dataset/nuplan-v1.1" 
    },
    rebuild_cache=False,
)

dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=dataset.get_collate_fn(),
    num_workers=4   #os.cpu_count(), # This can be set to 0 for single-threaded loading, if desired.
)

batch: AgentBatch
for batch in dataloader:
    print(batch)
    