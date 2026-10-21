from pathlib import Path
from trajdata import MapAPI, VectorMap
import numpy as np

cache_path = Path("/home/yogeshwar10/.unified_data_cache").expanduser()
map_api = MapAPI(cache_path)

vector_map: VectorMap = map_api.get_map("nuplan_mini:boston")
lane = vector_map.get_closest_lane(np.array([50.0, 100.0, 0.0]))
print(lane)