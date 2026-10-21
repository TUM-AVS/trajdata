import matplotlib.pyplot as plt

# import functions to read xml file and visualize commonroad objects
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer


# generate path of the file to be opened
file_path = "commonroad/cr_scenarios/alienware_scenarios/ARG_Carcarana-1_8_T-1.xml"

# read in the scenario and planning problem set
scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

# left verticies of a partcular lanelet in the scenario
print(scenario.lanelet_network.lanelets[0].left_vertices)
