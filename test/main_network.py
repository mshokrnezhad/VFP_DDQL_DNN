# This file includes some examples showing how to instantiate the NETWORK class, get its state, and update its state.
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR.replace('/test/main_environment.py', '')
sys.path.append(os.path.dirname(SCRIPT_DIR))
from Network import Network

# Creating a network instance using a predefined sample:
net_obj = Network({"SAMPLE": "NET1"})
net_obj.plot()

# # Creating a network instance using the default configurations:
# net_obj = Network({"NUM_NODES": 9, "NUM_PRIORITY_LEVELS": 1, "DC_COST_MU": 500})
# net_obj.plot()
# # node_features, links_matrix, link_features = net_obj.get_state() #Getting the state.
# # net_obj.update_state(action={"node": 1, "priority": 1}, req_info={"dc_capacity_requirement": 40, "burst_size": 50, "req_path": 0, "bw_requirement": 270}) #Updating the state.
