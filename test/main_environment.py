# This file includes some examples showing how to instantiate the ENVIRONMENT class, get its state, update it, and reset the object.
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR.replace('/test/main_environment.py', '')
sys.path.append(os.path.dirname(SCRIPT_DIR))
from Environment import Environment

# Creating an environment instance:
INPUT = {
    "NET": {"NUM_NODES": 6},
    "REQ": {"NUM_REQUESTS": 10},
    "SRV": {"SAMPLE": "SRVSET1"}
}
env_obj = Environment(INPUT)
state = env_obj.get_state()
print(state["NODE_FEATURES"][1])

# Updating the state of the environment
ACTION = {
    "node": 1,
    "priority": 1,
    "dc_capacity_requirement": 40,
    "burst_size": 50,
    "req_path": 0,
    "bw_requirement": 270
}
env_obj.update_state(ACTION)
state = env_obj.get_state()
print(state["NODE_FEATURES"][1])

# Reset the environment
env_obj.reset()
state = env_obj.get_state()
print(state["NODE_FEATURES"][1])
