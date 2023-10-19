# This file includes some examples showing how to instantiate the REQUEST class and get its state.
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR.replace('/test/main_environment.py', '')
sys.path.append(os.path.dirname(SCRIPT_DIR))
from Request import Request
from Service import Service
from Network import Network
net_obj = Network({"SAMPLE": "NET1"})
srv_obj = Service({"SAMPLE": "SRVSET1"})


# # Creating a request instance using a predefined sample:
# req_obj = Request({"NET_OBJ": net_obj, "SRV_OBJ": srv_obj, "SAMPLE": "REQ1"})
# # request_features = req_obj.get_state()
# # print(request_features)
# print(req_obj.ASSIGNED_SERVICES)

# Creating a request instance using the default configurations:
req_obj = Request({"NET_OBJ": net_obj, "SRV_OBJ": srv_obj, "NUM_REQUESTS": 10})
# request_features = req_obj.get_state()
# print(request_features)
print(req_obj.ENTRY_NODES)
print(req_obj.ASSIGNED_SERVICES)
print(req_obj.DELAY_REQUIREMENTS)
