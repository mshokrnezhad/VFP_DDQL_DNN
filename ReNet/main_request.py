# This file includes some examples showing how to instantiate the REQUEST class and get its state.
from request.Request import Request
from service.Service import Service
from network.Network import Network
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
