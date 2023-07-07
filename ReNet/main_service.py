# This file includes some examples showing how to instantiate the SERVICE class and get its state.
from service.Service import Service

# Creating a service instance using a predefined sample:
srv_obj = Service({"SAMPLE": "SRVSET1"})
print(srv_obj.SERVICES)

