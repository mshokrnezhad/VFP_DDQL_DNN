# This file includes some examples showing how to instantiate the SERVICE class and get its state.
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR.replace('/test/main_environment.py', '')
sys.path.append(os.path.dirname(SCRIPT_DIR))
from Service import Service

# Creating a service instance using a predefined sample:
srv_obj = Service({"SAMPLE": "SRVSET1"})
print(srv_obj.SERVICES)

