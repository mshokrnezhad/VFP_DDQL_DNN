from Functions import read_list_from_file, parse_state
from Agent import Agent
from Environment import Environment

NUM_PRIORITY_LEVELS = 1
NUM_SERVICES = 1
NUM_NODES = 3
NUM_REQUESTS = 10
NUM_GAMES = 10000

SEEDS = read_list_from_file("inputs/", "SEEDS_100.txt", "int")
FILE_NAME = "V" + str(NUM_NODES) + "_K" + str(NUM_PRIORITY_LEVELS) + "_R" + str(NUM_REQUESTS) + "_S" + str(NUM_SERVICES) + "_G" + str(NUM_GAMES)

ENV_OBJ = Environment({"NET": {"SAMPLE": "NET2"}, "REQ": {"NUM_REQUESTS": NUM_REQUESTS}, "SRV": {"SAMPLE": "SRVSET1"}})
parsed_state = parse_state(NUM_NODES, ENV_OBJ.get_state(), ENV_OBJ.NET_OBJ)
AGN_OBJ = Agent({"NUM_ACTIONS": NUM_NODES, "INPUT_SHAPE": parsed_state.size, "NAME": FILE_NAME})
