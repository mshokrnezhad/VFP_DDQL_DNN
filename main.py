from Functions import read_list_from_file, parse_state, save_list_to_file
from Agent import Agent
from Environment import Environment
import numpy as np
import copy
import random
import sys

NUM_PRIORITY_LEVELS = 1
NUM_SERVICES = 1
NUM_NODES = 3
NUM_REQUESTS = 1
NUM_GAMES = 10000

SEEDS = read_list_from_file("inputs/", "SEEDS_100.txt", "int")
FILE_NAME = "V" + str(NUM_NODES) + "_K" + str(NUM_PRIORITY_LEVELS) + "_R" + str(NUM_REQUESTS) + "_S" + str(NUM_SERVICES) + "_G" + str(NUM_GAMES)

ENV_OBJ = Environment({"NET": {"SAMPLE": "NET3"}, "REQ": {"NUM_REQUESTS": NUM_REQUESTS}, "SRV": {"SAMPLE": "SRVSET2"}})
parsed_state = parse_state(NUM_NODES, ENV_OBJ.get_state(), ENV_OBJ.NET_OBJ)
AGN_OBJ = Agent({"NUM_ACTIONS": NUM_NODES, "INPUT_SHAPE": parsed_state.size, "NAME": FILE_NAME})

bst_rwd = -np.inf
game_stps = 0
rwds, epss, stps, reqs, avg_ofs, avg_dlys, dc_vars = [], [], [], [], [], [], []

for i in range(NUM_GAMES):
    SEED = SEEDS[i]
    ENV_OBJ.reset(SEED)
    BASE_DC_CAPACITIES = copy.deepcopy(ENV_OBJ.NET_OBJ.DC_CAPACITIES)
    state = parse_state(NUM_NODES, ENV_OBJ.get_state(), ENV_OBJ.NET_OBJ)
    game_rwd = 0
    game_reqs = 0
    game_of = 0
    game_dly = 0

    for r in ENV_OBJ.REQ_OBJ.REQUESTS:
        ACTION_SEED = int(random.randrange(sys.maxsize) / (10 ** 15))
        a = {"req_id": r, "node_id": AGN_OBJ.choose_action(state, ACTION_SEED)}
        resulted_state, req_rwd, done, _, req_of, req_dly = ENV_OBJ.step(a)

        game_rwd += req_rwd
        game_reqs = game_reqs + 1 if not done else game_reqs
        game_of += req_of
        game_dly += req_dly
        game_stps += 1

        AGN_OBJ.store_transition(state, a["node_id"], req_rwd, resulted_state, int(done))
        AGN_OBJ.learn()

        state = resulted_state

        # print(a["node_id"], game_reqs, game_rwd)

    rwds.append(game_rwd)
    stps.append(game_stps)
    reqs.append(game_reqs)
    avg_game_of = 0 if game_reqs == 0 else game_of / game_reqs
    avg_ofs.append(avg_game_of)
    avg_game_dly = 0 if game_reqs == 0 else game_dly / game_reqs
    avg_dlys.append(avg_game_dly)
    game_dc_var = np.var(100 * (ENV_OBJ.NET_OBJ.DC_CAPACITIES / BASE_DC_CAPACITIES))
    dc_vars.append(game_dc_var)
    epss.append(AGN_OBJ.EPSILON)

    avg_rwd = np.mean(rwds[-100:])
    if avg_rwd > bst_rwd:
        AGN_OBJ.save_models()
        bst_rwd = avg_rwd

    print(
        'episode:', i,
        'cost: %.2f' % avg_game_of,
        'reqs: %.0f' % game_reqs,
        # 'delay: %.2f' % avg_game_dly,
        # 'dc_var: %.2f' % game_dc_var,
        'reward: %.2f' % bst_rwd,
        'eps: %.4f' % AGN_OBJ.EPSILON,
        'steps:', game_stps
    )

    _suffix = ""
    _file_name = str("d_" + FILE_NAME + "_ml" + _suffix)
    _address = str("results/" + FILE_NAME + "/")
    save_list_to_file(reqs, _address, _file_name + "_reqs")
    save_list_to_file(avg_ofs, _address, _file_name + "_avg_ofs")
    save_list_to_file(rwds, _address, _file_name + "_rwds")
    save_list_to_file(epss, _address, _file_name + "_epss")
    save_list_to_file(avg_dlys, _address, _file_name + "_avg_dlys")
    save_list_to_file(dc_vars, _address, _file_name + "_dc_vars")
