from Environment import Environment
from Agent import Agent
import numpy as np
import random
import sys
from Functions import parse_state, plot_learning_curve, calculate_input_shape, save_list_to_file, simple_plot, append_list_to_file
import numpy as np
import matplotlib.pyplot as plt
import copy
rnd = np.random


class Single_Game_VNF_Placement(object):
    def __init__(self, NUM_NODES, NUM_REQUESTS, NUM_SERVICES, NUM_PRIORITY_LEVELS, NUM_GAMES, SEEDS, MIN_NUM_REQUESTS=0, MAX_NUM_REQUESTS=0, MODE="none", REVISION="1"):
        self.SWITCH = "vnf_plc"
        self.NUM_NODES = NUM_NODES
        self.NUM_REQUESTS = NUM_REQUESTS
        self.MIN_NUM_REQUESTS = MIN_NUM_REQUESTS
        self.MAX_NUM_REQUESTS = MAX_NUM_REQUESTS
        self.NUM_SERVICES = NUM_SERVICES
        self.NUM_PRIORITY_LEVELS = NUM_PRIORITY_LEVELS
        self.NUM_GAMES = NUM_GAMES
        self.NUM_ACTIONS = NUM_NODES
        self.SEEDS = SEEDS
        self.FILE_NAME = "K" + str(NUM_PRIORITY_LEVELS) + "_S" + str(NUM_SERVICES) + "_" + MODE + "_V" + REVISION
        self.env_obj = Environment(NUM_NODES=NUM_NODES, NUM_REQUESTS=NUM_REQUESTS, NUM_SERVICES=NUM_SERVICES, NUM_PRIORITY_LEVELS=NUM_PRIORITY_LEVELS)
        self.agent = Agent(NUM_ACTIONS=self.NUM_ACTIONS, INPUT_SHAPE=self.env_obj.get_state().size, NAME=self.FILE_NAME)

    def wf_alloc(self):
        reqs, avg_ofs, avg_dlys, dc_vars = [], [], [], []

        for i in range(self.NUM_GAMES):
            SEED = self.SEEDS[2]
            self.env_obj.reset(SEED)

            result = self.env_obj.heu_obj.solve()

            reqs.append(result["reqs"])
            avg_ofs.append(result["avg_of"])
            avg_dlys.append(result["avg_dly"])
            dc_vars.append(result["dc_var"])

            print(
                'num_reqs:', self.NUM_REQUESTS,
                'num_nodes:', self.NUM_NODES,
                'cost: %.2f' % result["avg_of"],
                'reqs: %.0f' % result["reqs"],
                'dly: %.2f' % result["avg_dly"],
                'dc_var: %.2f' % result["dc_var"],
            )

        append_list_to_file(reqs, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_wf_reqs")
        append_list_to_file("\n", "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_wf_reqs")
        append_list_to_file(avg_ofs, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_wf_avg_ofs")
        append_list_to_file("\n", "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_wf_avg_ofs")
        append_list_to_file(avg_dlys, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_wf_avg_dlys")
        append_list_to_file("\n", "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_wf_avg_dlys")
        append_list_to_file(dc_vars, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_wf_dc_vars")
        append_list_to_file("\n", "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_wf_dc_vars")

    def rnd_alloc(self):
        reqs, avg_ofs, avg_dlys, dc_vars = [], [], [], []

        for i in range(self.NUM_GAMES):
            SEED = self.SEEDS[i]
            self.env_obj.reset(SEED)
            BASE_DC_CAPACITIES = copy.deepcopy(
                self.env_obj.net_obj.DC_CAPACITIES)
            game_reqs = 0
            game_of = 0
            game_dly = 0

            for r in self.env_obj.req_obj.REQUESTS:
                ACTION_SEED = int(random.randrange(
                    sys.maxsize) / (10 ** 15))  # 4
                rnd.seed(ACTION_SEED)
                a = {"req_id": r, "node_id": rnd.choice(
                    self.env_obj.net_obj.NODES)}
                resulted_state, req_rwd, done, info, req_of, req_dly = self.env_obj.step(
                    a, "none")

                game_reqs = game_reqs + 1 if not done else game_reqs
                game_of += req_of
                game_dly += req_dly

            reqs.append(game_reqs)

            avg_game_of = 0 if game_reqs == 0 else game_of / game_reqs
            avg_ofs.append(avg_game_of)

            avg_game_dly = 0 if game_reqs == 0 else game_dly / game_reqs
            avg_dlys.append(avg_game_dly)

            game_dc_var = np.var(
                100 * (self.env_obj.net_obj.DC_CAPACITIES / BASE_DC_CAPACITIES))
            dc_vars.append(game_dc_var)

            print(
                'num_reqs:', self.NUM_REQUESTS,
                'num_nodes:', self.NUM_NODES,
                'cost: %.2f' % avg_game_of,
                'reqs: %.0f' % game_reqs,
                'delay: %.2f' % avg_game_dly,
                'dc_var: %.2f' % game_dc_var,
            )

        append_list_to_file(reqs, "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_rnd_reqs")
        append_list_to_file("\n", "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_rnd_reqs")
        append_list_to_file(avg_ofs, "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_rnd_avg_ofs")
        append_list_to_file("\n", "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_rnd_avg_ofs")
        append_list_to_file(avg_dlys, "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_rnd_avg_dlys")
        append_list_to_file("\n", "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_rnd_avg_dlys")
        append_list_to_file(dc_vars, "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_rnd_dc_vars")
        append_list_to_file("\n", "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_rnd_dc_vars")

    def cm_alloc(self):
        reqs, avg_ofs, avg_dlys, dc_vars = [], [], [], []

        for i in range(self.NUM_GAMES):
            SEED = self.SEEDS[i]
            self.env_obj.reset(SEED)
            BASE_DC_CAPACITIES = copy.deepcopy(
                self.env_obj.net_obj.DC_CAPACITIES)
            game_reqs = 0
            game_of = 0
            game_dly = 0

            for r in self.env_obj.req_obj.REQUESTS:
                a = {"req_id": r, "node_id": np.argmin(
                    self.env_obj.net_obj.DC_COSTS)}
                resulted_state, req_rwd, done, info, req_of, req_dly = self.env_obj.step(
                    a, "none")

                game_reqs = game_reqs + 1 if not done else game_reqs
                game_of += req_of
                game_dly += req_dly

            reqs.append(game_reqs)

            avg_game_of = 0 if game_reqs == 0 else game_of / game_reqs
            avg_ofs.append(avg_game_of)

            avg_game_dly = 0 if game_reqs == 0 else game_dly / game_reqs
            avg_dlys.append(avg_game_dly)

            game_dc_var = np.var(
                100 * (self.env_obj.net_obj.DC_CAPACITIES / BASE_DC_CAPACITIES))
            dc_vars.append(game_dc_var)

            print(
                'num_reqs:', self.NUM_REQUESTS,
                'num_nodes:', self.NUM_NODES,
                'cost: %.2f' % avg_game_of,
                'reqs: %.0f' % game_reqs,
                'delay: %.2f' % avg_game_dly,
                'dc_var: %.2f' % game_dc_var,
            )

        append_list_to_file(reqs, "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_cm_reqs")
        append_list_to_file("\n", "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_cm_reqs")
        append_list_to_file(avg_ofs, "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_cm_avg_ofs")
        append_list_to_file("\n", "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_cm_avg_ofs")
        append_list_to_file(avg_dlys, "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_cm_avg_dlys")
        append_list_to_file("\n", "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_cm_avg_dlys")
        append_list_to_file(dc_vars, "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_cm_dc_vars")
        append_list_to_file("\n", "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_cm_dc_vars")

    def dm_alloc(self):
        reqs, avg_ofs, avg_dlys, dc_vars = [], [], [], []

        for i in range(self.NUM_GAMES):
            SEED = self.SEEDS[i]
            self.env_obj.reset(SEED)
            BASE_DC_CAPACITIES = copy.deepcopy(
                self.env_obj.net_obj.DC_CAPACITIES)
            game_reqs = 0
            game_of = 0
            game_dly = 0

            for r in self.env_obj.req_obj.REQUESTS:
                # node_delays = np.zeros(self.NUM_NODES)
                # for v in self.env_obj.net_obj.NODES:
                # node_delay = self.env_obj.net_obj.find_argmin_e2e_delay_per_node_pair(self.env_obj.REQUESTS_ENTRY_NODES[r], v, self.env_obj.req_obj.CAPACITY_REQUIREMENTS[r])
                # node_delays[v] = node_delay if node_delay != -1 else 1000
                # a = {"req_id": r, "node_id": node_delays.argmin()}
                a = {"req_id": r, "node_id": 0}
                resulted_state, req_rwd, done, info, req_of, req_dly = self.env_obj.step(
                    a, "none")

                game_reqs = game_reqs + 1 if not done else game_reqs
                game_of += req_of
                game_dly += req_dly

                # print(a["node_id"], a["node_id"])

            reqs.append(game_reqs)

            avg_game_of = 0 if game_reqs == 0 else game_of / game_reqs
            avg_ofs.append(avg_game_of)

            avg_game_dly = 0 if game_reqs == 0 else game_dly / game_reqs
            avg_dlys.append(avg_game_dly)

            game_dc_var = np.var(
                100 * (self.env_obj.net_obj.DC_CAPACITIES / BASE_DC_CAPACITIES))
            dc_vars.append(game_dc_var)

            print(
                'num_reqs:', self.NUM_REQUESTS,
                'num_nodes:', self.NUM_NODES,
                'cost: %.2f' % avg_game_of,
                'reqs: %.0f' % game_reqs,
                'delay: %.2f' % avg_game_dly,
                'dc_var: %.2f' % game_dc_var,
            )

        append_list_to_file(reqs, "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_dm_reqs")
        append_list_to_file(avg_ofs, "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_dm_avg_ofs")
        append_list_to_file(avg_dlys, "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_dm_avg_dlys")
        append_list_to_file(dc_vars, "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_dm_dc_vars")
        append_list_to_file("\n", "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_dm_reqs")
        append_list_to_file("\n", "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_dm_avg_ofs")
        append_list_to_file("\n", "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_dm_avg_dlys")
        append_list_to_file("\n", "results/" + self.FILE_NAME +
                            "/", "d_" + self.FILE_NAME + "_dm_dc_vars")
