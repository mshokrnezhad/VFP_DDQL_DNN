from Environment import Environment
from Agent import Agent
import numpy as np
import random
import sys
from Functions import parse_state, plot_learning_curve, calculate_input_shape, save_list_to_file, simple_plot, append_list_to_file
import numpy as np
import matplotlib.pyplot as plt
import copy
from P1FSA import P1FSA
from P1BSA import P1BSA
from P2CEP import P2CEP
from P3ADDPG import P3ADDPG
from P4MDRLSADS import P4MDRLSADS
rnd = np.random


class Single_Game_VNF_Placement(object):
    def __init__(self, inputs={}):
        self.SWITCH = "vnf_plc"
        self.NUM_NODES = inputs["NUM_NODES"]
        self.NUM_REQUESTS = inputs["NUM_REQUESTS"]
        self.NUM_SERVICES = inputs["NUM_SERVICES"]
        self.NUM_PRIORITY_LEVELS = inputs["NUM_PRIORITY_LEVELS"]
        self.NUM_GAMES = inputs["NUM_GAMES"]
        self.NUM_ACTIONS = inputs["NUM_NODES"]
        self.SEEDS = inputs["SEEDS"]
        self.FILE_NAME = "K" + str(self.NUM_PRIORITY_LEVELS) + "_S" + str(
            self.NUM_SERVICES) + "_" + inputs["MODE"] + "_" + inputs["REVISION"] + "_V" + inputs["VERSION"]
        self.env_obj = Environment(NUM_NODES=self.NUM_NODES, NUM_REQUESTS=self.NUM_REQUESTS,
                                   NUM_SERVICES=self.NUM_SERVICES, NUM_PRIORITY_LEVELS=self.NUM_PRIORITY_LEVELS)
        self.agent = Agent(NUM_ACTIONS=self.NUM_ACTIONS,
                           INPUT_SHAPE=self.env_obj.get_state().size, NAME=self.FILE_NAME)
        self.p1fsa_obj = P1FSA(net_obj=self.env_obj.net_obj, req_obj=self.env_obj.req_obj, srv_obj=self.env_obj.srv_obj,
                               REQUESTED_SERVICES=self.env_obj.REQUESTED_SERVICES, REQUESTS_ENTRY_NODES=self.env_obj.REQUESTS_ENTRY_NODES)
        self.p1bsa_obj = P1BSA(net_obj=self.env_obj.net_obj, req_obj=self.env_obj.req_obj, srv_obj=self.env_obj.srv_obj,
                               REQUESTED_SERVICES=self.env_obj.REQUESTED_SERVICES, REQUESTS_ENTRY_NODES=self.env_obj.REQUESTS_ENTRY_NODES)
        self.p2cep_obj = P2CEP(net_obj=self.env_obj.net_obj, req_obj=self.env_obj.req_obj, srv_obj=self.env_obj.srv_obj,
                               REQUESTED_SERVICES=self.env_obj.REQUESTED_SERVICES, REQUESTS_ENTRY_NODES=self.env_obj.REQUESTS_ENTRY_NODES)
        self.p3addpg_obj = P3ADDPG(net_obj=self.env_obj.net_obj, req_obj=self.env_obj.req_obj, srv_obj=self.env_obj.srv_obj,
                               REQUESTED_SERVICES=self.env_obj.REQUESTED_SERVICES, REQUESTS_ENTRY_NODES=self.env_obj.REQUESTS_ENTRY_NODES)
        self.p4mdrlsads_obj = P4MDRLSADS(net_obj=self.env_obj.net_obj, req_obj=self.env_obj.req_obj, srv_obj=self.env_obj.srv_obj,
                               REQUESTED_SERVICES=self.env_obj.REQUESTED_SERVICES, REQUESTS_ENTRY_NODES=self.env_obj.REQUESTS_ENTRY_NODES)

    def refine_outputs(self, reqs, avg_ofs, avg_dlys, dc_vars, mname):
        addr = "results/" + self.FILE_NAME + "/"
        fname = "d_" + self.FILE_NAME + "_" + mname + "_"
        append_list_to_file(reqs, addr, fname + "reqs")
        append_list_to_file("\n", addr, fname + "reqs")
        append_list_to_file(avg_ofs, addr, fname + "avg_ofs")
        append_list_to_file("\n", addr, fname + "avg_ofs")
        append_list_to_file(avg_dlys, addr, fname + "avg_dlys")
        append_list_to_file("\n", addr, fname + "avg_dlys")
        append_list_to_file(dc_vars, addr, fname + "dc_vars")
        append_list_to_file("\n", addr, fname + "dc_vars")

    def wfccra(self):
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

        self.refine_outputs(reqs, avg_ofs, avg_dlys, dc_vars, mname="wfccra")

    def p1fsa(self):
        reqs, avg_ofs, avg_dlys, dc_vars = [], [], [], []

        for i in range(self.NUM_GAMES):
            SEED = self.SEEDS[2]
            self.env_obj.reset(SEED)

            result = self.p1fsa_obj.solve()

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

        self.refine_outputs(reqs, avg_ofs, avg_dlys, dc_vars, mname="p1fsa")

    def p1bsa(self):
        reqs, avg_ofs, avg_dlys, dc_vars = [], [], [], []

        for i in range(self.NUM_GAMES):
            SEED = self.SEEDS[2]
            self.env_obj.reset(SEED)

            result = self.p1bsa_obj.solve()

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

        self.refine_outputs(reqs, avg_ofs, avg_dlys, dc_vars, mname="p1bsa")

    def p2cep(self):
        reqs, avg_ofs, avg_dlys, dc_vars = [], [], [], []

        for i in range(self.NUM_GAMES):
            SEED = self.SEEDS[2]
            self.env_obj.reset(SEED)

            result = self.p2cep_obj.solve()

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

        self.refine_outputs(reqs, avg_ofs, avg_dlys, dc_vars, mname="p2cep")

    def p3addpg(self):  
        reqs, avg_ofs, avg_dlys, dc_vars = [], [], [], []

        for i in range(self.NUM_GAMES):
            SEED = self.SEEDS[2]
            self.env_obj.reset(SEED)

            result = self.p3addpg_obj.solve()

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

        self.refine_outputs(reqs, avg_ofs, avg_dlys, dc_vars, mname="p3addpg")

    def p4mdrlsads(self):
        reqs, avg_ofs, avg_dlys, dc_vars = [], [], [], []

        for i in range(self.NUM_GAMES):
            SEED = self.SEEDS[2]
            self.env_obj.reset(SEED)

            result = self.p4mdrlsads_obj.solve()

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

        self.refine_outputs(reqs, avg_ofs, avg_dlys, dc_vars, mname="p4mdrlsads")
