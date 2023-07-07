from Environment import Environment
from Agent import Agent
import numpy as np
import random
import sys
from Functions import parse_state, plot_learning_curve, calculate_input_shape, save_list_to_file, simple_plot
import numpy as np
import matplotlib.pyplot as plt
import copy

rnd = np.random


class Multi_Game_VNF_Placement(object):
    def __init__(self, NUM_NODES, NUM_REQUESTS, NUM_SERVICES, NUM_PRIORITY_LEVELS, NUM_GAMES, SEEDS):
        self.SWITCH = "vnf_plc"
        self.NUM_NODES = NUM_NODES
        self.NUM_REQUESTS = NUM_REQUESTS
        self.NUM_SERVICES = NUM_SERVICES
        self.NUM_PRIORITY_LEVELS = NUM_PRIORITY_LEVELS
        self.NUM_GAMES = NUM_GAMES
        self.NUM_ACTIONS = NUM_NODES
        self.SEEDS = SEEDS
        self.FILE_NAME = "V" + str(NUM_NODES) + "_K" + str(NUM_PRIORITY_LEVELS) + "_R" + str(NUM_REQUESTS) + "_S" + str(NUM_SERVICES) + "_G" + str(NUM_GAMES)
        self.env_obj = Environment(NUM_NODES=NUM_NODES, NUM_REQUESTS=NUM_REQUESTS, NUM_SERVICES=NUM_SERVICES, NUM_PRIORITY_LEVELS=NUM_PRIORITY_LEVELS)
        self.agent = Agent(NUM_ACTIONS=self.NUM_ACTIONS, INPUT_SHAPE=self.env_obj.get_state().size, NAME=self.FILE_NAME)

    def wf_alloc(self):
        reqs, avg_ofs, avg_dlys, dc_vars = [], [], [], []

        for i in range(self.NUM_GAMES):
            SEED = self.SEEDS[i]
            self.env_obj.reset(SEED)

            result = self.env_obj.heu_obj.solve()

            reqs.append(result["reqs"])

            avg_ofs.append(result["avg_of"])

            avg_dlys.append(result["avg_dly"])

            dc_vars.append(result["dc_var"])

            print(
                'episode:', i,
                'cost: %.2f' % result["avg_of"],
                'reqs: %.0f' % result["reqs"],
                'dly: %.2f' % result["avg_dly"],
                'dc_var: %.2f' % result["dc_var"],
            )

        save_list_to_file(reqs, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_wf_reqs")
        save_list_to_file(avg_ofs, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_wf_avg_ofs")
        save_list_to_file(avg_dlys, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_wf_avg_dlys")
        save_list_to_file(dc_vars, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_wf_dc_vars")

    def ddql_alloc_train(self):
        bst_rwd = -np.inf
        game_stps = 0
        rwds, epss, stps, reqs, avg_ofs, avg_dlys, dc_vars = [], [], [], [], [], [], []

        for i in range(self.NUM_GAMES):
            SEED = self.SEEDS[i]
            self.env_obj.reset(SEED)
            BASE_DC_CAPACITIES = copy.deepcopy(self.env_obj.net_obj.DC_CAPACITIES)
            state = self.env_obj.get_state()
            game_rwd = 0
            game_reqs = 0
            game_of = 0
            game_dly = 0

            for r in self.env_obj.req_obj.REQUESTS:
                ACTION_SEED = int(random.randrange(sys.maxsize) / (10 ** 15))  # 4
                a = {"req_id": r, "node_id": self.agent.choose_action(state, ACTION_SEED)}
                resulted_state, req_rwd, done, info, req_of, req_dly = self.env_obj.step(a, "none")

                game_rwd += req_rwd
                game_reqs = game_reqs + 1 if not done else game_reqs
                game_of += req_of
                game_dly += req_dly
                game_stps += 1

                self.agent.store_transition(state, a["node_id"], req_rwd, resulted_state, int(done))
                self.agent.learn()

                state = resulted_state
                # print(a["node_id"], req_rwd)

            rwds.append(game_rwd)

            stps.append(game_stps)

            reqs.append(game_reqs)

            avg_game_of = 0 if game_reqs == 0 else game_of / game_reqs
            avg_ofs.append(avg_game_of)

            avg_game_dly = 0 if game_reqs == 0 else game_dly / game_reqs
            avg_dlys.append(avg_game_dly)

            game_dc_var = np.var(100 * (self.env_obj.net_obj.DC_CAPACITIES / BASE_DC_CAPACITIES))
            dc_vars.append(game_dc_var)

            epss.append(self.agent.EPSILON)

            avg_rwd = np.mean(rwds[-100:])
            if avg_rwd > bst_rwd:
                self.agent.save_models()
                bst_rwd = avg_rwd

            print(
                'episode:', i,
                'cost: %.2f' % avg_game_of,
                'reqs: %.0f' % game_reqs,
                'delay: %.2f' % avg_game_dly,
                'dc_var: %.2f' % game_dc_var,
                'reward: %.2f' % bst_rwd,
                'eps: %.4f' % self.agent.EPSILON,
                'steps:', game_stps
            )

        """
        save_list_to_file(reqs, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_ml_reqs")
        save_list_to_file(avg_ofs, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_ml_avg_ofs")
        save_list_to_file(rwds, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_ml_rwds")
        save_list_to_file(epss, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_ml_epss")
        save_list_to_file(avg_dlys, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_ml_avg_dlys")
        save_list_to_file(dc_vars, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_ml_dc_vars")
        """
        suffix = ""
        save_list_to_file(reqs, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_ml" + suffix + "_reqs")
        save_list_to_file(avg_ofs, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_ml" + suffix + "_avg_ofs")
        save_list_to_file(rwds, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_ml" + suffix + "_rwds")
        save_list_to_file(epss, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_ml" + suffix + "_epss")
        save_list_to_file(avg_dlys, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_ml" + suffix + "_avg_dlys")
        save_list_to_file(dc_vars, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_ml" + suffix + "_dc_vars")

    def ddql_alloc_eval(self):
        self.agent.load_models()
        self.agent.EPSILON = 0

        best_reward = -np.inf
        num_steps = 0
        rewards, epsilons, steps, ml_nums_act_reqs, ml_avg_ofs = [], [], [], [], []

        for i in range(self.NUM_GAMES):
            SEED = self.SEEDS[i]
            self.env_obj.reset(SEED)
            state = self.env_obj.get_state()
            game_reward = 0
            ml_game_num_act_reqs = 0
            ml_game_of = 0
            for r in self.env_obj.req_obj.REQUESTS:
                ACTION_SEED = int(random.randrange(sys.maxsize) / (10 ** 15))  # 4
                a = {"req_id": r, "node_id": self.agent.choose_action(state, ACTION_SEED)}
                resulted_state, req_reward, done, info, req_of, req_dly = self.env_obj.step(a, "none")
                game_reward += req_reward
                if not done:
                    ml_game_num_act_reqs += 1
                ml_game_of += req_of
                state = resulted_state
                num_steps += 1
                # print(a["node_id"], req_reward)
            rewards.append(game_reward)
            steps.append(num_steps)
            ml_nums_act_reqs.append(ml_game_num_act_reqs)
            ml_avg_game_of = 0 if ml_game_num_act_reqs == 0 else ml_game_of / ml_game_num_act_reqs
            ml_avg_ofs.append(ml_avg_game_of)
            avg_reward = np.mean(rewards[-100:])
            epsilons.append(self.agent.EPSILON)
            if avg_reward > best_reward:
                best_reward = avg_reward
            print('episode:', i, 'cost: %.3f, num_act_reqs: %.0f, eps: %.4f' % (ml_avg_game_of, ml_game_num_act_reqs, self.agent.EPSILON), 'steps:', num_steps)

        save_list_to_file(ml_nums_act_reqs, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_ml_nums_act_reqs_eval")
        save_list_to_file(ml_avg_ofs, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_ml_avg_ofs_eval")
        save_list_to_file(rewards, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_rewards_eval")

        # simple_plot(range(self.NUM_GAMES), ml_avg_ofs, filename="results/" + self.FILE_NAME + "/" + self.FILE_NAME + "_ml_avg_ofs" + '.png')

    def rnd_alloc(self):
        reqs, avg_ofs, avg_dlys, dc_vars = [], [], [], []

        for i in range(self.NUM_GAMES):
            SEED = self.SEEDS[i]
            self.env_obj.reset(SEED)
            BASE_DC_CAPACITIES = copy.deepcopy(self.env_obj.net_obj.DC_CAPACITIES)
            game_reqs = 0
            game_of = 0
            game_dly = 0

            for r in self.env_obj.req_obj.REQUESTS:
                ACTION_SEED = int(random.randrange(sys.maxsize) / (10 ** 15))  # 4
                rnd.seed(ACTION_SEED)
                a = {"req_id": r, "node_id": rnd.choice(self.env_obj.net_obj.NODES)}
                resulted_state, req_rwd, done, info, req_of, req_dly = self.env_obj.step(a, "none")

                game_reqs = game_reqs + 1 if not done else game_reqs
                game_of += req_of
                game_dly += req_dly

            reqs.append(game_reqs)

            avg_game_of = 0 if game_reqs == 0 else game_of / game_reqs
            avg_ofs.append(avg_game_of)

            avg_game_dly = 0 if game_reqs == 0 else game_dly / game_reqs
            avg_dlys.append(avg_game_dly)

            game_dc_var = np.var(100 * (self.env_obj.net_obj.DC_CAPACITIES / BASE_DC_CAPACITIES))
            dc_vars.append(game_dc_var)

            print(
                'episode:', i,
                'cost: %.2f' % avg_game_of,
                'reqs: %.0f' % game_reqs,
                'delay: %.2f' % avg_game_dly,
                'dc_var: %.2f' % game_dc_var,
            )

        save_list_to_file(reqs, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_rnd_reqs")
        save_list_to_file(avg_ofs, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_rnd_avg_ofs")
        save_list_to_file(avg_dlys, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_rnd_avg_dlys")
        save_list_to_file(dc_vars, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_rnd_dc_vars")

    def cm_alloc(self):
        reqs, avg_ofs, avg_dlys, dc_vars = [], [], [], []

        for i in range(self.NUM_GAMES):
            SEED = self.SEEDS[i]
            self.env_obj.reset(SEED)
            BASE_DC_CAPACITIES = copy.deepcopy(self.env_obj.net_obj.DC_CAPACITIES)
            game_reqs = 0
            game_of = 0
            game_dly = 0

            for r in self.env_obj.req_obj.REQUESTS:
                a = {"req_id": r, "node_id": np.argmin(self.env_obj.net_obj.DC_COSTS)}
                resulted_state, req_rwd, done, info, req_of, req_dly = self.env_obj.step(a, "none")

                game_reqs = game_reqs + 1 if not done else game_reqs
                game_of += req_of
                game_dly += req_dly

            reqs.append(game_reqs)

            avg_game_of = 0 if game_reqs == 0 else game_of / game_reqs
            avg_ofs.append(avg_game_of)

            avg_game_dly = 0 if game_reqs == 0 else game_dly / game_reqs
            avg_dlys.append(avg_game_dly)

            game_dc_var = np.var(100 * (self.env_obj.net_obj.DC_CAPACITIES / BASE_DC_CAPACITIES))
            dc_vars.append(game_dc_var)

            print(
                'episode:', i,
                'cost: %.2f' % avg_game_of,
                'reqs: %.0f' % game_reqs,
                'delay: %.2f' % avg_game_dly,
                'dc_var: %.2f' % game_dc_var,
            )

        save_list_to_file(reqs, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_cm_reqs")
        save_list_to_file(avg_ofs, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_cm_avg_ofs")
        save_list_to_file(avg_dlys, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_cm_avg_dlys")
        save_list_to_file(dc_vars, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_cm_dc_vars")

    def dm_alloc(self):
        reqs, avg_ofs, avg_dlys, dc_vars = [], [], [], []

        for i in range(self.NUM_GAMES):
            SEED = self.SEEDS[i]
            self.env_obj.reset(SEED)
            BASE_DC_CAPACITIES = copy.deepcopy(self.env_obj.net_obj.DC_CAPACITIES)
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
                resulted_state, req_rwd, done, info, req_of, req_dly = self.env_obj.step(a, "none")

                game_reqs = game_reqs + 1 if not done else game_reqs
                game_of += req_of
                game_dly += req_dly

                # print(a["node_id"], a["node_id"])

            reqs.append(game_reqs)

            avg_game_of = 0 if game_reqs == 0 else game_of / game_reqs
            avg_ofs.append(avg_game_of)

            avg_game_dly = 0 if game_reqs == 0 else game_dly / game_reqs
            avg_dlys.append(avg_game_dly)

            game_dc_var = np.var(100 * (self.env_obj.net_obj.DC_CAPACITIES / BASE_DC_CAPACITIES))
            dc_vars.append(game_dc_var)

            print(
                'episode:', i,
                'cost: %.2f' % avg_game_of,
                'reqs: %.0f' % game_reqs,
                'delay: %.2f' % avg_game_dly,
                'dc_var: %.2f' % game_dc_var,
            )

        save_list_to_file(reqs, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_dm_reqs")
        save_list_to_file(avg_ofs, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_dm_avg_ofs")
        save_list_to_file(avg_dlys, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_dm_avg_dlys")
        save_list_to_file(dc_vars, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_dm_dc_vars")

    def lb_alloc(self):
        reqs, avg_ofs, avg_dlys, dc_vars = [], [], [], []

        for i in range(self.NUM_GAMES):
            SEED = self.SEEDS[i]
            self.env_obj.reset(SEED)
            BASE_DC_CAPACITIES = copy.deepcopy(self.env_obj.net_obj.DC_CAPACITIES)
            game_reqs = 0
            game_of = 0
            game_dly = 0

            assigned_node = 0
            done = False
            assigned_nodes = []

            for r in self.env_obj.req_obj.REQUESTS:
                dcls = 100 * (self.env_obj.net_obj.DC_CAPACITIES / BASE_DC_CAPACITIES)
                for v in assigned_nodes:
                    dcls[v] = 0
                assigned_node = dcls.argmax()
                a = {"req_id": r, "node_id": assigned_node}
                resulted_state, req_rwd, done, info, req_of, req_dly = self.env_obj.step(a, "none")

                if done:
                    assigned_nodes.append(assigned_node)

                game_reqs = game_reqs + 1 if not done else game_reqs
                game_of += req_of
                game_dly += req_dly

                # print(a["node_id"], a["node_id"])

            reqs.append(game_reqs)

            avg_game_of = 0 if game_reqs == 0 else game_of / game_reqs
            avg_ofs.append(avg_game_of)

            avg_game_dly = 0 if game_reqs == 0 else game_dly / game_reqs
            avg_dlys.append(avg_game_dly)

            game_dc_var = np.var(100 * (self.env_obj.net_obj.DC_CAPACITIES / BASE_DC_CAPACITIES))
            dc_vars.append(game_dc_var)

            print(
                'episode:', i,
                'cost: %.2f' % avg_game_of,
                'reqs: %.0f' % game_reqs,
                'delay: %.2f' % avg_game_dly,
                'dc_var: %.2f' % game_dc_var,
            )

        save_list_to_file(reqs, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_lb_reqs")
        save_list_to_file(avg_ofs, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_lb_avg_ofs")
        save_list_to_file(avg_dlys, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_lb_avg_dlys")
        save_list_to_file(dc_vars, "results/" + self.FILE_NAME + "/", "d_" + self.FILE_NAME + "_lb_dc_vars")

    """
    def random_alloc(self):  # everything is random
        random_nums_act_reqs, random_avg_ofs = [], []

        for i in range(self.NUM_GAMES):
            SEED = self.SEEDS[i]
            self.env_obj.reset(SEED)
            
            random_results = self.env_obj.random_obj.solve()
            # print(opt_results["pairs"])
            random_game_num_act_reqs = random_results["num_act_reqs"]
            random_nums_act_reqs.append(random_game_num_act_reqs)
            random_avg_game_of = random_results["avg_of"]
            random_avg_ofs.append(random_avg_game_of)
            print('episode:', i, 'cost: %.0f, num_act_reqs: %.0f' %(random_avg_game_of, random_game_num_act_reqs))

        save_list_to_file(random_nums_act_reqs, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_random_nums_act_reqs")
        save_list_to_file(random_avg_ofs, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_random_avg_ofs")

        # simple_plot(range(self.NUM_GAMES), opt_avg_ofs, filename="results/" + self.FILE_NAME + "/" + self.FILE_NAME + "_opt_avg_ofs" + '.png')
    """
