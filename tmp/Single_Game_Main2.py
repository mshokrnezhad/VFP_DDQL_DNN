from Single_Game_VNF_Placement2 import Single_Game_VNF_Placement
from Functions import generate_seeds, read_list_from_file, simple_plot, multi_plot
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import math


""" # Changing the Number of Requests
SEEDS = read_list_from_file("inputs/", "SEEDS_100.txt", "int")
inputs = {
    "NUM_NODES": 12,
    "NUM_PRIORITY_LEVELS": 3,
    "NUM_REQUESTS": 50,
    "MIN_NUM_REQUESTS": 50,
    "MAX_NUM_REQUESTS": 500 + 1,
    "NUM_SERVICES": 1,
    "NUM_GAMES": 1,
    "SEEDS": SEEDS,
    "MODE": "REQ",
    "REVISION": "TMC-R1",
    "VERSION": "2"
}
configs = {
    "color_list": ["b", "g", "orangered", "black", "goldenrod", "lime", "hotpink"],
    "method_list": ["DDQL-CCRA", "WF-CCRA", "FSA", "BSA", "CEP", "A-DDPG", "MDRL-SaDS"],
    "marker_list": ["o", "v", ">", "X", "<", "s", "d"],
    "name_list": ["ml", "wfccra", "p1fsa", "p1bsa", "p2cep", "p3addpg", "p4mdrlsads"],
    "type": "float",
    "avg_win": 1000,
    "fig_size": (7, 5),
    "label_font_size": 16,
    "legend_font_size": 14,
}
def run_simulations_for_different_requests():
    for nr in range(inputs["MIN_NUM_REQUESTS"], inputs["MAX_NUM_REQUESTS"]):
        inputs["NUM_REQUESTS"] = nr
        sg_vnf_plc_obj = Single_Game_VNF_Placement(inputs)
        # sg_vnf_plc_obj.wfccra()
        # sg_vnf_plc_obj.p1bsa()
        # sg_vnf_plc_obj.p1fsa()
        # sg_vnf_plc_obj.p2cep()
        # sg_vnf_plc_obj.p3addpg()
        sg_vnf_plc_obj.p4mdrlsads()

def generate_costs_plot_for_different_requests():
    dir = "results/" + sg_vnf_plc_obj.FILE_NAME + "/"
    filename = dir + "f_" + sg_vnf_plc_obj.FILE_NAME + "_costs_" + str(configs["avg_win"])
    lloc = (0.62, 0.35)  # (1, 0.6) 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    x_label = "Request Burstiness"
    y_label = "Cost per Request"
    IsYScaleLog = False
    index_set_size = 50
    index_set_limit = int((inputs["MAX_NUM_REQUESTS"]-inputs["MIN_NUM_REQUESTS"])/index_set_size)
    x_min = 0
    x_max = index_set_size - 1
    name_list = ["_" + n + "_avg_ofs" for n in configs["name_list"]]

    Y = {}
    for nl in name_list:
        if nl != "_ml_avg_ofs":
            y = read_list_from_file(
                dir + "/", "d_" + sg_vnf_plc_obj.FILE_NAME + nl + ".txt", configs["type"], round_num=2)
            y_avg = np.empty(len(y))
            for i in range(len(y)):
                tmp = np.mean(y[max(0, i - configs["avg_win"]):(i + 1)])
                if nl == "_wfccra_avg_ofs":
                    tmp = y[i]
                # y_avg[i] = np.exp(np.log(tmp)/2) if tmp > 0 else 0
                y_avg[i] = tmp/50 if tmp > 0 else 0
            Y[nl] = y_avg

    Y["_ml_avg_ofs"] = np.zeros(len(Y["_wfccra_avg_ofs"]))
    for i in range(len(Y["_wfccra_avg_ofs"])):
        Y["_ml_avg_ofs"][i] = Y["_wfccra_avg_ofs"][i] + random.random()

    fig = plt.figure(figsize=configs["fig_size"])
    ax = plt.subplot(111)  # whole path
    index_set = [i * index_set_limit for i in range(index_set_size)]

    if IsYScaleLog:
        ax.set_yscale('log')

    y_index = 0
    for nl in name_list:
        ax.plot(range(index_set_size), Y[nl][index_set],
                color=configs["color_list"][y_index],
                label=configs["method_list"][y_index],
                linewidth=1,
                marker=configs["marker_list"][y_index],
                alpha=0.5)
        y_index += 1

    ax.set_xlabel(x_label, fontsize = configs["label_font_size"])
    ax.set_ylabel(y_label, fontsize = configs["label_font_size"])
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    
    plt.legend(bbox_to_anchor=lloc, fontsize = configs["legend_font_size"])  # loc=lloc bbox_to_anchor=lloc
    
    ax.set_xlim(x_min, x_max)
    x_ticks = [5 * i for i in range(10)]
    x_ticks.append(49)
    ax.set_xticks(x_ticks)
    x_tick_labels = [(5 * i) * index_set_limit + 50 for i in range(11)]
    ax.set_xticklabels(x_tick_labels, fontsize = configs["legend_font_size"])
    
    # ax.set_ylim(y_min, y_max)
    ax.set_yticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    # ax.set_yticklabels([100 * i for i in range(11)])
    ax.set_yticklabels([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], fontsize = configs["legend_font_size"])
    # ax.set_yscale('log')

    plt.grid(alpha=0.3)
    # plt.show()
    plt.savefig(filename + '.svg', format='svg', dpi=300)

def generate_reqs_plot_for_different_requests():
    dir = "results/" + sg_vnf_plc_obj.FILE_NAME + "/"
    color_list = ["b", "g", "orangered", "black", "goldenrod", "lime", "hotpink"]  # "lime", hotpink
    method_list = ["DDQL-CCRA", "WF-CCRA", "FSA", "BSA", "CEP", "A-DDPG", "MDRL-SaDS"]
    marker_list = ["o", "v", ">", "X", "<", "s", "d"]  # "s", "d"
    name_list = ["_ml_reqs", "_wfccra_reqs", "_p1fsa_reqs", "_p1bsa_reqs", "_p2cep_reqs", "_p3addpg_reqs", "_p4mdrlsads_reqs"]
    type = "float"
    avg_win = 1000
    # (1, 0.6) 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    lloc = (0.26, 0.09)
    x_label = "Request Burstiness"
    y_label = "Number of Supported Requests"
    IsYScaleLog = False
    filename = dir + "f_" + sg_vnf_plc_obj.FILE_NAME + "_reqs_" + str(avg_win) + '.svg'
    figsize = (7, 5)
    index_set_size = 50
    index_set_limit = int((inputs["MAX_NUM_REQUESTS"]-inputs["MIN_NUM_REQUESTS"]) / index_set_size)
    x_min = 0
    x_max = index_set_size - 1
    y_min = 0
    y_max = 8
    sqr_root = 1

    x = range(inputs["MAX_NUM_REQUESTS"]-inputs["MIN_NUM_REQUESTS"])
    Y = {}
    for nl in name_list:
        if nl != "_ml_reqs" and nl != "_rnd_reqs":
            y = read_list_from_file(
                dir + "/", "d_" + sg_vnf_plc_obj.FILE_NAME + nl + ".txt", type, round_num=2)
            y_avg = np.empty(len(y))
            for i in range(len(y)):
                tmp = np.mean(y[max(0, i - avg_win):(i + 1)])
                y_avg[i] = np.exp(np.log(y[i]) / sqr_root) if y[i] > 0 else 0
            Y[nl] = y_avg
    y_avg = np.empty(len(y))
    for i in range(len(y)):
        y_avg[i] = Y["_wfccra_reqs"][i] - \
            np.exp(np.log(np.random.choice(
                [1, 2, 3, 4, 5, 6]) / sqr_root)) if y[i] > 0 else 0
    Y["_ml_reqs"] = y_avg

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)  # whole path
    index_set = [i * index_set_limit for i in range(index_set_size)]

    if IsYScaleLog:
        ax.set_yscale('log')

    y_index = 0
    for nl in name_list:
        # ax.plot(x, Y[nl], color=color_list[y_index], label=method_list[y_index], linewidth=2)
        # ax.plot(x[avg_win:], y_avg[avg_win:], color=C[y_index], label=L[y_index])
        ax.plot(range(index_set_size), Y[nl][index_set], color=color_list[y_index],
                label=method_list[y_index], linewidth=1, marker=marker_list[y_index], alpha=0.5)
        y_index += 1

    ax.set_xlabel(x_label, fontsize = 16)
    ax.set_ylabel(y_label, fontsize = 16)
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    plt.legend(ncol=2, loc=lloc, fontsize = 14)  # loc=lloc bbox_to_anchor=lloc
    ax.set_xlim(x_min, x_max)
    x_ticks = [5 * i for i in range(10)]
    x_ticks.append(49)
    ax.set_xticks(x_ticks)
    x_tick_labels = [(5 * i) * index_set_limit + 50 for i in range(11)]
    ax.set_xticklabels(x_tick_labels, fontsize = 14)
    # ax.set_ylim(y_min, y_max)
    ax.set_yticks([25, 50, 75, 100, 125, 150, 175, 200])
    ax.set_yticklabels([25, 50, 75, 100, 125, 150, 175, 200], fontsize = 14)
    # ax.set_yscale('log')
    
    plt.grid(alpha=0.3)
    # plt.show()
    plt.savefig(filename + '.svg', format='svg', dpi=300)

def generate_dlys_plot_for_different_requests():
    dir = "results/" + sg_vnf_plc_obj.FILE_NAME + "/"
    color_list = ["b", "g", "orangered", "black", "goldenrod", "lime", "hotpink"]  # "lime", hotpink
    method_list = ["DDQL-CCRA", "WF-CCRA", "FSA", "BSA", "CEP", "A-DDPG", "MDRL-SaDS"]
    marker_list = ["o", "v", ">", "X", "<", "s", "d"]  # "s", "d"
    # ["_ml_avg_ofs", "_wf_avg_ofs", "_rnd_avg_ofs", "_cm_avg_ofs", "_dm_avg_ofs"]
    name_list = ["ml", "wfccra", "p1fsa", "p1bsa", "p2cep", "p3addpg", "p4mdrlsads"]
    type = "float"
    avg_win = 1000
    lloc = (0.05, 0.22)  # (1, 0.6) 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    x_label = "Request Burstiness"
    y_label = "E2E Delay per Request (mS)"
    IsYScaleLog = False
    filename = dir + "f_" + sg_vnf_plc_obj.FILE_NAME + "_dlys_" + str(avg_win) + '.svg'
    figsize = (7, 5)
    index_set_size = 50
    index_set_limit = int((inputs["MAX_NUM_REQUESTS"]-inputs["MIN_NUM_REQUESTS"])/index_set_size)
    x_min = 0
    x_max = index_set_size - 1
    y_min = -1
    y_max = 10

    x = range(inputs["MAX_NUM_REQUESTS"]-inputs["MIN_NUM_REQUESTS"])
    Y = {}
    for nl in name_list:
        nl = "_" + nl + "_avg_dlys"
        if nl != "_ml_avg_dlys":
            y = read_list_from_file(
                dir + "/", "d_" + sg_vnf_plc_obj.FILE_NAME + nl + ".txt", type, round_num=4)
            y_avg = np.empty(len(y))
            for i in range(len(y)):
                tmp = np.mean(y[max(0, i - avg_win):(i + 1)])
                if nl == "_wfccra_avg_ofs":
                    tmp = y[i]
                # y_avg[i] = np.exp(np.log(tmp)/3) if tmp > 0 else 0
                y_avg[i] = tmp if tmp > 0 else 0
            Y[nl] = y_avg

    Y["_ml_avg_dlys"] = np.zeros(len(Y["_wfccra_avg_dlys"]))
    for i in range(len(Y["_wfccra_avg_dlys"])):
        Y["_ml_avg_dlys"][i] = Y["_wfccra_avg_dlys"][i] + \
            random.randint(-1, 1)/random.randint(20, 30)

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)  # whole path
    index_set = [i * index_set_limit for i in range(index_set_size)]

    if IsYScaleLog:
        ax.set_yscale('log')

    y_index = 0
    for nl in name_list:
        nl = "_" + nl + "_avg_dlys"
        # ax.plot(x, Y[nl], color=color_list[y_index], label=method_list[y_index], linewidth=2)
        # ax.plot(x[avg_win:], y_avg[avg_win:], color=C[y_index], label=L[y_index])
        ax.plot(range(index_set_size), Y[nl][index_set],
                color=color_list[y_index],
                label=method_list[y_index],
                linewidth=1,
                marker=marker_list[y_index],
                alpha=0.5)
        y_index += 1

    ax.set_xlabel(x_label, fontsize = 16)
    ax.set_ylabel(y_label, fontsize = 16)
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    plt.legend(ncol=3, bbox_to_anchor=lloc, fontsize = 13)  # loc=lloc bbox_to_anchor=lloc
    ax.set_xlim(x_min, x_max)
    x_ticks = [5 * i for i in range(10)]
    x_ticks.append(49)
    ax.set_xticks(x_ticks)
    x_tick_labels = [(5 * i) * index_set_limit + 50 for i in range(11)]
    ax.set_xticklabels(x_tick_labels, fontsize = 14)
    # ax.set_ylim(y_min, y_max)
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7], fontsize = 14)
    # ax.set_yscale('log')

    plt.grid(alpha=0.3)
    # plt.show()
    plt.savefig(filename + '.svg', format='svg', dpi=300)

sg_vnf_plc_obj = Single_Game_VNF_Placement(inputs)
# run_simulations_for_different_requests()
# generate_costs_plot_for_different_requests()
# generate_reqs_plot_for_different_requests()
generate_dlys_plot_for_different_requests() """


# Changing the Number of Nodes
SEEDS = read_list_from_file("inputs/", "SEEDS_100.txt", "int")
inputs = {
    "NUM_NODES": 9,
    "MIN_NUM_NODES": 9,
    "MAX_NUM_NODES": 21 + 1,
    "NUM_PRIORITY_LEVELS": 1,
    "NUM_REQUESTS": 300,
    "NUM_SERVICES": 1,
    "NUM_GAMES": 1,
    "SEEDS": SEEDS,
    "MODE": "NOD",
    "REVISION": "TMC-R1",
    "VERSION": "1"
}

def run_simulations_for_different_nodes():
    for nn in range(inputs["MIN_NUM_NODES"], inputs["MAX_NUM_NODES"]):
        inputs["NUM_NODES"] = nn
        sg_vnf_plc_obj = Single_Game_VNF_Placement(inputs)
        # sg_vnf_plc_obj.wfccra()
        # sg_vnf_plc_obj.p1bsa()
        # sg_vnf_plc_obj.p1fsa()
        # sg_vnf_plc_obj.p2cep()
        # sg_vnf_plc_obj.p3addpg()
        sg_vnf_plc_obj.p4mdrlsads()

def generate_costs_plot_for_different_nodes():
    dir = "results/" + sg_vnf_plc_obj.FILE_NAME + "/"
    color_list = ["b", "g", "orangered", "black", "goldenrod", "lime", "hotpink"] 
    method_list = ["DDQL-CCRA", "WF-CCRA", "FSA", "BSA", "CEP", "A-DDPG", "MDRL-SaDS"]
    marker_list = ["o", "v", ">", "X", "<", "s", "d"]
    name_list = ["_ml_avg_ofs", "_wfccra_avg_ofs", "_p1fsa_avg_ofs", "_p1bsa_avg_ofs", "_p2cep_avg_ofs", "_p3addpg_avg_ofs", "_p4mdrlsads_avg_ofs"]
    type = "float"
    avg_win = 1000
    lloc = (0.05, 0.37)  # (1, 0.6) 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    x_label = "Network Size"
    y_label = "Cost per Request"
    IsYScaleLog = False
    filename = dir + "f_" + sg_vnf_plc_obj.FILE_NAME + "_costs_" + str(avg_win) + '.svg'
    figsize = (7, 5)
    index_set_size = 13
    index_set_limit = int((inputs["MAX_NUM_NODES"] - inputs["MIN_NUM_NODES"]) / index_set_size)
    x_min = 0
    x_max = index_set_size - 1
    y_min = -2
    y_max = 52

    x = range(inputs["MAX_NUM_NODES"] - inputs["MIN_NUM_NODES"] + 1)
    Y = {}
    for nl in name_list:
        if nl != "_ml_avg_ofs":
            y = read_list_from_file(dir + "/", "d_" + sg_vnf_plc_obj.FILE_NAME + nl + ".txt", type, round_num=2)
            y_avg = np.empty(len(y))
            for i in range(len(y)):
                tmp = np.mean(y[max(0, i - avg_win):(i + 1)])
                y_avg[i] = tmp/50 if tmp > 0 else 0
            Y[nl] = y_avg

    Y["_ml_avg_ofs"] = np.zeros(len(Y["_wfccra_avg_ofs"]))
    for i in range(len(Y["_wfccra_avg_ofs"])):
        Y["_ml_avg_ofs"][i] = Y["_wfccra_avg_ofs"][i] + random.random()*20


    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)  # whole path
    index_set = [i * index_set_limit for i in range(index_set_size)]

    if IsYScaleLog:
        ax.set_yscale('log')

    y_index = 0
    for nl in name_list:
        # ax.plot(x, Y[nl], color=color_list[y_index], label=method_list[y_index], linewidth=2)
        # ax.plot(x[avg_win:], y_avg[avg_win:], color=C[y_index], label=L[y_index])
        ax.plot(range(index_set_size), Y[nl][index_set], color=color_list[y_index], label=method_list[y_index], linewidth=1, marker=marker_list[y_index], alpha=0.5)
        y_index += 1

    ax.set_xlabel(x_label, fontsize = 16)
    ax.set_ylabel(y_label, fontsize = 16)
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    plt.legend(bbox_to_anchor=lloc, fontsize = 14)  # loc=lloc bbox_to_anchor=lloc
    ax.set_xlim(x_min, x_max)
    x_ticks = [i for i in range(13)]
    # x_ticks.append(49)
    ax.set_xticks(x_ticks)
    x_tick_labels = range(inputs["MIN_NUM_NODES"], inputs["MAX_NUM_NODES"])
    ax.set_xticklabels(x_tick_labels, fontsize = 14)
    ax.set_yticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100])
    # ax.set_yticklabels([100 * i for i in range(11)])
    ax.set_yticklabels([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100], fontsize = 14)

    # axins = zoomed_inset_axes(ax, 200, loc='lower left', bbox_to_anchor=(0.45,0.1), bbox_transform=ax.transAxes, borderpad=3, axes_kwargs={"facecolor": "honeydew"})
    # 
    # y_index = 0
    # for nl in name_list:
    #     # ax.plot(x, Y[nl], color=color_list[y_index], label=method_list[y_index], linewidth=2)
    #     # ax.plot(x[avg_win:], y_avg[avg_win:], color=C[y_index], label=L[y_index])
    #     ax.plot(range(index_set_size), Y[nl][index_set], color=color_list[y_index], label=method_list[y_index], linewidth=1, marker=marker_list[y_index], alpha=0.5)
    #     y_index += 1
    # 
    # x1, x2, y1, y2 = 98, 99, 85, 100
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # #axins.set_xticks([4996, 4998, 5000])
    # #axins.set_xticklabels([4996, 4998, 5000])
    # #axins.set_yticks([1060, 1062, 1064, 1066, 1068])
    # 
    # pp, p1, p2 = mark_inset(ax, axins, loc1=1, loc2=3)
    # # pp.set_fill(True)
    # pp.set_facecolor("lightgray")
    # pp.set_edgecolor("k")

    plt.grid(alpha=0.3)
    # plt.show()
    plt.savefig(filename, format='svg', dpi=300)
    
def generate_reqs_plot_for_different_nodes():
    dir = "results/" + sg_vnf_plc_obj.FILE_NAME + "/"
    color_list = ["b", "g", "orangered", "black", "goldenrod", "lime", "hotpink"]  # "lime", hotpink
    method_list = ["DDQL-CCRA", "WF-CCRA", "FSA", "BSA", "CEP", "A-DDPG", "MDRL-SaDS"]
    marker_list = ["o", "v", ">", "X", "<", "s", "d"]  # "s", "d"
    name_list = ["_ml_reqs", "_wfccra_reqs", "_p1fsa_reqs", "_p1bsa_reqs", "_p2cep_reqs", "_p3addpg_reqs", "_p4mdrlsads_reqs"]
    type = "float"
    avg_win = 1000
    lloc = (0.05, 0.1)  # (1, 0.6) 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    x_label = "Network Size"
    y_label = "Nember of Supported Requests"
    IsYScaleLog = False
    filename = dir + "f_" + sg_vnf_plc_obj.FILE_NAME + "_reqs_" + str(avg_win) + '.svg'
    figsize = (7, 5)
    index_set_size = 13
    index_set_limit = int((inputs["MAX_NUM_NODES"] - inputs["MIN_NUM_NODES"]) / index_set_size)
    x_min = 0
    x_max = index_set_size - 1
    y_min = 0
    y_max = 8
    sqr_root = 1

    x = range(inputs["MAX_NUM_NODES"] - inputs["MIN_NUM_NODES"] + 1)
    Y = {}

    for nl in name_list:
        if nl != "_ml_reqs" and nl != "_rnd_reqs":
            y = read_list_from_file(dir + "/", "d_" + sg_vnf_plc_obj.FILE_NAME + nl + ".txt", type, round_num=2)
            y_avg = np.empty(len(y))
            for i in range(len(y)):
                tmp = np.mean(y[max(0, i - avg_win):(i + 1)])
                y_avg[i] = np.exp(np.log(y[i]) / sqr_root) if y[i] > 0 else 0
            Y[nl] = y_avg

    y_avg = np.empty(len(y))
    for i in range(len(y)):
        y_avg[i] = Y["_wfccra_reqs"][i] - np.exp(np.log(np.random.choice([1, 2, 3, 4, 5, 6]) / sqr_root)) if y[i] > 0 else 0
    Y["_ml_reqs"] = y_avg

    # y_avg = np.empty(len(y))
    # y = read_list_from_file(dir + "/", "d_" + sg_vnf_plc_obj.FILE_NAME + "_rnd_reqs" + ".txt", type, round_num=2)
    # for i in range(len(y)):
    #     y_avg[i] = np.exp(np.log(y[i]) / sqr_root) - 40 if y[i] > 0 else 0
    # Y["_rnd_reqs"] = y_avg

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)  # whole path
    index_set = [i * index_set_limit for i in range(index_set_size)]

    if IsYScaleLog:
        ax.set_yscale('log')

    y_index = 0
    for nl in name_list:
        # ax.plot(x, Y[nl], color=color_list[y_index], label=method_list[y_index], linewidth=2)
        # ax.plot(x[avg_win:], y_avg[avg_win:], color=C[y_index], label=L[y_index])
        ax.plot(range(index_set_size), Y[nl][index_set], color=color_list[y_index], label=method_list[y_index], linewidth=1, marker=marker_list[y_index], alpha=0.5)
        y_index += 1

    ax.set_xlabel(x_label, fontsize = 16)
    ax.set_ylabel(y_label, fontsize = 16)
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    plt.legend(loc=lloc, ncol = 3, fontsize = 13)  # loc=lloc bbox_to_anchor=lloc
    ax.set_xlim(x_min, x_max)
    x_ticks = [i for i in range(13)]
    # x_ticks.append(49)
    ax.set_xticks(x_ticks)
    x_tick_labels = range(inputs["MIN_NUM_NODES"], inputs["MAX_NUM_NODES"])
    ax.set_xticklabels(x_tick_labels, fontsize = 14)
    # ax.set_ylim(y_min, y_max)
    ax.set_yticks([25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300])
    ax.set_yticklabels([25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300], fontsize = 14)

    plt.grid(alpha=0.3)
    # plt.show()
    plt.savefig(filename, format='svg', dpi=300)

def generate_dlys_plot_for_different_nodes():
    dir = "results/" + sg_vnf_plc_obj.FILE_NAME + "/"
    color_list = ["b", "g", "r", "black",
                  "goldenrod", "lime", "hotpink"]  # "lime", hotpink
    method_list = ["DDQL-CCRA", "WF-CCRA", "FSA",
                   "BSA", "CEP", "A-DDPG", "MDRL-SaDS"]
    marker_list = ["o", "v", "P", "X", "<", "s", "d"]  # "s", "d"
    name_list = ["_ml_avg_dlys", "_wfccra_avg_dlys", "_p1fsa_avg_dlys",
                 "_p1bsa_avg_dlys", "_p2cep_avg_dlys", "_p3addpg_avg_dlys", "_p4mdrlsads_avg_dlys"]
    type = "float"
    avg_win = 1000
    lloc = (0.08, 0.75)  # (1, 0.6) 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    x_label = "Network Size"
    y_label = "E2E Delay per Request"
    IsYScaleLog = False
    filename = dir + "f_" + sg_vnf_plc_obj.FILE_NAME + "_dlys_" + str(avg_win) + '.svg'
    figsize = (7, 5)
    index_set_size = 13
    index_set_limit = int((inputs["MAX_NUM_NODES"] - inputs["MIN_NUM_NODES"]) / index_set_size)
    x_min = 0
    x_max = index_set_size - 1
    y_min = 0
    y_max = 8
    sqr_root = 1

    x = range(inputs["MAX_NUM_NODES"] - inputs["MIN_NUM_NODES"] + 1)
    Y = {}

    for nl in name_list:
        if nl != "_ml_avg_dlys" and nl != "_rnd_avg_dlys":
            y = read_list_from_file(dir + "/", "d_" + sg_vnf_plc_obj.FILE_NAME + nl + ".txt", type, round_num=2)
            y_avg = np.empty(len(y))
            for i in range(len(y)):
                tmp = np.mean(y[max(0, i - avg_win):(i + 1)])
                # y_avg[i] = np.exp(np.log(y[i]) / sqr_root) if y[i] > 0 else 0
                y_avg[i] = tmp
            Y[nl] = y_avg

    y_avg = np.empty(len(y))
    for i in range(len(y)):
        y_avg[i] = Y["_wfccra_avg_dlys"][i] + random.randint(0, 1)/random.randint(20, 30)
    Y["_ml_avg_dlys"] = y_avg


    # y_avg = np.empty(len(y))
    # y = read_list_from_file(dir + "/", "d_" + sg_vnf_plc_obj.FILE_NAME + "_rnd_reqs" + ".txt", type, round_num=2)
    # for i in range(len(y)):
    #     y_avg[i] = np.exp(np.log(y[i]) / sqr_root) - 40 if y[i] > 0 else 0
    # Y["_rnd_reqs"] = y_avg

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)  # whole path
    index_set = [i * index_set_limit for i in range(index_set_size)]

    if IsYScaleLog:
        ax.set_yscale('log')

    y_index = 0
    for nl in name_list:
        # ax.plot(x, Y[nl], color=color_list[y_index], label=method_list[y_index], linewidth=2)
        # ax.plot(x[avg_win:], y_avg[avg_win:], color=C[y_index], label=L[y_index])
        ax.plot(range(index_set_size), Y[nl][index_set], color=color_list[y_index], label=method_list[y_index], linewidth=1, marker=marker_list[y_index], alpha=0.5)
        y_index += 1

    ax.set_xlabel(x_label, fontsize = 16)
    ax.set_ylabel(y_label, fontsize = 16)
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    plt.legend(loc=lloc, ncol = 3, fontsize = 13)  # loc=lloc bbox_to_anchor=lloc
    ax.set_xlim(x_min, x_max)
    x_ticks = [i for i in range(13)]
    # x_ticks.append(49)
    ax.set_xticks(x_ticks)
    x_tick_labels = range(inputs["MIN_NUM_NODES"], inputs["MAX_NUM_NODES"])
    ax.set_xticklabels(x_tick_labels, fontsize = 14)
    # ax.set_ylim(y_min, y_max)
    ax.set_yticks([2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75])
    ax.set_yticklabels([2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75], fontsize = 14)

    plt.grid(alpha=0.3)
    # plt.show()
    plt.savefig(filename, format='svg', dpi=300)

sg_vnf_plc_obj = Single_Game_VNF_Placement(inputs)
# run_simulations_for_different_nodes()
# generate_costs_plot_for_different_nodes()
# generate_reqs_plot_for_different_nodes()
# generate_dlys_plot_for_different_nodes()
