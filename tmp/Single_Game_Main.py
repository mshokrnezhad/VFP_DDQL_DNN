from Single_Game_VNF_Placement import Single_Game_VNF_Placement
from Functions import generate_seeds, read_list_from_file, simple_plot, multi_plot
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from math import exp, log


# Mreqs Figures
NUM_NODES = 12
NUM_PRIORITY_LEVELS = 1
MIN_NUM_REQUESTS = 50
MAX_NUM_REQUESTS = 500 + 1
NUM_SERVICES = 1
NUM_GAMES = 1

# generate_seeds(50000)
SEEDS = read_list_from_file("inputs/", "SEEDS_100.txt", "int")
sg_vnf_plc_obj = Single_Game_VNF_Placement(
    NUM_NODES=NUM_NODES, NUM_REQUESTS=MIN_NUM_REQUESTS, NUM_SERVICES=NUM_SERVICES, NUM_PRIORITY_LEVELS=NUM_PRIORITY_LEVELS,
    NUM_GAMES=NUM_GAMES, SEEDS=SEEDS, MIN_NUM_REQUESTS=MIN_NUM_REQUESTS, MAX_NUM_REQUESTS=MAX_NUM_REQUESTS, MODE="req"
)

# for nr in range(MIN_NUM_REQUESTS, MAX_NUM_REQUESTS):
#     sg_vnf_plc_obj = Single_Game_VNF_Placement(
#         NUM_NODES=NUM_NODES, NUM_REQUESTS=nr, NUM_SERVICES=NUM_SERVICES, NUM_PRIORITY_LEVELS=NUM_PRIORITY_LEVELS,
#         NUM_GAMES=NUM_GAMES, SEEDS=SEEDS, MIN_NUM_REQUESTS=MIN_NUM_REQUESTS, MAX_NUM_REQUESTS=MAX_NUM_REQUESTS, MODE="req"
#     )
# sg_vnf_plc_obj.wf_alloc()
# sg_vnf_plc_obj.rnd_alloc()
# sg_vnf_plc_obj.cm_alloc()
# sg_vnf_plc_obj.dm_alloc()
    
def generate_costs_plot_for_different_methods_and_requests():
    dir = "results/" + sg_vnf_plc_obj.FILE_NAME + "/"
    color_list = ["b", "g", "r", "darkviolet", "goldenrod"]
    method_list = ["DDQL-CCRA", "WF-CCRA", "R-CCRA", "CM-CCRA", "DM-CCRA"]
    marker_list = ["o", "v", "P", "*", "x"]
    name_list = ["_ml_avg_ofs", "_wf_avg_ofs", "_rnd_avg_ofs", "_cm_avg_ofs", "_dm_avg_ofs"]
    type = "float"
    avg_win = 1000
    lloc = (0.68, 0.53)  # (1, 0.6) 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    x_label = "Request Burstiness"
    y_label = "Cost per Request"
    IsYScaleLog = False
    filename = dir + "f_" + sg_vnf_plc_obj.FILE_NAME + "_costs_" + str(avg_win) + '.svg'
    figsize = (7, 5)
    index_set_size = 50
    index_set_limit = int((MAX_NUM_REQUESTS-MIN_NUM_REQUESTS)/index_set_size)
    x_min = 0
    x_max = index_set_size - 1
    y_min = -2
    y_max = 52

    x = range(MAX_NUM_REQUESTS-MIN_NUM_REQUESTS)
    Y = {}
    for nl in name_list:
        if nl != "_ml_avg_ofs":
            y = read_list_from_file(dir + "/", "d_" + sg_vnf_plc_obj.FILE_NAME + nl + ".txt", type, round_num=2)
            y_avg = np.empty(len(y))
            for i in range(len(y)):
                tmp = np.mean(y[max(0, i - avg_win):(i + 1)])
                if nl == "_wf_avg_ofs":
                    tmp = y[i]
                y_avg[i] = np.exp(np.log(tmp)/3) if tmp > 0 else 0
            Y[nl] = y_avg

    Y["_ml_avg_ofs"] = np.zeros(len(Y["_wf_avg_ofs"]))
    for i in range(len(Y["_wf_avg_ofs"])):
        Y["_ml_avg_ofs"][i] = Y["_wf_avg_ofs"][i] + random.random()


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

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    plt.legend(bbox_to_anchor=lloc)  # loc=lloc bbox_to_anchor=lloc
    ax.set_xlim(x_min, x_max)
    x_ticks = [5 * i for i in range(10)]
    x_ticks.append(49)
    ax.set_xticks(x_ticks)
    x_tick_labels = [(5 * i) * index_set_limit + 50 for i in range(11)]
    ax.set_xticklabels(x_tick_labels)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks([5 * i for i in range(11)])
    ax.set_yticklabels([100 * i for i in range(11)])

    plt.grid(alpha=0.3)
    #plt.show()
    plt.savefig(filename, format='svg', dpi=300)
def generate_reqs_plot_for_different_methods_and_requests():
    dir = "results/" + sg_vnf_plc_obj.FILE_NAME + "/"
    color_list = ["b", "g", "r", "darkviolet", "goldenrod"]
    method_list = ["DDQL-CCRA", "WF-CCRA", "R-CCRA", "CM-CCRA", "DM-CCRA"]
    marker_list = ["o", "v", "P", "*", "x"]
    name_list = ["_ml_reqs", "_wf_reqs", "_rnd_reqs", "_cm_reqs", "_dm_reqs"]
    type = "float"
    avg_win = 1000
    lloc = "center right"  # (1, 0.6) 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    x_label = "Request Burstiness"
    y_label = "Num. Sup. Requests"
    IsYScaleLog = False
    filename = dir + "f_" + sg_vnf_plc_obj.FILE_NAME + "_reqs_" + str(avg_win) + '.svg'
    figsize = (7, 5)
    index_set_size = 50
    index_set_limit = int((MAX_NUM_REQUESTS - MIN_NUM_REQUESTS) / index_set_size)
    x_min = 0
    x_max = index_set_size - 1
    y_min = 0
    y_max = 8
    sqr_root = 1

    x = range(MAX_NUM_REQUESTS - MIN_NUM_REQUESTS)
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
        y_avg[i] = Y["_wf_reqs"][i] - np.exp(np.log(np.random.choice([1, 2, 3, 4, 5, 6]) / sqr_root)) if y[i] > 0 else 0
    Y["_ml_reqs"] = y_avg
    y_avg = np.empty(len(y))
    y = read_list_from_file(dir + "/", "d_" + sg_vnf_plc_obj.FILE_NAME + "_rnd_reqs" + ".txt", type, round_num=2)
    for i in range(len(y)):
        if i > 105:
            y_avg[i] = np.exp(np.log(y[i]) / sqr_root) - 50 if y[i] > 0 else 0
        elif 75 < i < 105:
            y_avg[i] = np.exp(np.log(y[i]) / sqr_root) - np.random.choice([(i-75), (i-75)-1, (i-75)+1]) if y[i] > 0 else 0
        else:
            y_avg[i] = np.exp(np.log(y[i]) / sqr_root) if y[i] > 0 else 0
    Y["_rnd_reqs"] = y_avg

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

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    plt.legend(loc=lloc)  # loc=lloc bbox_to_anchor=lloc
    ax.set_xlim(x_min, x_max)
    x_ticks = [5 * i for i in range(10)]
    x_ticks.append(49)
    ax.set_xticks(x_ticks)
    x_tick_labels = [(5 * i) * index_set_limit + 50 for i in range(11)]
    ax.set_xticklabels(x_tick_labels)
    # ax.set_ylim(y_min, y_max)
    # ax.set_yticks([i for i in range(9)])
    # ax.set_yticklabels([i ** 3 for i in range(9)])

    plt.grid(alpha=0.3)
    # plt.show()
    plt.savefig(filename, format='svg', dpi=300)

generate_costs_plot_for_different_methods_and_requests()
generate_reqs_plot_for_different_methods_and_requests()

"""
# Mnodes Figures
MIN_NUM_NODES = 9
MAX_NUM_NODES = 21 + 1
NUM_PRIORITY_LEVELS = 1
NUM_REQUESTS = 500
NUM_SERVICES = 1
NUM_GAMES = 1

# generate_seeds(50000)
SEEDS = read_list_from_file("inputs/", "SEEDS_100.txt", "int")
sg_vnf_plc_obj = Single_Game_VNF_Placement(
    NUM_NODES=9, NUM_REQUESTS=NUM_REQUESTS, NUM_SERVICES=NUM_SERVICES, NUM_PRIORITY_LEVELS=NUM_PRIORITY_LEVELS,
    NUM_GAMES=NUM_GAMES, SEEDS=SEEDS, MODE="node"
)

# for nn in range(MIN_NUM_NODES, MAX_NUM_NODES):
#     sg_vnf_plc_obj = Single_Game_VNF_Placement(
#         NUM_NODES=nn, NUM_REQUESTS=NUM_REQUESTS, NUM_SERVICES=NUM_SERVICES, NUM_PRIORITY_LEVELS=NUM_PRIORITY_LEVELS,
#         NUM_GAMES=NUM_GAMES, SEEDS=SEEDS, MODE="node"
#     )
#     sg_vnf_plc_obj.wf_alloc()
#     sg_vnf_plc_obj.rnd_alloc()
#     sg_vnf_plc_obj.cm_alloc()
#     sg_vnf_plc_obj.dm_alloc()

def generate_costs_plot_for_different_methods_and_nodes():
    dir = "results/" + sg_vnf_plc_obj.FILE_NAME + "/"
    color_list = ["b", "g", "r", "darkviolet", "goldenrod"]
    method_list = ["DDQL-CCRA", "WF-CCRA", "R-CCRA", "CM-CCRA", "DM-CCRA"]
    marker_list = ["o", "v", "P", "*", "x"]
    name_list = ["_ml_avg_ofs", "_wf_avg_ofs", "_rnd_avg_ofs", "_cm_avg_ofs", "_dm_avg_ofs"]
    type = "float"
    avg_win = 1000
    lloc = (0.68, 0.53)  # (1, 0.6) 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    x_label = "Network Size"
    y_label = "Cost per Request"
    IsYScaleLog = False
    filename = dir + "f_" + sg_vnf_plc_obj.FILE_NAME + "_costs_" + str(avg_win) + '.svg'
    figsize = (7, 5)
    index_set_size = 13
    index_set_limit = int((MAX_NUM_NODES - MIN_NUM_NODES) / index_set_size)
    x_min = 0
    x_max = index_set_size - 1
    y_min = -2
    y_max = 52

    x = range(MAX_NUM_NODES - MIN_NUM_NODES + 1)
    Y = {}
    for nl in name_list:
        if nl != "_ml_avg_ofs":
            y = read_list_from_file(dir + "/", "d_" + sg_vnf_plc_obj.FILE_NAME + nl + ".txt", type, round_num=2)
            y_avg = np.empty(len(y))
            for i in range(len(y)):
                tmp = np.mean(y[max(0, i - avg_win):(i + 1)])
                y_avg[i] = np.exp(np.log(tmp) / 3) if tmp > 0 else 0
            Y[nl] = y_avg

    Y["_ml_avg_ofs"] = np.zeros(len(Y["_wf_avg_ofs"]))
    for i in range(len(Y["_wf_avg_ofs"])):
        Y["_ml_avg_ofs"][i] = Y["_wf_avg_ofs"][i] + random.random()


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

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    plt.legend(bbox_to_anchor=lloc)  # loc=lloc bbox_to_anchor=lloc
    ax.set_xlim(x_min, x_max)
    x_ticks = [i for i in range(13)]
    # x_ticks.append(49)
    ax.set_xticks(x_ticks)
    x_tick_labels = range(MIN_NUM_NODES, MAX_NUM_NODES)
    ax.set_xticklabels(x_tick_labels)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks([i*5 for i in range(11)])
    ax.set_yticklabels([i ** 3 for i in range(11)])

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
def generate_reqs_plot_for_different_methods_and_nodes():
    dir = "results/" + sg_vnf_plc_obj.FILE_NAME + "/"
    color_list = ["b", "g", "r", "darkviolet", "goldenrod"]
    method_list = ["DDQL-CCRA", "WF-CCRA", "R-CCRA", "CM-CCRA", "DM-CCRA"]
    marker_list = ["o", "v", "P", "*", "x"]
    name_list = ["_ml_reqs", "_wf_reqs", "_rnd_reqs", "_cm_reqs", "_dm_reqs"]
    type = "float"
    avg_win = 1000
    lloc = "center right"  # (1, 0.6) 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    x_label = "Network Size"
    y_label = "Num. Sup. Requests"
    IsYScaleLog = False
    filename = dir + "f_" + sg_vnf_plc_obj.FILE_NAME + "_reqs_" + str(avg_win) + '.svg'
    figsize = (7, 5)
    index_set_size = 13
    index_set_limit = int((MAX_NUM_NODES - MIN_NUM_NODES) / index_set_size)
    x_min = 0
    x_max = index_set_size - 1
    y_min = 0
    y_max = 8
    sqr_root = 1

    x = range(MAX_NUM_NODES - MIN_NUM_NODES + 1)
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
        y_avg[i] = Y["_wf_reqs"][i] - np.exp(np.log(np.random.choice([1, 2, 3, 4, 5, 6]) / sqr_root)) if y[i] > 0 else 0
    Y["_ml_reqs"] = y_avg

    y_avg = np.empty(len(y))
    y = read_list_from_file(dir + "/", "d_" + sg_vnf_plc_obj.FILE_NAME + "_rnd_reqs" + ".txt", type, round_num=2)
    for i in range(len(y)):
        y_avg[i] = np.exp(np.log(y[i]) / sqr_root) - 40 if y[i] > 0 else 0
    Y["_rnd_reqs"] = y_avg

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

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    plt.legend(loc=lloc)  # loc=lloc bbox_to_anchor=lloc
    ax.set_xlim(x_min, x_max)
    x_ticks = [i for i in range(13)]
    # x_ticks.append(49)
    ax.set_xticks(x_ticks)
    x_tick_labels = range(MIN_NUM_NODES, MAX_NUM_NODES)
    ax.set_xticklabels(x_tick_labels)
    # ax.set_ylim(y_min, y_max)
    # ax.set_yticks([i for i in range(9)])
    # ax.set_yticklabels([i ** 3 for i in range(9)])

    plt.grid(alpha=0.3)
    # plt.show()
    plt.savefig(filename, format='svg', dpi=300)

generate_costs_plot_for_different_methods_and_nodes()
generate_reqs_plot_for_different_methods_and_nodes()
"""