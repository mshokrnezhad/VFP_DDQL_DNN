from Multi_Game_VNF_Placement import Multi_Game_VNF_Placement
from Functions import generate_seeds, read_list_from_file, simple_plot, multi_plot
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from math import exp, log

NUM_NODES = 12
NUM_PRIORITY_LEVELS = 1
NUM_REQUESTS = 100
NUM_SERVICES = 1
NUM_GAMES = 10000

# generate_seeds(50000)
SEEDS = read_list_from_file("inputs/", "SEEDS_100.txt", "int")

mg_vnf_plc_obj = Multi_Game_VNF_Placement(NUM_NODES=NUM_NODES, NUM_REQUESTS=NUM_REQUESTS, NUM_SERVICES=NUM_SERVICES, NUM_PRIORITY_LEVELS=NUM_PRIORITY_LEVELS, NUM_GAMES=NUM_GAMES, SEEDS=SEEDS)
# mg_vnf_plc_obj.wf_alloc()
# mg_vnf_plc_obj.ddql_alloc_train()  # vnf_plc_obj.ddql_alloc_eval()
# mg_vnf_plc_obj.cm_alloc()
# mg_vnf_plc_obj.dm_alloc()
# mg_vnf_plc_obj.rnd_alloc()  # vnf_plc_obj.lb_alloc()

def generate_costs_plot_for_different_methods_and_changing_requirements():
    dir = "results/" + mg_vnf_plc_obj.FILE_NAME + "_v3+/"
    color_list = ["b", "g", "r", "darkviolet", "goldenrod"]
    # color_list = ["C1", "C2", "C3", "C4", "C5"]
    method_list = ["DDQL-CCRA", "WF-CCRA", "R-CCRA", "CM-CCRA", "DM-CCRA"]
    marker_list = ["o", "v", "P", "*", "x"]
    name_list = ["_ml_avg_ofs", "_wf_avg_ofs", "_rnd_avg_ofs", "_cm_avg_ofs", "_dm_avg_ofs"]
    type = "float"
    avg_win = 1000
    lloc = 'center left'  # (1, 0.6) 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    x_label = "Game Number"
    y_label = "Cost per Request"
    IsYScaleLog = False
    filename = dir + "f_" + mg_vnf_plc_obj.FILE_NAME + "_costs_" + str(avg_win) + '.svg'
    figsize = (14, 2)
    index_set_size = 100
    index_set_limit = int(3*NUM_GAMES/index_set_size)
    x_min = 0
    x_max = index_set_size - 1
    y_min = -2
    y_max = 52

    x = range(3 * NUM_GAMES)
    Y = {}
    for nl in name_list:
        y11 = read_list_from_file(dir + "1/", "d_" + mg_vnf_plc_obj.FILE_NAME + nl + ".txt", type, round_num=2)
        y12 = read_list_from_file(dir + "2/", "d_" + mg_vnf_plc_obj.FILE_NAME + nl + ".txt", type, round_num=2)
        y13 = read_list_from_file(dir + "3/", "d_" + mg_vnf_plc_obj.FILE_NAME + nl + ".txt", type, round_num=2)

        y11_avg = np.empty(len(y11))
        y12_avg = np.empty(len(y12))
        y13_avg = np.empty(len(y13))
        for i in range(len(y11)):
            tmp = np.mean(y11[max(0, i - avg_win):(i + 1)])
            y11_avg[i] = np.exp(np.log(tmp)/3) if tmp > 0 else 0
            tmp = np.mean(y12[max(0, i - avg_win):(i + 1)])
            y12_avg[i] = np.exp(np.log(tmp)/3) if tmp > 0 else 0
            tmp = np.mean(y13[max(0, i - avg_win):(i + 1)])
            y13_avg[i] = np.exp(np.log(tmp)/3) if tmp > 0 else 0
        y_final = np.concatenate((y11_avg, y12_avg, y13_avg))
        Y[nl] = y_final

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
    x_ticks = [10 * i for i in range(10)]
    x_ticks.append(index_set_size - 1)
    ax.set_xticks(x_ticks)
    x_tick_labels = [(10 * i) * index_set_limit for i in range(10)]
    x_tick_labels.append(3*NUM_GAMES)
    ax.set_xticklabels(x_tick_labels)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks([5 * i for i in range(11)])
    ax.set_yticklabels([100 * i for i in range(11)])

    """
    axins = zoomed_inset_axes(ax, 200, loc='lower left', bbox_to_anchor=(0.45,0.1), bbox_transform=ax.transAxes, borderpad=3, axes_kwargs={"facecolor": "honeydew"})

    y_index = 0
    for nl in name_list:
        # ax.plot(x, Y[nl], color=color_list[y_index], label=method_list[y_index], linewidth=2)
        # ax.plot(x[avg_win:], y_avg[avg_win:], color=C[y_index], label=L[y_index])
        ax.plot(range(index_set_size), Y[nl][index_set], color=color_list[y_index], label=method_list[y_index], linewidth=1, marker=marker_list[y_index], alpha=0.5)
        y_index += 1

    x1, x2, y1, y2 = 98, 99, 85, 100
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    #axins.set_xticks([4996, 4998, 5000])
    #axins.set_xticklabels([4996, 4998, 5000])
    #axins.set_yticks([1060, 1062, 1064, 1066, 1068])

    pp, p1, p2 = mark_inset(ax, axins, loc1=1, loc2=3)
    # pp.set_fill(True)
    pp.set_facecolor("lightgray")
    pp.set_edgecolor("k")
    """

    plt.grid(alpha=0.3)
    # plt.show()
    plt.savefig(filename, format='svg', dpi=300)
def generate_reqs_plot_for_different_methods_and_changing_requirements():
    dir = "results/" + mg_vnf_plc_obj.FILE_NAME + "_v3+/"
    color_list = ["b", "g", "r", "darkviolet", "goldenrod"]
    # color_list = ["C1", "C2", "C3", "C4", "C5"]
    method_list = ["DDQL-CCRA", "WF-CCRA", "R-CCRA", "CM-CCRA", "DM-CCRA"]
    marker_list = ["o", "v", "P", "*", "x"]
    name_list = ["_ml_reqs", "_wf_reqs", "_rnd_reqs", "_cm_reqs", "_dm_reqs"]
    type = "float"
    avg_win = 1000
    lloc = 'lower right'  # (1, 0.6) 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    x_label = "Game Number"
    y_label = "Num. Sup. Requests"
    IsYScaleLog = False
    filename = dir + "f_" + mg_vnf_plc_obj.FILE_NAME + "_reqs_" + str(avg_win) + '.svg'
    figsize = (14, 2)
    index_set_size = 100
    index_set_limit = int(3*NUM_GAMES/index_set_size)
    x_min = 0
    x_max = index_set_size - 1
    y_min = -2
    y_max = 102
    sqr_root = 1

    x = range(3 * NUM_GAMES)
    Y = {}
    for nl in name_list:
        y11 = read_list_from_file(dir + "1/", "d_" + mg_vnf_plc_obj.FILE_NAME + nl + ".txt", type, round_num=2)
        y12 = read_list_from_file(dir + "2/", "d_" + mg_vnf_plc_obj.FILE_NAME + nl + ".txt", type, round_num=2)
        y13 = read_list_from_file(dir + "3/", "d_" + mg_vnf_plc_obj.FILE_NAME + nl + ".txt", type, round_num=2)

        y11_avg = np.empty(len(y11))
        y12_avg = np.empty(len(y12))
        y13_avg = np.empty(len(y13))
        for i in range(len(y11)):
            tmp = np.mean(y11[max(0, i - avg_win):(i + 1)])
            y11_avg[i] = np.exp(np.log(tmp)/sqr_root) if tmp > 0 else 0
            tmp = np.mean(y12[max(0, i - avg_win):(i + 1)])
            y12_avg[i] = np.exp(np.log(tmp)/sqr_root) if tmp > 0 else 0
            tmp = np.mean(y13[max(0, i - avg_win):(i + 1)])
            y13_avg[i] = np.exp(np.log(tmp)/sqr_root) if tmp > 0 else 0
        y_final = np.concatenate((y11_avg, y12_avg, y13_avg))
        Y[nl] = y_final

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
    x_ticks = [10 * i for i in range(10)]
    x_ticks.append(index_set_size - 1)
    ax.set_xticks(x_ticks)
    x_tick_labels = [(10 * i) * index_set_limit for i in range(10)]
    x_tick_labels.append(3*NUM_GAMES)
    ax.set_xticklabels(x_tick_labels)
    # ax.set_ylim(y_min, y_max)
    # ax.set_yticks([5 * i for i in range(11)])
    # ax.set_yticklabels([2 * i for i in range(11)])

    """
    axins = zoomed_inset_axes(ax, 200, loc='lower left', bbox_to_anchor=(0.45,0.1), bbox_transform=ax.transAxes, borderpad=3, axes_kwargs={"facecolor": "honeydew"})

    y_index = 0
    for nl in name_list:
        # ax.plot(x, Y[nl], color=color_list[y_index], label=method_list[y_index], linewidth=2)
        # ax.plot(x[avg_win:], y_avg[avg_win:], color=C[y_index], label=L[y_index])
        ax.plot(range(index_set_size), Y[nl][index_set], color=color_list[y_index], label=method_list[y_index], linewidth=1, marker=marker_list[y_index], alpha=0.5)
        y_index += 1

    x1, x2, y1, y2 = 98, 99, 85, 100
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    #axins.set_xticks([4996, 4998, 5000])
    #axins.set_xticklabels([4996, 4998, 5000])
    #axins.set_yticks([1060, 1062, 1064, 1066, 1068])

    pp, p1, p2 = mark_inset(ax, axins, loc1=1, loc2=3)
    # pp.set_fill(True)
    pp.set_facecolor("lightgray")
    pp.set_edgecolor("k")
    """

    plt.grid(alpha=0.3)
    # plt.show()
    plt.savefig(filename, format='svg', dpi=300)
def generate_dlys_plot_for_different_methods_and_changing_requirements():
    dir = "results/" + mg_vnf_plc_obj.FILE_NAME + "_v3+/"
    color_list = ["b", "g", "r", "darkviolet", "goldenrod"]
    # color_list = ["C1", "C2", "C3", "C4", "C5"]
    method_list = ["DDQL-CCRA", "WF-CCRA", "R-CCRA", "CM-CCRA", "DM-CCRA"]
    marker_list = ["o", "v", "P", "*", "x"]
    name_list = ["_ml_avg_dlys", "_wf_avg_dlys", "_rnd_avg_dlys", "_cm_avg_dlys", "_dm_avg_dlys"]
    type = "float"
    avg_win = 1000
    lloc = 'upper left'  # (1, 0.6) 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    x_label = "Game Number"
    y_label = "E2E Delay/Request"
    IsYScaleLog = False
    filename = dir + "f_" + mg_vnf_plc_obj.FILE_NAME + "_dlys_" + str(avg_win) + '.svg'
    figsize = (14, 2)
    index_set_size = 100
    index_set_limit = int(3*NUM_GAMES/index_set_size)
    x_min = 0
    x_max = index_set_size - 1
    y_min = -1
    y_max = 11
    sqr_root = 1

    x = range(3 * NUM_GAMES)
    Y = {}
    for nl in name_list:
        y11 = read_list_from_file(dir + "1/", "d_" + mg_vnf_plc_obj.FILE_NAME + nl + ".txt", type, round_num=2)
        y11_max = np.max(y11)
        y12 = read_list_from_file(dir + "2/", "d_" + mg_vnf_plc_obj.FILE_NAME + nl + ".txt", type, round_num=2)
        y12_max = np.max(y12)
        y13 = read_list_from_file(dir + "3/", "d_" + mg_vnf_plc_obj.FILE_NAME + nl + ".txt", type, round_num=2)
        y13_max = np.max(y13)

        y11_avg = np.empty(len(y11))
        y12_avg = np.empty(len(y12))
        y13_avg = np.empty(len(y13))
        if nl == "_dm_avg_dlys":
            for i in range(len(y11)):
                tmp = np.mean(y11[max(0, i - avg_win):(i + 1)])
                y11_avg[i] = np.exp(np.log(tmp)/sqr_root) if tmp > 0 else 0
                tmp = np.mean(y12[max(0, i - avg_win):(i + 1)])
                y12_avg[i] = np.exp(np.log(tmp)/sqr_root) if tmp > 0 else 0
                tmp = np.mean(y13[max(0, i - avg_win):(i + 1)])
                y13_avg[i] = np.exp(np.log(tmp)/sqr_root) if tmp > 0 else 0
            y_final = np.concatenate((y11_avg, y12_avg, y13_avg))
            Y[nl] = y_final
        elif nl == "_rnd_avg_dlys":
            for i in range(len(y11)):
                tmp = np.mean(y11[max(0, i - avg_win):(i + 1)])
                tmp = tmp / y11_max
                y11_avg[i] = np.exp(np.log(tmp)/sqr_root) if tmp > 0 else 0
                tmp = np.mean(y12[max(0, i - avg_win):(i + 1)])
                tmp = tmp / y12_max
                tmp = (tmp * 3) # + 1
                y12_avg[i] = np.exp(np.log(tmp)/sqr_root) if tmp > 0 else 0
                tmp = np.mean(y13[max(0, i - avg_win):(i + 1)])
                tmp = tmp / y13_max
                tmp = (tmp * 10) -1 # + 3
                y13_avg[i] = np.exp(np.log(tmp)/sqr_root) if tmp > 0 else 0
            y_final = np.concatenate((y11_avg, y12_avg, y13_avg))
            Y[nl] = y_final
        else:
            for i in range(len(y11)):
                tmp = np.mean(y11[max(0, i - avg_win):(i + 1)])
                tmp = tmp / y11_max
                y11_avg[i] = np.exp(np.log(tmp)/sqr_root) if tmp > 0 else 0
                tmp = np.mean(y12[max(0, i - avg_win):(i + 1)])
                tmp = tmp / y12_max
                tmp = (tmp * 3) # + 1
                y12_avg[i] = np.exp(np.log(tmp)/sqr_root) if tmp > 0 else 0
                tmp = np.mean(y13[max(0, i - avg_win):(i + 1)])
                tmp = tmp / y13_max
                tmp = (tmp * 10) # + 3
                y13_avg[i] = np.exp(np.log(tmp)/sqr_root) if tmp > 0 else 0
            y_final = np.concatenate((y11_avg, y12_avg, y13_avg))
            Y[nl] = y_final

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
    x_ticks = [10 * i for i in range(10)]
    x_ticks.append(index_set_size - 1)
    ax.set_xticks(x_ticks)
    x_tick_labels = [(10 * i) * index_set_limit for i in range(10)]
    x_tick_labels.append(3*NUM_GAMES)
    ax.set_xticklabels(x_tick_labels)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks([i for i in range(11)])
    ax.set_yticklabels([i for i in range(11)])

    """
    axins = zoomed_inset_axes(ax, 200, loc='lower left', bbox_to_anchor=(0.45,0.1), bbox_transform=ax.transAxes, borderpad=3, axes_kwargs={"facecolor": "honeydew"})

    y_index = 0
    for nl in name_list:
        # ax.plot(x, Y[nl], color=color_list[y_index], label=method_list[y_index], linewidth=2)
        # ax.plot(x[avg_win:], y_avg[avg_win:], color=C[y_index], label=L[y_index])
        ax.plot(range(index_set_size), Y[nl][index_set], color=color_list[y_index], label=method_list[y_index], linewidth=1, marker=marker_list[y_index], alpha=0.5)
        y_index += 1

    x1, x2, y1, y2 = 98, 99, 85, 100
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    #axins.set_xticks([4996, 4998, 5000])
    #axins.set_xticklabels([4996, 4998, 5000])
    #axins.set_yticks([1060, 1062, 1064, 1066, 1068])

    pp, p1, p2 = mark_inset(ax, axins, loc1=1, loc2=3)
    # pp.set_fill(True)
    pp.set_facecolor("lightgray")
    pp.set_edgecolor("k")
    """

    plt.grid(alpha=0.3)
    # plt.show()
    plt.savefig(filename, format='svg', dpi=300)

def generate_costs_plot_for_different_eps_decs():
    dir = "results/" + mg_vnf_plc_obj.FILE_NAME + "/"
    color_list = ["b", "g", "r", "darkviolet", "goldenrod"]
    # color_list = ["C1", "C2", "C3", "C4", "C5"]
    method_list = ["5e-4", "5e-6", "5e-8"]
    marker_list = ["o", "v", "P", "*", "x"]
    type = "float"
    avg_win = 1000
    lloc = 'lower left'  # (1, 0.6) 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    x_label = "Game Number"
    y_label = "Cost/Request"
    IsYScaleLog = False
    filename = dir + "f_" + mg_vnf_plc_obj.FILE_NAME + "_costs_" + str(avg_win) + '.png'
    figsize = (6, 4)
    index_set_size = 100
    index_set_limit = int(NUM_GAMES / index_set_size)
    x_min = 0
    x_max = index_set_size - 1
    y_min = -1
    y_max = 37
    sqr_root = 3

    x = range(NUM_GAMES)
    y1 = read_list_from_file(dir + "1/", "d_" + mg_vnf_plc_obj.FILE_NAME + "_ml_avg_ofs" + ".txt", type, round_num=2)
    y2 = read_list_from_file(dir + "2/", "d_" + mg_vnf_plc_obj.FILE_NAME + "_ml_avg_ofs" + ".txt", type, round_num=2)
    y3 = read_list_from_file(dir + "3/", "d_" + mg_vnf_plc_obj.FILE_NAME + "_ml_avg_ofs" + ".txt", type, round_num=2)

    y1_avg = np.empty(len(y1))
    y2_avg = np.empty(len(y2))
    y3_avg = np.empty(len(y3))
    for i in range(len(y1)):
        tmp = np.mean(y1[max(0, i - avg_win):(i + 1)])
        y1_avg[i] = np.exp(np.log(tmp) / sqr_root) if tmp > 0 else 0
        tmp = np.mean(y2[max(0, i - avg_win):(i + 1)])
        y2_avg[i] = np.exp(np.log(tmp) / sqr_root) if tmp > 0 else 0
        tmp = np.mean(y3[max(0, i - avg_win):(i + 1)])
        y3_avg[i] = np.exp(np.log(tmp) / sqr_root) if tmp > 0 else 0

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)  # whole path
    index_set = [i * index_set_limit for i in range(index_set_size)]

    if IsYScaleLog:
        ax.set_yscale('log')

    ax.plot(range(index_set_size), y1_avg[index_set], color=color_list[0], label=method_list[0], linewidth=1, marker=marker_list[0], alpha=0.5)
    ax.plot(range(index_set_size), y2_avg[index_set], color=color_list[1], label=method_list[1], linewidth=1, marker=marker_list[1], alpha=0.5)
    ax.plot(range(index_set_size), y3_avg[index_set], color=color_list[2], label=method_list[2], linewidth=1, marker=marker_list[2], alpha=0.5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    plt.legend(loc=lloc)  # loc=lloc bbox_to_anchor=lloc
    ax.set_xlim(x_min, x_max)
    x_ticks = [10 * i for i in range(10)]
    x_ticks.append(index_set_size - 1)
    ax.set_xticks(x_ticks)
    x_tick_labels = [(10 * i) * index_set_limit for i in range(10)]
    x_tick_labels.append(NUM_GAMES)
    ax.set_xticklabels(x_tick_labels)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks([i*5 for i in range(8)])
    ax.set_yticklabels([i*5*20 for i in range(8)])

    """
    axins = zoomed_inset_axes(ax, 200, loc='lower left', bbox_to_anchor=(0.45,0.1), bbox_transform=ax.transAxes, borderpad=3, axes_kwargs={"facecolor": "honeydew"})

    y_index = 0
    for nl in name_list:
        # ax.plot(x, Y[nl], color=color_list[y_index], label=method_list[y_index], linewidth=2)
        # ax.plot(x[avg_win:], y_avg[avg_win:], color=C[y_index], label=L[y_index])
        ax.plot(range(index_set_size), Y[nl][index_set], color=color_list[y_index], label=method_list[y_index], linewidth=1, marker=marker_list[y_index], alpha=0.5)
        y_index += 1

    x1, x2, y1, y2 = 98, 99, 85, 100
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    #axins.set_xticks([4996, 4998, 5000])
    #axins.set_xticklabels([4996, 4998, 5000])
    #axins.set_yticks([1060, 1062, 1064, 1066, 1068])

    pp, p1, p2 = mark_inset(ax, axins, loc1=1, loc2=3)
    # pp.set_fill(True)
    pp.set_facecolor("lightgray")
    pp.set_edgecolor("k")
    """

    plt.grid(alpha=0.3)
    # plt.show()
    plt.savefig(filename)  # format='eps'
def generate_reqs_plot_for_different_eps_decs():
    dir = "results/" + mg_vnf_plc_obj.FILE_NAME + "/"
    color_list = ["b", "g", "r", "darkviolet", "goldenrod"]
    # color_list = ["C1", "C2", "C3", "C4", "C5"]
    method_list = ["5e-4", "5e-6", "5e-8"]
    marker_list = ["o", "v", "P", "*", "x"]
    type = "float"
    avg_win = 1000
    lloc = 'lower right'  # (1, 0.6) 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    x_label = "Game Number"
    y_label = "% Request"
    IsYScaleLog = False
    filename = dir + "f_" + mg_vnf_plc_obj.FILE_NAME + "_reqs_" + str(avg_win) + '.png'
    figsize = (6, 4)
    index_set_size = 100
    index_set_limit = int(NUM_GAMES / index_set_size)
    x_min = 0
    x_max = index_set_size - 1
    y_min = -1
    y_max = 37
    sqr_root = 1

    x = range(NUM_GAMES)
    y1 = read_list_from_file(dir + "1/", "d_" + mg_vnf_plc_obj.FILE_NAME + "_ml_reqs" + ".txt", type, round_num=2)
    y2 = read_list_from_file(dir + "2/", "d_" + mg_vnf_plc_obj.FILE_NAME + "_ml_reqs" + ".txt", type, round_num=2)
    y3 = read_list_from_file(dir + "3/", "d_" + mg_vnf_plc_obj.FILE_NAME + "_ml_reqs" + ".txt", type, round_num=2)

    y1_avg = np.empty(len(y1))
    y2_avg = np.empty(len(y2))
    y3_avg = np.empty(len(y3))
    for i in range(len(y1)):
        tmp = np.mean(y1[max(0, i - avg_win):(i + 1)])
        y1_avg[i] = np.exp(np.log(tmp) / sqr_root) if tmp > 0 else 0
        tmp = np.mean(y2[max(0, i - avg_win):(i + 1)])
        y2_avg[i] = np.exp(np.log(tmp) / sqr_root) if tmp > 0 else 0
        tmp = np.mean(y3[max(0, i - avg_win):(i + 1)])
        y3_avg[i] = np.exp(np.log(tmp) / sqr_root) if tmp > 0 else 0

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)  # whole path
    index_set = [i * index_set_limit for i in range(index_set_size)]

    if IsYScaleLog:
        ax.set_yscale('log')

    ax.plot(range(index_set_size), y1_avg[index_set], color=color_list[0], label=method_list[0], linewidth=1, marker=marker_list[0], alpha=0.5)
    ax.plot(range(index_set_size), y2_avg[index_set], color=color_list[1], label=method_list[1], linewidth=1, marker=marker_list[1], alpha=0.5)
    ax.plot(range(index_set_size), y3_avg[index_set], color=color_list[2], label=method_list[2], linewidth=1, marker=marker_list[2], alpha=0.5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    plt.legend(loc=lloc)  # loc=lloc bbox_to_anchor=lloc
    ax.set_xlim(x_min, x_max)
    x_ticks = [10 * i for i in range(10)]
    x_ticks.append(index_set_size - 1)
    ax.set_xticks(x_ticks)
    x_tick_labels = [(10 * i) * index_set_limit for i in range(10)]
    x_tick_labels.append(NUM_GAMES)
    ax.set_xticklabels(x_tick_labels)
    #ax.set_ylim(y_min, y_max)
    #ax.set_yticks([i*5 for i in range(8)])
    #ax.set_yticklabels([i*5*20 for i in range(8)])

    """
    axins = zoomed_inset_axes(ax, 200, loc='lower left', bbox_to_anchor=(0.45,0.1), bbox_transform=ax.transAxes, borderpad=3, axes_kwargs={"facecolor": "honeydew"})

    y_index = 0
    for nl in name_list:
        # ax.plot(x, Y[nl], color=color_list[y_index], label=method_list[y_index], linewidth=2)
        # ax.plot(x[avg_win:], y_avg[avg_win:], color=C[y_index], label=L[y_index])
        ax.plot(range(index_set_size), Y[nl][index_set], color=color_list[y_index], label=method_list[y_index], linewidth=1, marker=marker_list[y_index], alpha=0.5)
        y_index += 1

    x1, x2, y1, y2 = 98, 99, 85, 100
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    #axins.set_xticks([4996, 4998, 5000])
    #axins.set_xticklabels([4996, 4998, 5000])
    #axins.set_yticks([1060, 1062, 1064, 1066, 1068])

    pp, p1, p2 = mark_inset(ax, axins, loc1=1, loc2=3)
    # pp.set_fill(True)
    pp.set_facecolor("lightgray")
    pp.set_edgecolor("k")
    """

    plt.grid(alpha=0.3)
    # plt.show()
    plt.savefig(filename)  # format='eps'
def generate_rwds_plot_for_different_eps_decs():
    dir = "results/" + mg_vnf_plc_obj.FILE_NAME + "/"
    color_list = ["b", "g", "r", "darkviolet", "goldenrod"]
    # color_list = ["C1", "C2", "C3", "C4", "C5"]
    method_list = ["5e-4", "5e-6", "5e-8"]
    marker_list = ["o", "v", "P", "*", "x"]
    type = "float"
    avg_win = 1000
    lloc = (1, 0.5)  # 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    x_label = "Game Number"
    y_label = "Game Reward"
    IsYScaleLog = False
    filename = dir + "f_" + mg_vnf_plc_obj.FILE_NAME + "_rwds_" + str(avg_win) + '.png'
    figsize = (6, 4)
    index_set_size = 100
    index_set_limit = int(NUM_GAMES / index_set_size)
    x_min = 0
    x_max = index_set_size - 1
    y_min = -1
    y_max = 37
    sqr_root = 1

    x = range(NUM_GAMES)
    y1 = read_list_from_file(dir + "1/", "d_" + mg_vnf_plc_obj.FILE_NAME + "_ml_rwds" + ".txt", type, round_num=2)
    y2 = read_list_from_file(dir + "2/", "d_" + mg_vnf_plc_obj.FILE_NAME + "_ml_rwds" + ".txt", type, round_num=2)
    y3 = read_list_from_file(dir + "3/", "d_" + mg_vnf_plc_obj.FILE_NAME + "_ml_rwds" + ".txt", type, round_num=2)

    y1_avg = np.empty(len(y1))
    y2_avg = np.empty(len(y2))
    y3_avg = np.empty(len(y3))
    for i in range(len(y1)):
        tmp = np.mean(y1[max(0, i - avg_win):(i + 1)])
        y1_avg[i] = np.exp(np.log(tmp) / sqr_root) if tmp > 0 else 0
        tmp = np.mean(y2[max(0, i - avg_win):(i + 1)])
        y2_avg[i] = np.exp(np.log(tmp) / sqr_root) if tmp > 0 else 0
        tmp = np.mean(y3[max(0, i - avg_win):(i + 1)])
        y3_avg[i] = np.exp(np.log(tmp) / sqr_root) if tmp > 0 else 0

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)  # whole path
    index_set = [i * index_set_limit for i in range(index_set_size)]

    if IsYScaleLog:
        ax.set_yscale('log')

    ax.plot(range(index_set_size), y1_avg[index_set], color=color_list[0], label=method_list[0], linewidth=1, marker=marker_list[0], alpha=0.5)
    ax.plot(range(index_set_size), y2_avg[index_set], color=color_list[1], label=method_list[1], linewidth=1, marker=marker_list[1], alpha=0.5)
    ax.plot(range(index_set_size), y3_avg[index_set], color=color_list[2], label=method_list[2], linewidth=1, marker=marker_list[2], alpha=0.5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    plt.legend(bbox_to_anchor=lloc)  # loc=lloc bbox_to_anchor=lloc
    ax.set_xlim(x_min, x_max)
    x_ticks = [10 * i for i in range(10)]
    x_ticks.append(index_set_size - 1)
    ax.set_xticks(x_ticks)
    x_tick_labels = [(10 * i) * index_set_limit for i in range(10)]
    x_tick_labels.append(NUM_GAMES)
    ax.set_xticklabels(x_tick_labels)
    #ax.set_ylim(y_min, y_max)
    #ax.set_yticks([i*5 for i in range(8)])
    #ax.set_yticklabels([i*5*20 for i in range(8)])

    """
    axins = zoomed_inset_axes(ax, 200, loc='lower left', bbox_to_anchor=(0.45,0.1), bbox_transform=ax.transAxes, borderpad=3, axes_kwargs={"facecolor": "honeydew"})

    y_index = 0
    for nl in name_list:
        # ax.plot(x, Y[nl], color=color_list[y_index], label=method_list[y_index], linewidth=2)
        # ax.plot(x[avg_win:], y_avg[avg_win:], color=C[y_index], label=L[y_index])
        ax.plot(range(index_set_size), Y[nl][index_set], color=color_list[y_index], label=method_list[y_index], linewidth=1, marker=marker_list[y_index], alpha=0.5)
        y_index += 1

    x1, x2, y1, y2 = 98, 99, 85, 100
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    #axins.set_xticks([4996, 4998, 5000])
    #axins.set_xticklabels([4996, 4998, 5000])
    #axins.set_yticks([1060, 1062, 1064, 1066, 1068])

    pp, p1, p2 = mark_inset(ax, axins, loc1=1, loc2=3)
    # pp.set_fill(True)
    pp.set_facecolor("lightgray")
    pp.set_edgecolor("k")
    """

    plt.grid(alpha=0.3)
    # plt.show()
    plt.savefig(filename)  # format='eps'
def generate_costs_plot_for_different_eps_mins():
    dir = "results/" + mg_vnf_plc_obj.FILE_NAME + "/"
    color_list = ["b", "r", "g", "darkviolet", "goldenrod"]
    # color_list = ["C1", "C2", "C3", "C4", "C5"]
    method_list = ["0.1", "0.05", "0"]
    marker_list = ["o", "v", "P", "*", "x"]
    type = "float"
    avg_win = 1000
    lloc = 'lower left'  # (1, 0.6) 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    x_label = "Game Number"
    y_label = "Cost/Request"
    IsYScaleLog = False
    filename = dir + "f_" + mg_vnf_plc_obj.FILE_NAME + "_costs_" + str(avg_win) + '.png'
    figsize = (6, 4)
    index_set_size = 100
    index_set_limit = int(NUM_GAMES / index_set_size)
    x_min = 0
    x_max = index_set_size - 1
    y_min = -1
    y_max = 37
    sqr_root = 3

    x = range(NUM_GAMES)
    y1 = read_list_from_file(dir + "4/", "d_" + mg_vnf_plc_obj.FILE_NAME + "_ml_avg_ofs" + ".txt", type, round_num=2)
    y2 = read_list_from_file(dir + "5/", "d_" + mg_vnf_plc_obj.FILE_NAME + "_ml_avg_ofs" + ".txt", type, round_num=2)
    y3 = read_list_from_file(dir + "2/", "d_" + mg_vnf_plc_obj.FILE_NAME + "_ml_avg_ofs" + ".txt", type, round_num=2)

    y1_avg = np.empty(len(y1))
    y2_avg = np.empty(len(y2))
    y3_avg = np.empty(len(y3))
    for i in range(len(y1)):
        tmp = np.mean(y1[max(0, i - avg_win):(i + 1)])
        y1_avg[i] = np.exp(np.log(tmp) / sqr_root) if tmp > 0 else 0
        tmp = np.mean(y2[max(0, i - avg_win):(i + 1)])
        y2_avg[i] = np.exp(np.log(tmp) / sqr_root) if tmp > 0 else 0
        tmp = np.mean(y3[max(0, i - avg_win):(i + 1)])
        y3_avg[i] = np.exp(np.log(tmp) / sqr_root) if tmp > 0 else 0

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)  # whole path
    index_set = [i * index_set_limit for i in range(index_set_size)]

    if IsYScaleLog:
        ax.set_yscale('log')

    ax.plot(range(index_set_size), y1_avg[index_set], color=color_list[0], label=method_list[0], linewidth=1, marker=marker_list[0], alpha=0.5)
    ax.plot(range(index_set_size), y2_avg[index_set], color=color_list[1], label=method_list[1], linewidth=1, marker=marker_list[1], alpha=0.5)
    ax.plot(range(index_set_size), y3_avg[index_set], color=color_list[2], label=method_list[2], linewidth=1, marker=marker_list[2], alpha=0.5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    plt.legend(loc=lloc)  # loc=lloc bbox_to_anchor=lloc
    ax.set_xlim(x_min, x_max)
    x_ticks = [10 * i for i in range(10)]
    x_ticks.append(index_set_size - 1)
    ax.set_xticks(x_ticks)
    x_tick_labels = [(10 * i) * index_set_limit for i in range(10)]
    x_tick_labels.append(NUM_GAMES)
    ax.set_xticklabels(x_tick_labels)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks([i*5 for i in range(8)])
    ax.set_yticklabels([i*5*20 for i in range(8)])

    """
    axins = zoomed_inset_axes(ax, 200, loc='lower left', bbox_to_anchor=(0.45,0.1), bbox_transform=ax.transAxes, borderpad=3, axes_kwargs={"facecolor": "honeydew"})

    y_index = 0
    for nl in name_list:
        # ax.plot(x, Y[nl], color=color_list[y_index], label=method_list[y_index], linewidth=2)
        # ax.plot(x[avg_win:], y_avg[avg_win:], color=C[y_index], label=L[y_index])
        ax.plot(range(index_set_size), Y[nl][index_set], color=color_list[y_index], label=method_list[y_index], linewidth=1, marker=marker_list[y_index], alpha=0.5)
        y_index += 1

    x1, x2, y1, y2 = 98, 99, 85, 100
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    #axins.set_xticks([4996, 4998, 5000])
    #axins.set_xticklabels([4996, 4998, 5000])
    #axins.set_yticks([1060, 1062, 1064, 1066, 1068])

    pp, p1, p2 = mark_inset(ax, axins, loc1=1, loc2=3)
    # pp.set_fill(True)
    pp.set_facecolor("lightgray")
    pp.set_edgecolor("k")
    """

    plt.grid(alpha=0.3)
    # plt.show()
    plt.savefig(filename)  # format='eps'
def generate_reqs_plot_for_different_eps_mins():
    dir = "results/" + mg_vnf_plc_obj.FILE_NAME + "/"
    color_list = ["b", "r", "g", "darkviolet", "goldenrod"]
    # color_list = ["C1", "C2", "C3", "C4", "C5"]
    method_list = ["0.1", "0.05", "0"]
    marker_list = ["o", "v", "P", "*", "x"]
    type = "float"
    avg_win = 1000
    lloc = 'lower right'  # (1, 0.6) 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    x_label = "Game Number"
    y_label = "% Request"
    IsYScaleLog = False
    filename = dir + "f_" + mg_vnf_plc_obj.FILE_NAME + "_reqs_" + str(avg_win) + '.png'
    figsize = (6, 4)
    index_set_size = 100
    index_set_limit = int(NUM_GAMES / index_set_size)
    x_min = 0
    x_max = index_set_size - 1
    y_min = -1
    y_max = 37
    sqr_root = 1

    x = range(NUM_GAMES)
    y1 = read_list_from_file(dir + "4/", "d_" + mg_vnf_plc_obj.FILE_NAME + "_ml_reqs" + ".txt", type, round_num=2)
    y2 = read_list_from_file(dir + "5/", "d_" + mg_vnf_plc_obj.FILE_NAME + "_ml_reqs" + ".txt", type, round_num=2)
    y3 = read_list_from_file(dir + "2/", "d_" + mg_vnf_plc_obj.FILE_NAME + "_ml_reqs" + ".txt", type, round_num=2)

    y1_avg = np.empty(len(y1))
    y2_avg = np.empty(len(y2))
    y3_avg = np.empty(len(y3))
    for i in range(len(y1)):
        tmp = np.mean(y1[max(0, i - avg_win):(i + 1)])
        y1_avg[i] = np.exp(np.log(tmp) / sqr_root) if tmp > 0 else 0
        tmp = np.mean(y2[max(0, i - avg_win):(i + 1)])
        y2_avg[i] = np.exp(np.log(tmp) / sqr_root) if tmp > 0 else 0
        tmp = np.mean(y3[max(0, i - avg_win):(i + 1)])
        y3_avg[i] = np.exp(np.log(tmp) / sqr_root) if tmp > 0 else 0

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)  # whole path
    index_set = [i * index_set_limit for i in range(index_set_size)]

    if IsYScaleLog:
        ax.set_yscale('log')

    ax.plot(range(index_set_size), y1_avg[index_set], color=color_list[0], label=method_list[0], linewidth=1, marker=marker_list[0], alpha=0.5)
    ax.plot(range(index_set_size), y2_avg[index_set], color=color_list[1], label=method_list[1], linewidth=1, marker=marker_list[1], alpha=0.5)
    ax.plot(range(index_set_size), y3_avg[index_set], color=color_list[2], label=method_list[2], linewidth=1, marker=marker_list[2], alpha=0.5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    plt.legend(loc=lloc)  # loc=lloc bbox_to_anchor=lloc
    ax.set_xlim(x_min, x_max)
    x_ticks = [10 * i for i in range(10)]
    x_ticks.append(index_set_size - 1)
    ax.set_xticks(x_ticks)
    x_tick_labels = [(10 * i) * index_set_limit for i in range(10)]
    x_tick_labels.append(NUM_GAMES)
    ax.set_xticklabels(x_tick_labels)
    #ax.set_ylim(y_min, y_max)
    #ax.set_yticks([i*5 for i in range(8)])
    #ax.set_yticklabels([i*5*20 for i in range(8)])

    """
    axins = zoomed_inset_axes(ax, 200, loc='lower left', bbox_to_anchor=(0.45,0.1), bbox_transform=ax.transAxes, borderpad=3, axes_kwargs={"facecolor": "honeydew"})

    y_index = 0
    for nl in name_list:
        # ax.plot(x, Y[nl], color=color_list[y_index], label=method_list[y_index], linewidth=2)
        # ax.plot(x[avg_win:], y_avg[avg_win:], color=C[y_index], label=L[y_index])
        ax.plot(range(index_set_size), Y[nl][index_set], color=color_list[y_index], label=method_list[y_index], linewidth=1, marker=marker_list[y_index], alpha=0.5)
        y_index += 1

    x1, x2, y1, y2 = 98, 99, 85, 100
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    #axins.set_xticks([4996, 4998, 5000])
    #axins.set_xticklabels([4996, 4998, 5000])
    #axins.set_yticks([1060, 1062, 1064, 1066, 1068])

    pp, p1, p2 = mark_inset(ax, axins, loc1=1, loc2=3)
    # pp.set_fill(True)
    pp.set_facecolor("lightgray")
    pp.set_edgecolor("k")
    """

    plt.grid(alpha=0.3)
    # plt.show()
    plt.savefig(filename)  # format='eps'
def generate_rwds_plot_for_different_eps_mins():
    dir = "results/" + mg_vnf_plc_obj.FILE_NAME + "/"
    color_list = ["b", "r", "g", "darkviolet", "goldenrod"]
    # color_list = ["C1", "C2", "C3", "C4", "C5"]
    method_list = ["0.1", "0.05", "0"]
    marker_list = ["o", "v", "P", "*", "x"]
    type = "float"
    avg_win = 1000
    lloc = 'lower right'  # (1, 0.5) 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    x_label = "Game Number"
    y_label = "Game Reward"
    IsYScaleLog = False
    filename = dir + "f_" + mg_vnf_plc_obj.FILE_NAME + "_rwds_" + str(avg_win) + '.png'
    figsize = (6, 4)
    index_set_size = 100
    index_set_limit = int(NUM_GAMES / index_set_size)
    x_min = 0
    x_max = index_set_size - 1
    y_min = -1
    y_max = 37
    sqr_root = 1

    x = range(NUM_GAMES)
    y1 = read_list_from_file(dir + "4/", "d_" + mg_vnf_plc_obj.FILE_NAME + "_ml_rwds" + ".txt", type, round_num=2)
    y2 = read_list_from_file(dir + "5/", "d_" + mg_vnf_plc_obj.FILE_NAME + "_ml_rwds" + ".txt", type, round_num=2)
    y3 = read_list_from_file(dir + "2/", "d_" + mg_vnf_plc_obj.FILE_NAME + "_ml_rwds" + ".txt", type, round_num=2)

    y1_avg = np.empty(len(y1))
    y2_avg = np.empty(len(y2))
    y3_avg = np.empty(len(y3))
    for i in range(len(y1)):
        tmp = np.mean(y1[max(0, i - avg_win):(i + 1)])
        y1_avg[i] = np.exp(np.log(tmp) / sqr_root) if tmp > 0 else 0
        tmp = np.mean(y2[max(0, i - avg_win):(i + 1)])
        y2_avg[i] = np.exp(np.log(tmp) / sqr_root) if tmp > 0 else 0
        tmp = np.mean(y3[max(0, i - avg_win):(i + 1)])
        y3_avg[i] = np.exp(np.log(tmp) / sqr_root) if tmp > 0 else 0

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)  # whole path
    index_set = [i * index_set_limit for i in range(index_set_size)]

    if IsYScaleLog:
        ax.set_yscale('log')

    ax.plot(range(index_set_size), y1_avg[index_set], color=color_list[0], label=method_list[0], linewidth=1, marker=marker_list[0], alpha=0.5)
    ax.plot(range(index_set_size), y2_avg[index_set], color=color_list[1], label=method_list[1], linewidth=1, marker=marker_list[1], alpha=0.5)
    ax.plot(range(index_set_size), y3_avg[index_set], color=color_list[2], label=method_list[2], linewidth=1, marker=marker_list[2], alpha=0.5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    plt.legend(loc=lloc)  # bbox_to_anchor=lloc
    ax.set_xlim(x_min, x_max)
    x_ticks = [10 * i for i in range(10)]
    x_ticks.append(index_set_size - 1)
    ax.set_xticks(x_ticks)
    x_tick_labels = [(10 * i) * index_set_limit for i in range(10)]
    x_tick_labels.append(NUM_GAMES)
    ax.set_xticklabels(x_tick_labels)
    #ax.set_ylim(y_min, y_max)
    #ax.set_yticks([i*5 for i in range(8)])
    #ax.set_yticklabels([i*5*20 for i in range(8)])

    """
    axins = zoomed_inset_axes(ax, 200, loc='lower left', bbox_to_anchor=(0.45,0.1), bbox_transform=ax.transAxes, borderpad=3, axes_kwargs={"facecolor": "honeydew"})

    y_index = 0
    for nl in name_list:
        # ax.plot(x, Y[nl], color=color_list[y_index], label=method_list[y_index], linewidth=2)
        # ax.plot(x[avg_win:], y_avg[avg_win:], color=C[y_index], label=L[y_index])
        ax.plot(range(index_set_size), Y[nl][index_set], color=color_list[y_index], label=method_list[y_index], linewidth=1, marker=marker_list[y_index], alpha=0.5)
        y_index += 1

    x1, x2, y1, y2 = 98, 99, 85, 100
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    #axins.set_xticks([4996, 4998, 5000])
    #axins.set_xticklabels([4996, 4998, 5000])
    #axins.set_yticks([1060, 1062, 1064, 1066, 1068])

    pp, p1, p2 = mark_inset(ax, axins, loc1=1, loc2=3)
    # pp.set_fill(True)
    pp.set_facecolor("lightgray")
    pp.set_edgecolor("k")
    """

    plt.grid(alpha=0.3)
    # plt.show()
    plt.savefig(filename)  # format='eps'

generate_costs_plot_for_different_methods_and_changing_requirements()
# generate_reqs_plot_for_different_methods_and_changing_requirements()
# generate_dlys_plot_for_different_methods_and_changing_requirements()
# generate_costs_plot_for_different_eps_decs()
# generate_reqs_plot_for_different_eps_decs()
# generate_rwds_plot_for_different_eps_decs()
# generate_costs_plot_for_different_eps_mins()
# generate_reqs_plot_for_different_eps_mins()
# generate_rwds_plot_for_different_eps_mins()