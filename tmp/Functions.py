# from docplex.mp.model import Model
import numpy as np
import matplotlib.pyplot as plt
import os.path
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

rnd = np.random


def specify_requests_entry_nodes(FIRST_TIER_NODES, REQUESTS, seed=0):
    rnd.seed(seed)
    return np.array([rnd.choice(FIRST_TIER_NODES) for i in REQUESTS])


def assign_requests_to_services(SERVICES, REQUESTS, seed=1):
    rnd.seed(seed)
    return np.array([rnd.choice(SERVICES) for i in REQUESTS])


def calculate_input_shape(NUM_NODES, NUM_REQUESTS, NUM_PRIORITY_LEVELS, switch):
    counter = 0

    if switch == "srv_plc":
        counter = (2 * NUM_REQUESTS) + (5 * NUM_NODES) + ((2 + NUM_PRIORITY_LEVELS) * (NUM_NODES ** 2))
    """
    if switch == "srv_plc":
        counter = (2 * NUM_REQUESTS) + (5 * NUM_NODES) + ((2) * (NUM_NODES ** 2))
    """
    if switch == "pri_asg":
        counter = (2 * NUM_REQUESTS) + (6 * NUM_NODES) + ((2 + NUM_PRIORITY_LEVELS) * (NUM_NODES ** 2))

    return counter


def parse_state(state, NUM_NODES, NUM_REQUESTS, env_obj, switch="none"):
    np.set_printoptions(suppress=True, linewidth=100)
    counter = 0

    print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    """
    print("ACTIVE REQUESTS:")
    print(state[counter:NUM_REQUESTS].astype(int))
    counter += NUM_REQUESTS

    print("\nPER NODE REQUEST CAPACITY REQUIREMENTS:")
    print(state[counter:counter + NUM_NODES].astype(int))
    counter += NUM_NODES

    print("\nPER NODE REQUEST BW REQUIREMENTS:")
    print(state[counter:counter + NUM_NODES].astype(int))
    counter += NUM_NODES

    print("\nREQUEST DELAY REQUIREMENTS:")
    print(state[counter:counter + NUM_REQUESTS].astype(int))
    counter += NUM_REQUESTS

    print("\nPER NODE REQUEST BURST SIZES:")
    print(state[counter:counter + NUM_NODES].astype(int))
    counter += NUM_NODES

    if switch == "pri_asg":
        print("\nPER ASSIGNED NODE REQUEST BW REQUIREMENTS:")
        print(state[counter:counter + NUM_NODES].astype(int))
        counter += NUM_NODES

    print("\nDC CAPACITIES:")
    print(state[counter:counter + NUM_NODES].astype(int))
    counter += NUM_NODES

    print("\nDC COSTS:")
    print(state[counter:counter + NUM_NODES].astype(int))
    counter += NUM_NODES

    print("\nLINK BWS MATRIX:")
    print(state[counter:counter + NUM_NODES ** 2].astype(int).reshape(NUM_NODES, NUM_NODES))
    counter += NUM_NODES ** 2

    print("\nLINK COSTS MATRIX:")
    print(state[counter:counter + NUM_NODES ** 2].astype(int).reshape(NUM_NODES, NUM_NODES))
    counter += NUM_NODES ** 2

    print("\nLINK DELAYS MATRIX:")
    link_delays_matrix = state[counter:counter + env_obj.net_obj.NUM_PRIORITY_LEVELS * (NUM_NODES ** 2)]. \
        reshape(env_obj.net_obj.NUM_PRIORITY_LEVELS, NUM_NODES, NUM_NODES)
    # since we removed null index 0, index 0 of link_delays_matrix is for priority 1 and so on.
    for n in range(0, env_obj.net_obj.NUM_PRIORITY_LEVELS):
        print(f"Priority: {n + 1}")
        print(link_delays_matrix[n])
    counter += env_obj.net_obj.NUM_PRIORITY_LEVELS * (NUM_NODES ** 2)
    """

    print("ACTIVE REQUESTS:")
    print(state[counter:NUM_REQUESTS])
    counter += NUM_REQUESTS

    print("\nPER NODE REQUEST CAPACITY REQUIREMENTS:")
    print(state[counter:counter + NUM_NODES])
    counter += NUM_NODES

    print("\nPER NODE REQUEST BW REQUIREMENTS:")
    print(state[counter:counter + NUM_NODES])
    counter += NUM_NODES

    print("\nREQUEST DELAY REQUIREMENTS:")
    print(state[counter:counter + NUM_REQUESTS])
    counter += NUM_REQUESTS

    print("\nPER NODE REQUEST BURST SIZES:")
    print(state[counter:counter + NUM_NODES])
    counter += NUM_NODES

    if switch == "pri_asg":
        print("\nPER ASSIGNED NODE REQUEST BW REQUIREMENTS:")
        print(state[counter:counter + NUM_NODES])
        counter += NUM_NODES

    print("\nDC CAPACITIES:")
    print(state[counter:counter + NUM_NODES])
    counter += NUM_NODES

    print("\nDC COSTS:")
    print(state[counter:counter + NUM_NODES])
    counter += NUM_NODES

    print("\nLINK BWS MATRIX:")
    print(state[counter:counter + NUM_NODES ** 2].reshape(NUM_NODES, NUM_NODES))
    counter += NUM_NODES ** 2

    print("\nLINK COSTS MATRIX:")
    print(state[counter:counter + NUM_NODES ** 2].reshape(NUM_NODES, NUM_NODES))
    counter += NUM_NODES ** 2

    print("\nLINK DELAYS MATRIX:")
    link_delays_matrix = state[counter:counter + env_obj.net_obj.NUM_PRIORITY_LEVELS * (NUM_NODES ** 2)]. \
        reshape(env_obj.net_obj.NUM_PRIORITY_LEVELS, NUM_NODES, NUM_NODES)
    # since we removed null index 0, index 0 of link_delays_matrix is for priority 1 and so on.
    for n in range(0, env_obj.net_obj.NUM_PRIORITY_LEVELS):
        print(f"Priority: {n + 1}")
        print(link_delays_matrix[n])
    counter += env_obj.net_obj.NUM_PRIORITY_LEVELS * (NUM_NODES ** 2)

    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")


def plot_learning_curve(x, y, epsilons, filename=""):
    fig = plt.figure()
    s_plt1 = fig.add_subplot(111, label="1")  # "234" means "2x3 grid, 4th subplot".
    s_plt2 = fig.add_subplot(111, label="2", frame_on=False)

    s_plt1.plot(x, epsilons, color="C0")
    s_plt1.set_xlabel("Game Number", color="C0")
    s_plt1.set_ylabel("Epsilon", color="C0")
    s_plt1.tick_params(axis="x", color="C0")
    s_plt1.tick_params(axis="y", color="C0")

    n = len(y)
    y_avg = np.empty(n)
    for i in range(n):
        y_avg[i] = np.mean(y[max(0, i - 100):(i + 1)])

    s_plt2.plot(x, y_avg, color="C1")
    s_plt2.axes.get_xaxis().set_visible(False)
    s_plt2.yaxis.tick_right()
    s_plt2.set_ylabel('Cost', color="C1")
    s_plt2.yaxis.set_label_position('right')
    s_plt2.tick_params(axis='y', colors="C1")

    plt.show()
    plt.savefig(filename)


def simple_plot(x, y, filename="", avg_win=100):
    fig = plt.figure()
    plt1 = fig.add_subplot(111, label="2")

    y_avg = np.empty(len(y))
    for i in range(len(y)):
        y_avg[i] = np.mean(y[max(0, i - avg_win):(i + 1)])

    plt1.plot(x[avg_win:], y_avg[avg_win:], color="C1")
    plt1.set_xlabel("Game Number", color="C1")
    plt1.set_ylabel("???", color="C1")
    plt1.tick_params(axis="x", color="C1")
    plt1.tick_params(axis="y", color="C1")

    # plt.show()
    plt.savefig(filename)


""" 
def simple_plot(x, y, filename="", avg_win=100):
    fig = plt.figure()
    plt1 = fig.add_subplot(111, label="2")

    y_avg = np.empty(len(y))
    for i in range(len(y)):
        y_avg[i] = np.mean(y[max(0, i - avg_win):(i + 1)])

    plt1.plot(x[avg_win:], y_avg[avg_win:], color="C1")
    plt1.set_xlabel("Game Number", color="C1")
    plt1.set_ylabel("???", color="C1")
    plt1.tick_params(axis="x", color="C1")
    plt1.tick_params(axis="y", color="C1")

    # plt.show()
    plt.savefig(filename)
"""


def multi_plot(x, Y, filename="", avg_win=100, axis_label="", C=[], L=[], lloc="", IsYScaleLog=False):
    fig = plt.figure(figsize=(8, 5))
    ax = plt.subplot(111)  # whole path
    index_set = [i*10 for i in range(1000)]

    for y_index in range(len(Y)):
        y_avg = np.empty(len(Y[y_index]))
        for i in range(len(Y[y_index])):
            y_avg[i] = np.mean(Y[y_index][max(0, i - avg_win):(i + 1)])

        ax.plot(x, y_avg, color=C[y_index], label=L[y_index], linewidth=2)
        # ax.plot(x[avg_win:], y_avg[avg_win:], color=C[y_index], label=L[y_index])
        # ax.plot(range(1000), y_avg[index_set], color=C[y_index], label=L[y_index], linewidth=2)

    ax.set_xlabel("Game Number")
    ax.set_ylabel(axis_label)
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    plt.legend(bbox_to_anchor=lloc)  # loc="best"

    """
    axins = zoomed_inset_axes(ax, 200, loc='lower left', bbox_to_anchor=(0.45,0.1), bbox_transform=ax.transAxes, borderpad=3, axes_kwargs={"facecolor": "honeydew"})

    for y_index in range(len(Y)):
        y_avg = np.empty(len(Y[y_index]))
        for i in range(len(Y[y_index])):
            y_avg[i] = np.mean(Y[y_index][max(0, i - avg_win):(i + 1)])
        axins.plot(x[avg_win:], y_avg[avg_win:], color=C[y_index], label=L[y_index])

    x1, x2, y1, y2 = 4996, 5000, 1060, 1068
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticks([4996, 4998, 5000])
    axins.set_xticklabels([4996, 4998, 5000])
    axins.set_yticks([1060, 1062, 1064, 1066, 1068])

    pp, p1, p2 = mark_inset(ax, axins, loc1=1, loc2=3)
    # pp.set_fill(True)
    pp.set_facecolor("lightgray")
    pp.set_edgecolor("k")
    """

    if IsYScaleLog:
        ax.set_yscale('log')

    plt.grid()
    # plt.show()
    plt.savefig(filename)


def save_list_to_file(list, dir, file_name):
    full_name = dir + file_name + ".txt"
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    f = open(full_name, "w")
    for i in range(len(list)):
        if i < len(list) - 1:
            f.write(str(list[i]) + "\n")
        else:
            f.write(str(list[i]))
    f.close()


def append_list_to_file(list, dir, file_name):
    full_name = dir + file_name + ".txt"
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    f = open(full_name, "a")
    for i in range(len(list)):
        if i < len(list) - 1:
            f.write(str(list[i]) + "\n")
        else:
            f.write(str(list[i]))
    f.close()


def read_list_from_file(dir, file_name, type, round_num=2):
    addr = dir + file_name
    f = open(addr, "r")  # opens the file in read mode
    list = f.read().splitlines()  # puts the file into an array
    f.close()
    if type == "int":
        return [int(element) for element in list]
    if type == "float":
        return [round(float(element), round_num) for element in list]


def generate_seeds(num_seeds):
    seeds = []
    for i in range(num_seeds):
        seeds.append(np.random.randint(1, 100))
    save_list_to_file(seeds, "inputs/", "SEEDS_100")
