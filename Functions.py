# from docplex.mp.model import Model
import numpy as np
import matplotlib.pyplot as plt
import os.path
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
rnd = np.random


def read_list_from_file(dir, file_name, type, round_num=2):
    addr = dir + file_name
    f = open(addr, "r")  # opens the file in read mode
    list = f.read().splitlines()  # puts the file into an array
    f.close()
    if type == "int":
        return [int(element) for element in list]
    if type == "float":
        return [round(float(element), round_num) for element in list]

def parse_state(N, state, net_obj):
    norm = 1
    entry_node = 0
    
    _dc_capacities = np.array([state["NODE_FEATURES"][i][0] for i in range(N)])
    dc_capacities = np.round((_dc_capacities / norm), 3)
    _dc_costs = np.array([state["NODE_FEATURES"][i][1] for i in range(N)])
    dc_costs = np.round((_dc_costs / norm), 3)
    
    paths = [[] for v in range(N)]
    for v in range(N):
        _paths = []
        for path in net_obj.PATHS_LIST:
            if path[0] == entry_node and path[-1] == v:
                _paths.append(path)
        if len(_paths) > 0:
            for p in range(net_obj.NUM_PATHS_UB):
                paths[v].append(_paths[p])
        else:
            for p in range(net_obj.NUM_PATHS_UB):
                paths[v].append(-1)
    
    _path_bws = np.zeros((N, net_obj.NUM_PATHS_UB))
    for v in range(N):
        for p in range(net_obj.NUM_PATHS_UB):
            path = paths[v][p]
            path_index = net_obj.PATHS_LIST.index(path) if path in net_obj.PATHS_LIST else -1
            if path != -1:
                _path_bws[v][p] = net_obj.LINK_BWS[np.where(net_obj.LINKS_PATHS_MATRIX[:, path_index] == 1)[0]].min()
            else:
                _path_bws[v][p] = -1  
    path_bws = np.round((_path_bws.reshape(1, -1)[0] / norm), 3)

                
    _path_costs = np.zeros((N, net_obj.NUM_PATHS_UB))
    for v in range(N):
        for p in range(net_obj.NUM_PATHS_UB):
            path = paths[v][p]
            path_index = net_obj.PATHS_LIST.index(path) if path in net_obj.PATHS_LIST else -1
            if path != -1:
                _path_costs[v][p] = net_obj.LINK_COSTS[np.where(net_obj.LINKS_PATHS_MATRIX[:, path_index] == 1)[0]].sum()
            else:
                # path_costs[v][p] = net_obj.LINK_COST_MU + net_obj.LINK_COST_SIGMA + 1
                _path_costs[v][p] = -1
    path_costs = np.round((_path_costs.reshape(1, -1)[0] / norm), 3)
    
    
    _path_delays = np.zeros((N, net_obj.NUM_PATHS_UB, net_obj.NUM_PRIORITY_LEVELS))
    for v in range(N):
        for p in range(net_obj.NUM_PATHS_UB):
            path = paths[v][p]
            path_index = net_obj.PATHS_LIST.index(path) if path in net_obj.PATHS_LIST else -1
            if path != -1:
                for k in range(net_obj.NUM_PRIORITY_LEVELS):
                    _path_delays[v][p][k] = net_obj.LINK_DELAYS[np.where(net_obj.LINKS_PATHS_MATRIX[:, path_index] == 1)[0]][:, k + 1].sum()
            else:
                _path_delays[v][p] = [-1 for k in range(net_obj.NUM_PRIORITY_LEVELS)]
    path_delays = np.round((_path_delays.reshape(1, -1)[0] / norm), 3)

    state = np.concatenate((dc_capacities, dc_costs, path_bws, path_costs, path_delays))
    # print(state)
    
    return state
