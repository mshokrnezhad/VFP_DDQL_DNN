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
