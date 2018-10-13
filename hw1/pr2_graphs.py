# Homework 1 problem 2 - Graphing
# Sarah Scheffler
# Privacy in ML
# 10/13/18

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

MARKERS = ['o','s','v','P','x','*','h','+',]
N_LIST = [100, 500, 1000, 5000]

def read_experiment_csv(filename):
    """Returns a list of data points read from a csv"""
    to_ret = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            to_ret.append([int(row["n"]), float(row["mean"]), float(row["stdev"]), int(row["repetitions"])])
    return to_ret

def graph(data_list, graph_filename):
    """Takes in a list of the two pr2 data lists and returns a graph of the results"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("$n$")
    ax.set_ylabel("fraction of rows reconstructed")
    ax.set_xticks(N_LIST)
    ax.get_xaxis().set_major_formatter(tick.ScalarFormatter())
    for t in ax.get_xticklabels():
        t.set_rotation(45)

    counter = 0
    for data in data_list:
        data_xs = list(map(lambda d: d[0], data))
        data_ys = list(map(lambda d: d[1], data))
        data_sd = list(map(lambda d: d[2], data))

        plt.errorbar(data_xs, data_ys, yerr=data_sd, marker=MARKERS[counter], label="no aux info" if counter==0 else
                "with aux info")
        counter += 1

    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_filename)

graph([read_experiment_csv("no_aux_info.csv"),
    read_experiment_csv("with_aux_info.csv")], "running_counter.png")

