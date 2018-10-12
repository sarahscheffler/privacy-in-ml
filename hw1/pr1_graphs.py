# Homework 1 problem 1 - Graphing
# Sarah Scheffler
# Privacy in ML
# 10/9/18

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

MARKERS = ['o','s','v','P','x','*','h','+',]
SIGMAS = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125]

def read_experiment_csv(filename):
    """Returns a list of data points read from a csv"""
    to_ret = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            to_ret.append((int(row["n"]), float(row["sigma"]), int(row["m"]) if len(row["m"])>0 else None, float(row["mean"]),
                float(row["stdev"]), int(row["repetitions"])))
    return to_ret

def graph_hadamard(data, graph_filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("$\sigma$")
    ax.set_ylabel("fraction of rows reconstructed")
    ax.set_xscale("log")
    ax.set_xticks(SIGMAS)
    ax.get_xaxis().set_major_formatter(tick.ScalarFormatter())
    ax.set_xticklabels(["$2^{{-{0:d}}}$".format(x) for x in range(1, len(SIGMAS)+1)])
    for t in ax.get_xticklabels():
        t.set_rotation(45)

    counter = 0
    for n in set([d[0] for d in data]):
        sub_data = list(filter(lambda d: d[0] == n, data))
        sub_data_xs = list(map(lambda d: d[1], sub_data))
        sub_data_ys = list(map(lambda d: d[3], sub_data))
        sub_data_sd = list(map(lambda d: d[4], sub_data))

        plt.errorbar(sub_data_xs, sub_data_ys, yerr=sub_data_sd, marker=MARKERS[counter], label="n={0:d}".format(n))
        counter += 1

    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_filename)

graph_hadamard(read_experiment_csv("hadamard_results.csv"), "hadamard_graph.png")

