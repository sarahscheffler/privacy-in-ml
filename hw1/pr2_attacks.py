# Homework 1 problem 2 - Running counter
# Sarah Scheffler
# Privacy in ML
# 10/10/18

import csv
from itertools import takewhile
from math import ceil
import numpy as np

REPETITIONS = int(20)
N_LIST = [100, 500, 1000, 5000]

########## Helper functions ########## 

def rand_nbits(n: int):
    """Generate an n-element column vector of random integers in {0,1}"""
    n = int(n)
    return np.random.randint(2, size=n)


def experiment(n, with_aux_info=False):
    """Taking in a matrix of queries and a functor for attacker strategy, determine attacker success rate"""
    def answers(x):
        """Return noisy counter"""
        Z = rand_nbits(n) # roll all noise values in advance
        return np.cumsum(x) + Z

    def aux_info(x):
        """For each element of x, return x[i] with 2/3 probability and 1-x[i] otherwise"""
        cutoff = 2/3
        randomness = np.random.sample(n)
        return np.array([x[i] if randomness[i] < cutoff else int(1-x[i]) for i in range(len(x))])
    
    results = []
    reps = 0
    print("Running {0:d} iterations of experiment with n={1:d}...".format(REPETITIONS, n))
    for _ in range(REPETITIONS):
        try:
            x = rand_nbits(n)
            a = answers(x)
            xhat = attacker_guess(a, aux_info(x) if with_aux_info else None)
            result = (1/n) * np.count_nonzero(x == xhat) # fraction of correct guesses
            results.append(result)
            reps += 1
        except MemoryError as e:
            print(str(e))
    if len(results) > 0:
        mean = np.mean(results)
        std = np.std(results)
        print("    Mean fraction of correct results: {0:f}\n    Standard deviation of fraction of correct results: {1:f}".format(mean, std))
    else:
        mean = None
        std = None
        print("    Experiment failed due to memory error.")
    return (mean, std, reps)

def experiment_with_aux_info(n):
    return experiment(n, with_aux_info = True)

# this algorithm only relies on the answers a and the optional aux_input
def attacker_guess(a, aux_info=None):
    """Guess the values for x based on the noisy counter a and optional aux_info.
       Uses four principles:
           0. If we have aux_info, start with those initial guesses.  Else guess randomly.
           1. At each step, if a[i] = a[i-1]+2, then we know x[i] = 1
           2. At each step, if a[i] = a[i-1]-1, then we know x[i] = 0
           3. At each step, cumsum(x[i])+1 >= a[i] >= cumsum(x[i])
    """
    VERBOSE = False # To see the reasoning for modifying individual guesses, set this to True
    n = len(a)
    xhat = aux_info if aux_info is not None else rand_nbits(n) # start with either random guesses or aux info
    if VERBOSE:
        counter = 0
        print(counter, xhat, "start")
    # At each step, if a[i] = a[i-1]+2, then we know x[i] = 1
    # At each step, if a[i] = a[i-1]-1, then we know x[i] = 0
    if a[0] == 2 and xhat[0] != 1: # handle index 0 separately
        xhat[0] = 1
        if VERBOSE:
            counter += 1
            print(counter, xhat, "2=>1")
    elif a[0] == 0 and xhat[0] != 0:
        xhat[0] = 0
        if VERBOSE:
            counter += 1
            print(counter, xhat, "-1=>0")
    for i in range(1, n):
        if a[i] == 2 + a[i-1] and xhat[i] != 1:
            xhat[i] = 1
            if VERBOSE:
                counter += 1
                print(counter, xhat, "2=>1")
        elif a[i] == a[i-1] - 1 and xhat[i] != 0:
            xhat[i] = 0
            if VERBOSE:
                counter += 1
                print(counter, xhat, "-1=>0")

    # At each step, cumsum(x[i])+1 >= a[i] >= cumsum(x[i]) (by definition)
    complete = False
    while not complete:
        cumsum = np.cumsum(xhat)
        for i in range(n):
            if a[i] < cumsum[i]:
                for j in range(i, -1, -1):
                    if xhat[j] != 0:
                        xhat[j] = 0
                        break
                if VERBOSE:
                    counter += 1
                    print(counter, xhat, "ai<cs[{0:d}]".format(i))
                break
            if a[i] > cumsum[i] + 1:
                for j in range(i, -1, -1):
                    if xhat[j] != 1:
                        xhat[j] = 1
                        break
                if VERBOSE:
                    counter += 1
                    print(counter, xhat, "ai>cs[{0:d}]+1".format(i))
                break
            if i==n-1: # if we make it to the end of this loop without changing anything, we're done
                complete = True
    return xhat

def read_experiment_csv_dict(filename):
    """Returns a list of data points read from a csv into a dict (n,sigma,m) -> (mean,stdev,reps)"""
    to_ret = {}
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            to_ret[int(row["n"])] = (float(row["mean"]),
                float(row["stdev"]), int(row["repetitions"]))
    return to_ret

def generate_experiment_csv(filename, n_list, experiment):
    already_done = read_experiment_csv_dict(filename)
    with open(filename, 'a') as csvfile:
        fw = csv.writer(csvfile, delimiter=',')
        if len(already_done) == 0:
            fw.writerow(["n","mean","stdev","repetitions"])
        for n in n_list:
            if n not in already_done:
                mean, std, reps = experiment(n)
                fw.writerow([n, mean, std, reps])

generate_experiment_csv("no_aux_info.csv", N_LIST, experiment)
generate_experiment_csv("with_aux_info.csv", N_LIST, experiment_with_aux_info)





