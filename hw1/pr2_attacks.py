# Homework 1 problem 2 - Running counter
# Sarah Scheffler
# Privacy in ML
# 10/10/18

import csv
from itertools import takewhile
from math import ceil
import numpy as np

REPETITIONS = int(20)
N_LIST = [100, 5000, 1000, 5000]

########## Helper functions ########## 

def rand_nbits(n: int):
    """Generate an n-element column vector of random integers in {0,1}"""
    n = int(n)
    return np.random.randint(2, size=n)

def experiment(n, attacker_guess, with_aux_info=False):
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

def no_aux_info_experiment(n):
    def attacker_guess(a, aux_info=None):
        return a
    experiment(n, attacker_guess)

no_aux_info_experiment(8)



########## Hadamard attack implementation ########## 
#
#def hadamard_experiment(n, sigma, m=None):
    #"""Returns (mean, std) of fraction of correct results for Hadamard attack with this n and sigma"""
    #H = hadamard(n)
    #def attacker_guess(H, a):
        #"""attacker guess = H*a rounded to {0,1}"""
        #return np.fromiter((z > 0.5 for z in H.dot(a)), int)
    ##return experiment(n, sigma, n, H, attacker_guess)
#
#
#def random_query_experiment(n, sigma, m):
    #"""Returns (mean, std) of fraction of correct results for random query attack with this n, sigma, and m"""
#
    #B = np.random.randint(2, size=(m,n)) # random (m x n) matrix with elems in {0,1}
#
    #def attacker_guess(B, a):
        #"""Compute argmin_y of ||a-(1/b)By||_2, then return the rounded result"""
        #return np.fromiter((z > 0.5 for z in np.linalg.lstsq((1/n)*B, a)[0]), int)
    #return experiment(n, sigma, m, B, attacker_guess)
#
#
#def read_experiment_csv_dict(filename):
    #"""Returns a list of data points read from a csv into a dict (n,sigma,m) -> (mean,stdev,reps)"""
    #to_ret = {}
    #with open(filename, 'r') as csvfile:
        #reader = csv.DictReader(csvfile)
        #for row in reader:
            #to_ret[(int(row["n"]), float(row["sigma"]), int(row["m"]))] = (float(row["mean"]),
                #float(row["stdev"]), int(row["repetitions"]))
    #return to_ret
#
#def generate_experiment_csv(filename, n_list, sigma_list_gen, m_list_gen, experiment):
    #already_done = read_experiment_csv_dict(filename)
    #with open(filename, 'a') as csvfile:
        #fw = csv.writer(csvfile, delimiter=',')
        #if len(already_done) == 0:
            #fw.writerow(["n","sigma","m","mean","stdev","repetitions"])
        #for n in n_list:
            #for sigma in sigma_list_gen(n):
                #for m in m_list_gen(n):
                    #if (n,sigma,m) not in already_done:
                        #mean, std = experiment(n=n, sigma=sigma, m=m)
                        #fw.writerow([n, sigma, m, mean, std, REPETITIONS])
#
#
#
#
##test_hadamard()
##test_rand_nbits()
##test_evaluate_hadamard()
##generate_experiment_csv("hadamard_results.csv", N_LIST, SIGMA_LIST_GEN, lambda _: [None], hadamard_experiment)
#generate_experiment_csv("random_query_results.csv", N_LIST, SIGMA_LIST_GEN, M_LIST_GEN, random_query_experiment)





