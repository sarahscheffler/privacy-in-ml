# Homework 1 problem 1 - Attack implementations
# Sarah Scheffler
# Privacy in ML
# 10/3/18

import csv
import sys
from itertools import takewhile
from math import ceil
import numpy as np

REPETITIONS = int(20)
N_LIST = [128, 512, 2048, 8192]
SIGMA_LIST_GEN = lambda n: list(takewhile(lambda z: z >= 1/(32*n)**(1/2), [1/x for x in map(lambda y: 2**y, range(1, 10))])) # [1/2, 1/4, 1/8...] until 1/sqrt(32*n)
M_LIST_GEN = lambda n: [int(ceil(x*n)) for x in [1.1, 4, 16]]

########## Helper functions ########## 

def hadamard(n: int): 
    """Recursively generate the (n x n) Hadamard matrix with entries in {-1,1}"""
    n = int(n)
    assert '{0:b}'.format(n).count('1') == 1, "n not a power of 2"
    if n == 1:
        return np.ones(1)
    else:
        prev_hadamard = hadamard(n/2)
        to_ret = np.zeros((n, n))
        to_ret[0:n//2, 0:n//2] = prev_hadamard
        to_ret[n//2:n, 0:n//2] = prev_hadamard
        to_ret[0:n//2, n//2:n] = prev_hadamard
        to_ret[n//2:n, n//2:n] = (int(-1)) * prev_hadamard
        return to_ret

def rand_nbits(n: int):
    """Generate an n-element column vector of random integers in {0,1}"""
    n = int(n)
    return np.random.randint(2, size=n)

def sample_normal(sigma, size=None):
    """Wrapper for sampling from normal distribution"""
    return np.random.normal(scale=sigma, size=size)

def experiment(n, sigma, m, queries, attacker_guess):
    """Taking in a matrix of queries and a functor for attacker strategy, determine attacker success rate"""
    Q = queries
    def answers(Q, x, Y):
        return (1/n) * Q.dot(x) + Y
    results = []
    reps = 0
    print("Running {0:d} iterations of experiment with n={1:d}, sigma={2:f}, m={3:d}...".format(REPETITIONS, n, sigma, m))
    for i in range(REPETITIONS):
        print("    repetition {0:d}/20".format(i+1))
        try:
            x = rand_nbits(n)
            Y = sample_normal(sigma, size=m)
            a = answers(Q, x, Y)
            xhat = attacker_guess(Q, a)
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


########## Hadamard attack implementation ########## 

def hadamard_experiment(n, sigma, m=None):
    """Returns (mean, std) of fraction of correct results for Hadamard attack with this n and sigma"""
    H = hadamard(n)
    def attacker_guess(H, a):
        """attacker guess = H*a rounded to {0,1}"""
        return np.fromiter((z > 0.5 for z in H.dot(a)), int)
    return experiment(n, sigma, n, H, attacker_guess)


def random_query_experiment(n, sigma, m):
    """Returns (mean, std) of fraction of correct results for random query attack with this n, sigma, and m"""

    B = np.random.randint(2, size=(m,n)) # random (m x n) matrix with elems in {0,1}

    def attacker_guess(B, a):
        """Compute argmin_y of ||a-(1/b)By||_2, then return the rounded result"""
        return np.fromiter((z > 0.5 for z in np.linalg.lstsq((1/n)*B, a)[0]), int)
    return experiment(n, sigma, m, B, attacker_guess)


########## Unit tests ########## 

def test_hadamard():
    print("2", hadamard(int(2)))
    try:
        print("3", hadamard(int(3)))
    except AssertionError:
        print("3", "assert failed successfully")
    print("4", hadamard(int(4)))
    print("8", hadamard(int(8)))

def test_rand_nbits():
    print(rand_nbits(10))
    print(rand_nbits(10))
    print(rand_nbits(4))
    print(rand_nbits(5))

def test_evaluate_hadamard():
    print(hadamard_experiment(8, 0.5))
    print(hadamard_experiment(4, 0.5))
    print(hadamard_experiment(2, 0.25))
    print(hadamard_experiment(8, 0.25))
    print(hadamard_experiment(2048, 0.25))

def read_experiment_csv_dict(filename):
    """Returns a list of data points read from a csv into a dict (n,sigma,m) -> (mean,stdev,reps)"""
    to_ret = {}
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            to_ret[(int(row["n"]), float(row["sigma"]), int(row["m"]))] = (
                    float(row["mean"]) if len(row["mean"]) > 0 else None, 
                    float(row["stdev"]) if len(row["stdev"])>0 else None, 
                    int(row["repetitions"])
                    )
    return to_ret

def generate_experiment_csv(filename, n_list, sigma_list_gen, m_list_gen, experiment):
    already_done = read_experiment_csv_dict(filename)
    with open(filename, 'a') as csvfile:
        fw = csv.writer(csvfile, delimiter=',')
        if len(already_done) == 0:
            fw.writerow(["n","sigma","m","mean","stdev","repetitions"])
        for n in n_list:
            for sigma in sigma_list_gen(n):
                for m in m_list_gen(n):
                    if (n,sigma,m) not in already_done:
                        mean, std, reps = experiment(n=n, sigma=sigma, m=m)
                        fw.writerow([n, sigma, m, mean, std, reps])

def run_one_experiment(filename, n, sigma, m, experiment):
    already_done = read_experiment_csv_dict(filename)
    if (n,sigma,m) not in already_done:
        mean, std, reps = experiment(n=n, sigma=sigma, m=m)
        with open(filename, 'a') as csvfile:
            fw = csv.writer(csvfile, delimiter=',')
            fw.writerow([n, sigma, m, mean, std, reps])
    print(n, sigma, m, ": Written to file")



#test_hadamard()
#test_rand_nbits()
#test_evaluate_hadamard()
#generate_experiment_csv("hadamard_results.csv", N_LIST, SIGMA_LIST_GEN, lambda _: [None], hadamard_experiment)
#generate_experiment_csv("random_query_results.csv", N_LIST, SIGMA_LIST_GEN, M_LIST_GEN, random_query_experiment)
#generate_experiment_csv("2test.csv", N_LIST, SIGMA_LIST_GEN, M_LIST_GEN, random_query_experiment)
if len(sys.argv) < 4:
    print("Usage: python3 n sigma m")
    exit(0)
run_one_experiment("2test.csv", int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), random_query_experiment)






