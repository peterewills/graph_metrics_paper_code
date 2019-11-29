# Make some degree-matched preferential attachment graphs. The process of
# generating these graphs seems to get hung up quite often, so I've written a
# separate script to generate the graphs and save them one by one. The script
# PA_vs_randDS.py then compares these graphs.

from joblib import Parallel, delayed
import multiprocessing
import os
import networkx as nx
import pandas as pd
import time
import pickle
import numpy as np
from networkx import NetworkXError
from tqdm import tqdm

####################################
### SET PARAMETERS
####################################

data_dir = "../pickled_data"

num_cores = multiprocessing.cpu_count()
# size of ensemble
ensemble_len = 100

n = 1000
m = 6

####################################
## DEFINE IMPORTANT FUNCTIONS
####################################


def deg_seq(G):
    return [item[1] for item in G.degree_iter()]


# dump a list of graphs, without overwriting
def dump_graphs(g_list):
    # Make graphs into a dictionary
    G1, G2, G3 = g_list
    graph_dict = {"random 1": G1, "random 2": G2, "pref. attachment": G3}
    # figure out a file name that hasn't been used yet
    tag = 0
    file_name = "graphs/graph_{:03d}.p".format(tag)
    path = os.path.join(data_dir, file_name)
    while os.path.isfile(path):
        file_name = "graphs/graph_{:03d}.p".format(tag)
        path = os.path.join(data_dir, file_name)
        tag += 1
    # dump the dict at the specified location
    pickle.dump(graph_dict, open(path, "wb"))


def make_graphs(n, m, i, attempts=1):
    try:
        G3 = nx.barabasi_albert_graph(n, m)
        degs = deg_seq(G3)
        G1 = nx.random_degree_sequence_graph(degs, tries=2)
        G2 = nx.random_degree_sequence_graph(degs, tries=10)
        print("Iteration {} complete after {} attempts.".format(i, attempts))
        dump_graphs([G1, G2, G3])
    except NetworkXError:
        attempts += 1
        G1, G2 = make_graphs(n, m, i, attempts)
    return G1, G2


####################################
## TAKE DATA
####################################

print("Running on {} cores.".format(num_cores))

start = time.time()
results_null = Parallel(n_jobs=num_cores)(
    delayed(make_graphs)(n, m, i) for i in tqdm(range(ensemble_len))
)
end = time.time()

print("Graphs complete. Total time elapsed: {:.01f} seconds.".format(end - start))
