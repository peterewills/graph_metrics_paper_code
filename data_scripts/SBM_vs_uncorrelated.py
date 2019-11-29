# Null is uncorrelated random graph, alternative is stochastic blockmodel

import netcomp as nc
from joblib import Parallel, delayed
import multiprocessing
import os
import networkx as nx
import pandas as pd
import time
import pickle
import numpy as np
from tqdm import tqdm

####################################
### SET PARAMETERS
####################################

data_dir = "../pickled_data"

num_cores = multiprocessing.cpu_count()
# size of ensemble
ensemble_len = 500

n = 1000
p = 0.02

# SBM parameters, must average to p so that we have same volume as ER graph
# (note there are 2 partitions of equal size).
pp = 1.9 * p
qq = 0.1 * p


####################################
## DEFINE IMPORTANT FUNCTIONS
####################################


def distance(dist_func, A, B):
    return dist_func(A, B)


lambda_adj = lambda A1, A2: nc.lambda_dist(A1, A2, kind="adjacency")
lambda_lap = lambda A1, A2: nc.lambda_dist(A1, A2, kind="laplacian")
lambda_nlap = lambda A1, A2: nc.lambda_dist(A1, A2, kind="laplacian_norm")
res_dist = lambda A1, A2: nc.resistance_distance(A1, A2, check_connected=False)

distances = [
    nc.edit_distance,
    res_dist,
    nc.deltacon0,
    nc.netsimile,
    lambda_adj,
    lambda_lap,
    lambda_nlap,
]
labels = [
    "Edit",
    "Resistance Dist.",
    "DeltaCon",
    "NetSimile",
    "Lambda (Adjacency)",
    "Lambda (Laplacian)",
    "Lambda (Normalized Laplacian)",
]


def grab_data(i, null=True):

    G1 = nx.erdos_renyi_graph(n, p)
    if null:
        G2 = nx.erdos_renyi_graph(n, p)
    else:
        G2 = nx.planted_partition_graph(2, n // 2, pp, qq)
    A1, A2 = [nx.adjacency_matrix(G).todense() for G in [G1, G2]]

    adj_distances = pd.Series(
        [distance(dfun, A1, A2) for dfun in distances],
        index=labels,
        name="Adjacency Distances",
    )

    data = pd.concat([adj_distances], axis=1)

    return data


####################################
## TAKE DATA
####################################

print("Running on {} cores.".format(num_cores))

start = time.time()
results_null = Parallel(n_jobs=num_cores)(
    delayed(grab_data)(i) for i in tqdm(range(ensemble_len))
)
end = time.time()

print("Null data complete. Total time elapsed: {} seconds.".format(end - start))

results_df_null = pd.concat(results_null, axis=1)

start = time.time()
results_not_null = Parallel(n_jobs=num_cores)(
    delayed(grab_data)(i, null=False) for i in tqdm(range(ensemble_len))
)
end = time.time()

print("Not null data complete. Total time elapsed: {} seconds.".format(end - start))

results_df_not_null = pd.concat(results_not_null, axis=1)


####################################
## PICKLE IT ON UP
####################################


data_dict = {
    "n": n,
    "p": p,
    "results_df_null": results_df_null,
    "results_df_not_null": results_df_not_null,
    "description": """Comparison of distances between two ER graphs,
             and an ER and SBM graph. Uses given parameters, and calculates for a
             variety of distances. Adjacency distance is the distances between
             the true adjacency matrices.""",
}

# Ensure we don't inadvertently overwrite an extant file
file_name = "SBM_vs_uncorr.p"
path = os.path.join(data_dir, file_name)
tag = 1
while os.path.isfile(path):
    file_name = "SBM_vs_uncorr_{:02d}.p".format(tag)
    path = os.path.join(data_dir, file_name)
    tag += 1

pickle.dump(data_dict, open(os.path.join(data_dir, file_name), "wb"))
