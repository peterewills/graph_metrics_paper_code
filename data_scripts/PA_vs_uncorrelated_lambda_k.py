# Same as PA_vs_uncorrelated.py, but for lamnda_k distances

import netcomp as nc

data_dir = "../pickled_data"

from joblib import Parallel, delayed
import multiprocessing
import os
import networkx as nx
import pandas as pd
import time
import pickle
import itertools as it
from tqdm import tqdm

####################################
### SET PARAMETERS
####################################

num_cores = multiprocessing.cpu_count()
# size of ensemble
ensemble_len = 500

n = 1000
m = 6
p = (n - m) * m / (n * (n - 1) / 2)  # so that volume of ER and BA graphs match


####################################
## DEFINE IMPORTANT FUNCTIONS
####################################


def distance(dist_func, A, B):
    return dist_func(A, B)


k_list = [1, 2, 10, 100, 300, 999]


def flatten(l):
    return [item for sublist in l for item in sublist]


distances = []
labels = []


def return_lambdas(k):
    return [
        lambda A1, A2: nc.lambda_dist(A1, A2, kind="adjacency", k=k),
        lambda A1, A2: nc.lambda_dist(A1, A2, kind="laplacian", k=k),
        lambda A1, A2: nc.lambda_dist(A1, A2, kind="laplacian_norm", k=k),
    ]


# can't make this work without using map
distances = list(map(return_lambdas, k_list))
distances = flatten(distances)

for k in k_list:

    labels_k = [
        "Lambda (Adjacency, k={})".format(k),
        "Lambda (Laplacian, k={})".format(k),
        "Lambda (Normalized Laplacian, k={})".format(k),
    ]

    labels += labels_k


def grab_data(i, null=True):

    G1 = nx.erdos_renyi_graph(n, p)
    if null:
        G2 = nx.erdos_renyi_graph(n, p)
    else:
        G2 = nx.barabasi_albert_graph(n, m)
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

print("ER/BA Lambda K Distance Comparison.")

if not os.path.exists(data_dir):
    raise IOError(f"Data directory {data_dir} does not exist!")

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

print("Alternative data complete. Total time elapsed: {} seconds.".format(end - start))

results_df_not_null = pd.concat(results_not_null, axis=1)


####################################
## PICKLE IT ON UP
####################################


data_dict = {
    "n": n,
    "p": p,
    "m": m,
    "results_df_null": results_df_null,
    "results_df_not_null": results_df_not_null,
    "description": """Comparison of distances between two ER graphs,
             and an ER and BA graph. Uses given parameters, and calculates for a
             variety of distances. Adjacency distance is the distances between
             the true adjacency matrices.

             This data only compares lambda distances, using different numbers
             of eigenvalues.""",
}

# Ensure we don't inadvertently overwrite an extant file
file_name = "PA_vs_uncorr_lambdak.p"
path = os.path.join(data_dir, file_name)
tag = 1
while os.path.isfile(path):
    file_name = "PA_vs_uncorr_lambdak_{:02d}.p".format(tag)
    path = os.path.join(data_dir, file_name)
    tag += 1

pickle.dump(data_dict, open(os.path.join(data_dir, file_name), "wb"))
