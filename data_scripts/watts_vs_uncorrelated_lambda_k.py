# Same as watts_vs_uncorrelated.py, but for lambda_k distances.

import netcomp as nc
from joblib import Parallel, delayed  
import multiprocessing
import os
import networkx as nx
import pandas as pd
import time
import pickle
import numpy as np
import random

####################################
### SET PARAMETERS
####################################

data_dir = "../pickled_data"


def shuffle_vertex_labels(A):
    """Shuffle vertex labels on a graph with adjacency matrix A."""
    n,m = A.shape
    if n != m:
        raise ValueError('Input matrix must be square.')
    inds = list(range(n))
    random.shuffle(inds)
    A_shuff = A[np.ix_(inds,inds)]
    return A_shuff

num_cores = multiprocessing.cpu_count()
# size of ensemble
ensemble_len = 500

k = 4
n = 100
p = 0.0404
beta = 0.1

####################################
## DEFINE IMPORTANT FUNCTIONS
####################################


def distance(dist_func,A,B): return dist_func(A,B)

k_list = [1,2,5,10,20,50,75,90,99]

def flatten(l): return [item for sublist in l for item in sublist]

distances = []
labels = []

def return_lambdas(k):
    return [lambda A1,A2: nc.lambda_dist(A1,A2,kind='adjacency',k=k),
            lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian',k=k),
            lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian_norm',k=k)]

# can't make this work without using map
distances = list(map(return_lambdas,k_list))
distances = flatten(distances)

for k in k_list:

    labels_k = ['Lambda (Adjacency, k={})'.format(k),
                'Lambda (Laplacian, k={})'.format(k),
                'Lambda (Normalized Laplacian, k={})'.format(k)]

    labels += labels_k


def grab_data(i,null=True):

    if i % 100 == 0 : print('Watts-Strogatz Distance Comparison. Iteration {}.'.format(i))
    
    G1 = nx.erdos_renyi_graph(n,p)
    if null:
        G2 = nx.erdos_renyi_graph(n,p)
    else:
        G2 = nx.connected_watts_strogatz_graph(n,k,beta)
    A1,A2 = [nx.adjacency_matrix(G).todense() for G in [G1,G2]]
    A1,A2 = [shuffle_vertex_labels(A) for A in [A1,A2]]
    adj_distances = pd.Series([distance(dfun,A1,A2) for dfun in distances],
                              index = labels, name = 'Adjacency Distances')
    
    data = pd.concat([adj_distances],axis=1)
    
    return data


####################################
## Take DATA 
####################################

print('Running on {} cores.'.format(num_cores))

start = time.time()
results_null = Parallel(n_jobs=num_cores)(delayed(grab_data)(i)
                                          for i in range(ensemble_len))
end = time.time()

print('Null data complete. Total time elapsed: {} seconds.'.format(end-start))

results_df_null = pd.concat(results_null,axis=1)

start = time.time()
results_not_null = Parallel(n_jobs=num_cores)(delayed(grab_data)(i,null=False)
                                              for i in range(ensemble_len))
end = time.time()

print('Alternative data complete. Total time elapsed: {} seconds.'.format(end-start))

results_df_not_null = pd.concat(results_not_null,axis=1)


####################################
## PICKLE IT ON UP
####################################


data_dict = {'n' : n,
             'k' : k,
             'p' : p,
             'beta' : beta,
             'results_df_null' : results_df_null,
             'results_df_not_null' : results_df_not_null,
             'description' : """Comparison of distances between two ER graphs
             from G(n,P), vs distance between an ER graph and a SW graph with
             parameters n,k,beta."""}

# Ensure we don't inadvertently overwrite an extant file
file_name = 'watts_vs_uncorr_lambda_k.p'
path = os.path.join(data_dir,file_name)
tag = 1
while os.path.isfile(path):
    file_name = 'watts_vs_uncorr_lambda_k_{:02d}.p'.format(tag)
    path = os.path.join(data_dir,file_name)
    tag += 1

pickle.dump(data_dict,open(os.path.join(data_dir,file_name),'wb'))