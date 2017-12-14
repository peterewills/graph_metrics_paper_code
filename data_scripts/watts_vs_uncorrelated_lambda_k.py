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

lambda_adj_1 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='adjacency',k=1)
lambda_lap_1 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian',k=1)
lambda_nlap_1 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian_norm',k=1)

lambda_adj_2 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='adjacency',k=2)
lambda_lap_2 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian',k=2)
lambda_nlap_2 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian_norm',k=2)

lambda_adj_5 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='adjacency',k=5)
lambda_lap_5 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian',k=5)
lambda_nlap_5 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian_norm',k=5)

lambda_adj_10 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='adjacency',k=10)
lambda_lap_10 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian',k=10)
lambda_nlap_10 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian_norm',k=10)

lambda_adj_20 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='adjacency',k=20)
lambda_lap_20 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian',k=20)
lambda_nlap_20 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian_norm',k=20)

distances = [lambda_adj_1,lambda_lap_1,lambda_nlap_1,
             lambda_adj_2,lambda_lap_2,lambda_nlap_2,
             lambda_adj_5,lambda_lap_5,lambda_nlap_5,
             lambda_adj_10,lambda_lap_10,lambda_nlap_10,
             lambda_adj_20,lambda_lap_20,lambda_nlap_20]

labels = ['Lambda (Adjacency, k=1)','Lambda (Laplacian, k=1)',
          'Lambda (Normalized Laplacian, k=1)',
          'Lambda (Adjacency, k=2)','Lambda (Laplacian, k=2)',
          'Lambda (Normalized Laplacian, k=2)',
          'Lambda (Adjacency, k=5)','Lambda (Laplacian, k=5)',
          'Lambda (Normalized Laplacian, k=5)',
          'Lambda (Adjacency, k=10)','Lambda (Laplacian, k=10)',
          'Lambda (Normalized Laplacian, k=10)',
          'Lambda (Adjacency, k=20)','Lambda (Laplacian, k=20)',
          'Lambda (Normalized Laplacian, k=20)']


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
## TAKE DATA 
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
