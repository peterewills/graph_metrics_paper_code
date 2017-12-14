# Same as watts_vs_uncorrelated_lambda_k.py, but for larger values of k

import netcomp as nc
from joblib import Parallel, delayed  
import multiprocessing
import os
import networkx as nx
import pandas as pd
import time
import pickle
import random
import numpy as np
import itertools as it


####################################
### SET PARAMETERS
####################################

data_dir = "../pickled_data"

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

def shuffle_vertex_labels(A):
    """Shuffle vertex labels on a graph with adjacency matrix A."""
    n,m = A.shape
    if n != m:
        raise ValueError('Input matrix must be square.')
    inds = list(range(n))
    random.shuffle(inds)
    A_shuff = A[np.ix_(inds,inds)]
    return A_shuff


def distance(dist_func,A,B): return dist_func(A,B)


lambda_adj_50 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='adjacency',k=50)
lambda_lap_50 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian',k=50)
lambda_nlap_50 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian_norm',k=50)

lambda_adj_75 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='adjacency',k=75)
lambda_lap_75 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian',k=75)
lambda_nlap_75 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian_norm',k=75)

lambda_adj_90 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='adjacency',k=90)
lambda_lap_90 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian',k=90)
lambda_nlap_90 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian_norm',k=90)

lambda_adj_99 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='adjacency',k=99)
lambda_lap_99 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian',k=99)
lambda_nlap_99 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian_norm',k=99)

distances = [lambda_adj_50,lambda_lap_50,lambda_nlap_50,
             lambda_adj_75,lambda_lap_75,lambda_nlap_75,
             lambda_adj_90,lambda_lap_90,lambda_nlap_90,
             lambda_adj_99,lambda_lap_99,lambda_nlap_99]

labels = ['Lambda (Adjacency, k=50)','Lambda (Laplacian, k=50)',
          'Lambda (Normalized Laplacian, k=50)',
          'Lambda (Adjacency, k=75)','Lambda (Laplacian, k=75)',
          'Lambda (Normalized Laplacian, k=75)',
          'Lambda (Adjacency, k=90)','Lambda (Laplacian, k=90)',
          'Lambda (Normalized Laplacian, k=90)',
          'Lambda (Adjacency, k=99)','Lambda (Laplacian, k=99)',
          'Lambda (Normalized Laplacian, k=99)']


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

print('Not null data complete. Total time elapsed: {} seconds.'.format(end-start))

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
file_name = 'watts_vs_uncorrelated_bigk.p'
path = os.path.join(data_dir,file_name)
tag = 1
while os.path.isfile(path):
    file_name = 'watts_vs_uncorrelated_bigk_{:02d}.p'.format(tag)
    path = os.path.join(data_dir,file_name)
    tag += 1

pickle.dump(data_dict,open(os.path.join(data_dir,file_name),'wb'))
