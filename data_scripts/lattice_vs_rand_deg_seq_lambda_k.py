# Same as lattics_vs_rand_deg_seq.py, but for lambda_k distances.

import netcomp as nc
from joblib import Parallel, delayed  
import multiprocessing
import os
import networkx as nx
import pandas as pd
import time
import pickle
import numpy as np

####################################
### SET PARAMETERS
####################################

data_dir = "../pickled_data"

num_cores = multiprocessing.cpu_count()
# size of ensemble
ensemble_len = 500

n = 100
N = 10
M = 10

G = nx.grid_2d_graph(N,M)
deg_seq = [item[1] for item in G.degree_iter()]

####################################
## DEFINE IMPORTANT FUNCTIONS
####################################


def distance(dist_func,A,B): return dist_func(A,B)

k_list = [1,2,3,4,5,10,20,50,75,90,99]

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

    if i % 100 == 0 : print('Iteration {}.'.format(i))
    
    G1 = nx.random_degree_sequence_graph(deg_seq)
    if null:
        G2 = nx.random_degree_sequence_graph(deg_seq)
    else:
        G2 = nx.grid_2d_graph(N,M)
    
    A1,A2 = [nx.adjacency_matrix(G).todense() for G in [G1,G2]]
    adj_distances = pd.Series([distance(dfun,A1,A2) for dfun in distances],
                              index = labels, name = 'Adjacency Distances')
    
    data = pd.concat([adj_distances],axis=1)
    
    return data


####################################
## TAKE DATA 
####################################

print('Running on {} cores.'.format(num_cores))
print('Lattice Lambda K Distance Comparison.')

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
             'N' : N,
             'M' : M,
             'results_df_null' : results_df_null,
             'results_df_not_null' : results_df_not_null,
             'description' : """Comparison of distances between two random
             graphs with given degree sequence, and a grid graph, with same
             degree sequence.

             This dataset compares lambda k distances for various k."""}

# Ensure we don't inadvertently overwrite an extant file
file_name = 'lattice_vs_randDS_lambda_k_{}x{}.p'.format(N,M)
path = os.path.join(data_dir,file_name)
tag = 1
while os.path.isfile(path):
    file_name = 'lattice_vs_randDS_lambda_k_{}x{}_{:02d}.p'.format(N,M,tag)
    path = os.path.join(data_dir,file_name)
    tag += 1

pickle.dump(data_dict,open(os.path.join(data_dir,file_name),'wb'))
