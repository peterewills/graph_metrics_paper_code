# Alternative is N by M 2D lattice, alternative is random degree sequence graph
# with same deg. sequence as lattice.

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
N = 25
M = 4

G = nx.grid_2d_graph(N,M)
deg_seq = [item[1] for item in G.degree_iter()]

####################################
## DEFINE IMPORTANT FUNCTIONS
####################################


def distance(dist_func,A,B): return dist_func(A,B)

lambda_adj = lambda A1,A2: nc.lambda_dist(A1,A2,kind='adjacency')
lambda_lap = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian')
lambda_nlap = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian_norm')
res_dist = lambda A1,A2: nc.resistance_distance(A1,A2,check_connected=False)

distances = [nc.edit_distance,res_dist,nc.deltacon0,nc.netsimile,
            lambda_adj,lambda_lap,lambda_nlap]
labels = ['Edit','Resistance Dist.','DeltaCon','NetSimile',
          'Lambda (Adjacency)','Lambda (Laplacian)','Lambda (Normalized'
          ' Laplacian)']


def grab_data(i,null=True):

    
    if i % 100 == 0 : print('Lattice Distance Comparison, Iteration {}.'.format(i))
    
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
print('Lattice Distance Comparison.')

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
             'N' : 10,
             'M' : 10,
             'results_df_null' : results_df_null,
             'results_df_not_null' : results_df_not_null,
             'description' : """Comparison of distances between two random
             graphs with given degree sequence, and a grid graph, with same
             degree sequence."""}

# Ensure we don't inadvertently overwrite an extant file
file_name = 'lattice_vs_randDS_{}x{}.p'.format(N,M)
path = os.path.join(data_dir,file_name)
tag = 1
while os.path.isfile(path):
    file_name = 'lattice_vs_randDS_{}x{}_{:02d}.p'.format(N,M,tag)
    path = os.path.join(data_dir,file_name)
    tag += 1

pickle.dump(data_dict,open(os.path.join(data_dir,file_name),'wb'))
