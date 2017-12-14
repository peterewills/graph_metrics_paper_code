# Same as SBM_vs_uncorrelated.py, but for lambda_k distances. Can be run for
# various numberes of communities, designated below by variable l. 


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
p = 0.12
l = 2 # number of communities
n = n - n%l # ensures that n/l is an integer  
t = 0.05
# in the main body of the paper we use l=2 and t = 1/20

def pp(n,p,l,t):
    """calculate pp,qq for SBM given p from ER
    
    p : p from G(n,p)
    l : # of communities
    t : ratio of pp/qq
    """
    
    pp = p*n*(n-1) / ( n**2/l - n + t*n**2 *(l-1)/l)
    qq = t * pp
    return pp,qq

pp,qq = pp(n,p,l,t)


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

    if i % 100 == 0 : print('Iteration {}.'.format(i))
    
    G1 = nx.erdos_renyi_graph(n,p)
    if null:
        G2 = nx.erdos_renyi_graph(n,p)
    else:
        G2 = nx.planted_partition_graph(l,n//l,pp,qq)
    A1,A2 = [nx.adjacency_matrix(G).todense() for G in [G1,G2]]

    adj_distances = pd.Series([distance(dfun,A1,A2) for dfun in distances],
                              index = labels, name = 'Adjacency Distances')
    
    data = pd.concat([adj_distances],axis=1)
    
    return data


####################################
## TAKE DATA 
####################################

print('Running on {} cores.'.format(num_cores))
print('ER/SBM Lambda K Distance Comparison.')

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
             'p' : p,
             'l' : l,
             't' : t,
             'pp' : pp,
             'qq' : qq,
             'results_df_null' : results_df_null,
             'results_df_not_null' : results_df_not_null,
             'description' : """Comparison of distances between two ER graphs,
             and an ER and SBM graph. Uses given parameters, and calculates for a
             variety of distances. Adjacency distance is the distances between
             the true adjacency matrices.

             This dataset compares lambda k distances for various k."""}

# Ensure we don't inadvertently overwrite an extant file
file_name = 'SBM_vs_uncorr_lambda_k_l{}.p'.format(l)
path = os.path.join(data_dir,file_name)
tag = 1
while os.path.isfile(path):
    file_name = 'SBM_vs_uncorr_lambda_k_l{}_{:02d}.p'.format(l,tag)
    path = os.path.join(data_dir,file_name)
    tag += 1

pickle.dump(data_dict,open(os.path.join(data_dir,file_name),'wb'))
