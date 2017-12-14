# ALternative is preferential attachment, null is random degree sequence graph
# with matched deg. sequence.

import netcomp as nc
from joblib import Parallel, delayed  
import multiprocessing
import os
import networkx as nx
import pandas as pd
import time
import pickle
import numpy as np
from networkx import NetworkXError

####################################
### SET PARAMETERS
####################################

data_dir = "../pickled_data"

num_cores = multiprocessing.cpu_count()
# size of ensemble
ensemble_len = 500

n = 100
m = 2
p = (n-m)*m / (n*(n-1)/2) # so that volume of ER and BA graphs match


####################################
## DEFINE IMPORTANT FUNCTIONS
####################################


def distance(dist_func,A,B): return dist_func(A,B)
def deg_seq(G): return [item[1] for item in G.degree_iter()]

lambda_adj = lambda A1,A2: nc.lambda_dist(A1,A2,kind='adjacency')
lambda_lap = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian')
lambda_nlap = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian_norm')
res_dist = lambda A1,A2: nc.resistance_distance(A1,A2,check_connected=False)

distances = [nc.edit_distance,res_dist,nc.deltacon0,nc.netsimile,
            lambda_adj,lambda_lap,lambda_nlap]
labels = ['Edit','Resistance Dist.','DeltaCon','NetSimile',
          'Lambda (Adjacency)','Lambda (Laplacian)','Lambda (Normalized Laplacian)']


def grab_data(i,null=True):

    if i % 5 == 0 : print('ER/BA Distance Comparison. Iteration {}.'.format(i))
    # load pre-built graphs
    graph_dict = pickle.load(open(
        os.path.join(data_dir,'graphs/graph_{:03d}.p'.format(i)),'rb'))
    G1 = graph_dict['random 1']
    if null:
        G2 = graph_dict['random 2']
    else:
        G2 = graph_dict['pref. attachment']
        
    A1,A2 = [nx.adjacency_matrix(G).todense() for G in [G1,G2]]
 
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
             'p' : p,
             'm' : m,
             'results_df_null' : results_df_null,
             'results_df_not_null' : results_df_not_null,
             'description' : """Comparison of distances between two ER graphs,
             and an ER and BA graph. Uses given parameters, and calculates for a
             variety of distances. Adjacency distance is the distances between
             the true adjacency matrices."""}

# Ensure we don't inadvertently overwrite an extant file
file_name = 'PA_vs_randDS.p'
path = os.path.join(data_dir,file_name)
tag = 1
while os.path.isfile(path):
    file_name = 'PA_vs_randDS_{:02d}.p'.format(tag)
    path = os.path.join(data_dir,file_name)
    tag += 1

pickle.dump(data_dict,open(os.path.join(data_dir,file_name),'wb'))
