# Data Grabber for Distance Experiments between Random and Barabasi-Albert
# graphs. Random graph is degree-matched with BA graph.

import netcomp as nc
data_dir = "../pickled_data"
input_data_dir = "../data"

from joblib import Parallel, delayed  
import multiprocessing
import os
import random
import pandas as pd
import time
import pickle
import numpy as np
import scipy.io

####################################
### SET PARAMETERS
####################################

num_cores = multiprocessing.cpu_count()
# size of ensemble
ensemble_len = 500
threshold = 0.5
ones = True


####################################
## DEFINE IMPORTANT FUNCTIONS
####################################


def distance(dist_func,A,B): return dist_func(A,B)

lambda_adj = lambda A1,A2: nc.lambda_dist(A1,A2,kind='adjacency')
lambda_lap = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian')
lambda_nlap = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian_norm')


lambda_adj_5 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='adjacency')
lambda_lap_5 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian')
lambda_nlap_5 = lambda A1,A2: nc.lambda_dist(A1,A2,kind='laplacian_norm')

res_dist = lambda A1,A2: nc.resistance_distance(A1,A2,check_connected=False)

distances = [nc.edit_distance,res_dist,nc.deltacon0,nc.netsimile,
             lambda_adj,lambda_lap,lambda_nlap,lambda_adj_5,lambda_lap_5,
             lambda_nlap_5]
labels = ['Edit','Resistance Dist.','DeltaCon','NetSimile',
          'Lambda (Adjacency)','Lambda (Laplacian)',
          'Lambda (Normalized Laplacian)', 'Lambda (Adj. k=5)',
          'Lambda (Lap. k=5)', 'Lambda (N. Lap. k=5)']


def mask(matrix,threshold,ones=False):
    """Threshold an input matrix.
    
    Parameters
    ----------
    matrix : numpy array
        The matrix to be thresholded
        
    threshold : float
        Entries of matrix below threshold will be set to zero
        
    ones : Boolean (optional, default = False)
        If true, then entries above threshold are set to one. Otherwise, they are not modified.
        
    Returns
    -------
    masked : numpy array
        The input matrix, with entries below threshold set to zero.
    """
    if ones:
        masked = np.where(matrix > threshold, np.ones(matrix.shape), np.zeros(matrix.shape))
    else:
        masked = np.where(matrix > threshold, matrix, np.zeros(matrix.shape))
    return masked


def grab_data(i,null=True):

    if i % 100 == 0 : print(
            'ABIDE Autism Data Comparison. Iteration {}.'.format(i)
    )
    
    C1 = random.choice(TD_brains)
    if null:
        C2 = random.choice(TD_brains)
    else:
        C2 = random.choice(ASD_brains)
        
    C1,C2 = [C - np.diag(np.diag(C)) for C in [C1,C2]] # kill diagonal
    C1,C2 = [mask(C,threshold,ones=ones) for C in [C1,C2]]
    
    corr_distances = pd.Series([distance(dfun,C1,C2) for dfun in distances],
                               index = labels, name = 'Correlation Distances')
        
    return corr_distances

####################################
## IMPORT DATA
####################################

abide_data = scipy.io.loadmat(os.path.join(input_data_dir,'abide_connectivity.mat'))
correlations = np.abs(abide_data['correlation'])
diagnoses = abide_data['diagnosis'] - 1
diagnoses = [bool(item) for item in diagnoses.flatten()]

ASD_brains = list(correlations[diagnoses,:,:])
TD_brains = list(correlations[[~item for item in diagnoses],:,:])


####################################
## TAKE DATA 
####################################

print('Running on {} cores.'.format(num_cores))

start = time.time()
results_null = Parallel(n_jobs=num_cores)(delayed(grab_data)(i)
                                          for i in range(ensemble_len))
end = time.time()

print('Alternative complete. Total time elapsed: {} seconds.'.format(end-start))

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


data_dict = {'results_df_null' : results_df_null,
             'results_df_not_null' : results_df_not_null,
             'threshold' : threshold,
             'ones' : ones,
             'description' : """Comaprison of distance between two TD brains and
             a TD vs ASD brain, using ABIDE autism dataset. Thresholding is
             performed according to provided parameters. Parameter 'ones'
             indicates whether edges are weighted (ones=False) or not 
             (ones=True)."""
             }

# Ensure we don't inadvertently overwrite an extant file
file_name = 'abide_t_{}.p'.format(threshold)
path = os.path.join(data_dir,file_name)
tag = 1
while os.path.isfile(path):
    file_name = 'abide_t_{}_{:02d}.p'.format(threshold,tag)
    path = os.path.join(data_dir,file_name)
    tag += 1

pickle.dump(data_dict,open(os.path.join(data_dir,file_name),'wb'))
