import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def get_dist_matrix(S, T):
    '''
    Compute the matrix with distances between S, T. 

    S: source set, provided as a matrix with columns as the points.
    T: target set, provided as a matrix with columns as the points.
    '''

    s, t = np.shape(S)[1], np.shape(T)[1]
    length_S = np.outer(np.linalg.norm(S, axis = 0)**2,  np.ones(t))
    length_T = np.outer(np.ones(s), np.linalg.norm(T, axis = 0)**2)
    dist_matrix_squared = length_S + length_T - 2 * S.T @ T
    # print(dist_matrix_squared[dist_matrix_squared < 0])
    dist_matrix_squared[dist_matrix_squared < 0] = 0 # Is this necessary?
    return np.sqrt(dist_matrix_squared)


def weighted_Wasserstein_distance(S, T, dist_matrix = None):
    '''
    Compute the weighted Wasserstein distance of sets S, T. 

    S: source set, provided as a matrix with columns as the points.
    T: target set, provided as a matrix with columns as the points.
    dist_matrix: S x T sized matrix, giving distances between the corresponding points. If not given, then computes it.
    '''
    if dist_matrix.dtype != np.float64:
        dist_matrix = get_dist_matrix(S, T)
    
    # Since you're allowed to choose the weights, you just assign all the weight for t to the point in S closest to t.
    minimum_distances = np.min(dist_matrix, axis = 0)
    return np.mean(minimum_distances)



def find_Set(S, T, num_points):
    '''
    Given a source distribution (as a set of points) and a target distribution (as a set of points), finds a subset M of size
    num_points such that the weighted Wasserstein distance between S union M and T is (approximately) minimized.
    For this, a greedy algorithm is used. 

    S: the source set, provided as a matrix with columns as the points. 
    T: the target set, provided as a matrix with columns as the points.
    num_points: integer, the number of points added to S.
    '''

    all_points = np.concatenate([S,T], axis = 1)
    # print(all_points.shape)
    size_S, size_T = np.shape(S)[1], np.shape(T)[1]
    total_points = size_S + size_T
    
    # These are indicator vectors for the sets S, T.
    ind_S, ind_T = np.zeros(total_points, dtype = bool), np.zeros(total_points, dtype = bool)
    ind_S[:size_S] = True
    ind_T[size_S:] = True

    dist_matrix = get_dist_matrix(all_points, T)
    # We greedily choose points in T which have the smallest WWS after adding to S.
    for i in range(num_points):
        WWS_dists = 1000 * np.ones(size_T)
        
        for u in range(size_T):
            ind_S_u = deepcopy(ind_S)
            ind_S_u[size_S+u] = True
            # print(dist_matrix.shape)
            # print(np.sum(ind_S_u))
            # print(ind_T.shape)
            WWS_dists[u] = weighted_Wasserstein_distance(all_points[:, ind_S_u], T, dist_matrix[ind_S_u, :])
        
        u_best = np.argmin(WWS_dists)
        ind_S[size_S+u_best] = True

    new_S = all_points[:, ind_S]

    return new_S