from ._tree import evaluate_split
from ._splits import *
from ._utils import distgfs_f
import distgfs
import numpy as np
from numba import jit, prange

EPSILON = 1e-7


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def euclidean(p1x, p1y, p2x, p2y):
    return ((p1x - p2x)**2 + (p1y - p2y)**2)**(.5)




def find_best_split(X, y, parent_mse, n, ind_to_split_func, bounds, maxiter):

    problem_parameters = {
        'X' : X,
        'y' : y,
        'parent_mse' : parent_mse,
        'n' : n,
        'ind_to_split_func' : ind_to_split_func
    }
    distgfs_params = {'opt_id': 'distgfs_test',
                  'obj_fun_name': 'distgfs_f',
                  'obj_fun_module': 'scikit_geodect._generators_gfs',
                  'problem_parameters': problem_parameters,
                  'space': bounds,
                  'n_iter': maxiter,
                  "n_max_tasks": 1,
                 }
    params, val = distgfs.run(distgfs_params, verbose=False)

    return ind_to_split_func([i[1] for i in params])

class DiagonalSplitGenerator:
    
    def __init__(self,
        maxiter=100, regf=0):
        self.maxiter = maxiter
    
    def generate_candidates(self, X, y, parent_mse, features, geo_features, n, bbox, random_state):
        if np.all(X[:,geo_features]==X[0,geo_features]):
            return None
        for i1, f1 in enumerate(geo_features):
            for f2 in geo_features[i1+1:]:
                def ind_to_split_func(individual):
                    # avoid division by zero
                    if individual[0] == individual[2]:
                        return None
                    slope = (individual[1] - individual[3]) / (individual[0] - individual[2])
                    intercept = individual[1] - slope * individual[0]
                    return DiagonalSplit(f1, f2, intercept, slope)
                min_f1 = min(X[:,f1])
                max_f1 = max(X[:,f1])+EPSILON
                min_f2 = min(X[:,f2])
                max_f2 = max(X[:,f2])+EPSILON
                bounds = {
                    'x0' : [min_f1,max_f1],
                    'x1' : [min_f2,max_f2],
                    'x2' : [min_f1,max_f1],
                    'x3' : [min_f2,max_f2]
                    }
                yield(find_best_split(X, y, parent_mse, n, ind_to_split_func,
                                         bounds, self.maxiter
                                    ))
                                

class EllipseSplitGenerator:
    
    def __init__(self,
        maxiter=100, regf=0):
        self.maxiter = maxiter
        self.regf = regf
        self.bbox = None
    
    def generate_candidates(self, X, y, parent_mse, features, geo_features, n, bbox, random_state):
        self.bbox = bbox
        if np.all(X[:,geo_features]==X[0,geo_features]):
            return None
        for i1, f1 in enumerate(geo_features):
            for f2 in geo_features[i1+1:]:
                def ind_to_split_func(individual):
                    # calculate distance for each combination of points in individual
                    p1, p2, d = individual[0:2], individual[2:4], individual[4]
                    return EllipseSplit(f1, f2, p1, p2, d, self.bbox, self.regf)
                min_f1 = min(X[:,f1])
                max_f1 = max(X[:,f1])+EPSILON
                min_f2 = min(X[:,f2])
                max_f2 = max(X[:,f2])+EPSILON
                max_dist = np.sqrt( (max_f1-min_f1)**2 + (max_f2-min_f2)**2 )
                bounds = {
                    'x0' : [min_f1,max_f1],
                    'x1' : [min_f2,max_f2],
                    'x2' : [min_f1,max_f1],
                    'x3' : [min_f2,max_f2],
                    'x4' : [0,max_dist]
                    }
                yield(find_best_split(X, y, parent_mse, n, ind_to_split_func,
                                         bounds, self.maxiter
                                    ))