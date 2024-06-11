import numpy as np
from ._splits import *
from numba import jit, prange
from ._tree import TREE_BALANCE_BIAS 

FEATURE_THRESHOLD = 1e-7

@jit(nopython=True, nogil=True, fastmath=True)
def balance_correction(k, num):
    x = np.linspace(0, 1, num+2)
    c = .00001; x[0] += c; x[-1] -= c
    r = .5 + np.log(x / (1 - x)) / k
    return np.clip(r[1:-1], 0, 1)

@jit(nopython=True, nogil=True, fastmath=True)
def mse(y, y_hat):
    diffsq = 0
    for k in range(y.shape[0]):
        diffsq += (y[k] - y_hat)**2
    return diffsq / y.shape[0]

@jit(nopython=True, nogil=True, fastmath=True)
def euclidean(p1x, p1y, p2x, p2y):
    return ((p1x - p2x)**2 + (p1y - p2y)**2)**(.5)

@jit(nopython=True, nogil=True, fastmath=True)
def calc_gain(y, tidx, parent_mse, n):
    num_obs = y.shape[0]
    left_y, right_y = y[~tidx], y[tidx]
    num_left = left_y.shape[0]
    num_right = right_y.shape[0]
    if not num_left or not num_right:
        return 0
    left_mse = mse(left_y, np.mean(left_y))
    right_mse = mse(right_y, np.mean(right_y))
    left_ratio = (num_left / num_obs)
    if TREE_BALANCE_BIAS > 0:
        err = balance_correction(TREE_BALANCE_BIAS, len(y))
        left_ratio = err[len(left_y)]
    right_ratio = 1 - left_ratio
    weight = num_obs / n
    gain = weight * (parent_mse - (left_mse * left_ratio) - (right_mse * right_ratio))
    return np.float64(gain)

@jit(nopython=True, nogil=True, fastmath=True)
def ellipse_is_true(X, f1, f2, focal1x, focal1y, focal2x, focal2y, dist):
    dist_1 = np.sqrt((X[:, f1] - focal1x)**2 + (X[:, f2] - focal1y)**2)
    dist_2 = np.sqrt((X[:, f1] - focal2x)**2 + (X[:, f2] - focal2y)**2)
    return dist_1 + dist_2 >= dist

@jit(nopython=True, nogil=True, fastmath=True)
def diagonal_is_true(X, f1, f2, intercept, slope):
    return X[:,f2] >= intercept + slope * X[:,f1]

@jit(nopython=True, nogil=True, fastmath=True)
def orthogonal_is_true(X, f, split):
    return X[:,f] >= split

@jit(nopython=True, nogil=True, fastmath=True, cache=True, parallel=False)
def orthogonal_evaluate_candidates(X, y, parent_mse, features, n):
    best_mse_gain, best_split = 0.0, (None, None)
    for f in features:
        unique_sorted = np.sort(np.unique(X[:,f]))
        for i in range(1, len(unique_sorted)):
            if unique_sorted[i] <= unique_sorted[i-1] + FEATURE_THRESHOLD:
                continue
            split = (unique_sorted[i-1] + unique_sorted[i]) / 2
            if (split == unique_sorted[i] or
                split == np.inf or split == -np.inf):
                split = unique_sorted[i-1]
            tidx = orthogonal_is_true(X, f, split)
            gain = calc_gain(y, tidx, parent_mse,n)
            if gain >= best_mse_gain:
                best_mse_gain = gain
                best_split = (f, split)
    return best_mse_gain, best_split

@jit(nopython=True, nogil=True, fastmath=True)
def diagonal_evaluate_candidates(X, y, parent_mse, n):
    best_mse_gain, best_split = 0.0, (None, None, None, None)
    for f1 in range(X.shape[1]):
        for f2 in range(f1+1, X.shape[1]):
            for i in range(len(X)):
                for j in range(i+1, len(X)):
                    if X[j,f1] == X[i,f1]:
                        continue
                    slope = (X[j,f2] - X[i,f2]) / (X[j,f1] - X[i,f1])
                    if slope <= 0.:
                        continue
                    x1 = X[i,f1] + (X[j,f1] - X[i,f1]) / 2
                    x2 = slope * (x1 - X[i,f1]) + X[i,f2]
                    islope = -1/slope
                    iint = x2 - islope * x1
                    tidx = diagonal_is_true(X, f1, f2, iint, islope)
                    gain = calc_gain(y, tidx, parent_mse,n)
                    if gain >= best_mse_gain:
                        best_mse_gain = gain
                        best_split = (f1, f2, iint, islope)
    return best_mse_gain, best_split

@jit(nopython=True, nogil=True, fastmath=True)
def ellipse_evaluate_candidates(X, y, parent_mse, n):
    best_mse_gain, best_split = 0.0, (None, None, None, None, None, None, None)
    for f1 in range(X.shape[1]):
        for f2 in range(f1+1, X.shape[1]):
            f12 = np.array([f1, f2])
            for i in range(len(X)):
                xi = X[i,:]
                for j in range(i, len(X)):
                    xj = X[j,:]
                    xh = (xi[f12] + xj[f12]) / 2.0
                    for k in range(len(X)):
                        if k == i or k == j:
                            continue
                        xk = X[k,:]
                        distance = euclidean(xk[f1], xk[f2], xh[f1], xh[f2])
                        distanceij = euclidean(xi[f1], xi[f2], xj[f1], xj[f2])
                        if distance < distanceij:
                            continue
                        tidx = ellipse_is_true(X, f1, f2, xi[f1], xi[f2], xj[f1], xj[f2], distance)
                        gain = calc_gain(y, tidx, parent_mse,n)
                        if gain >= best_mse_gain:
                            best_mse_gain = gain
                            best_split = (f1, f2, xi[f1], xi[f2], xj[f1], xj[f2], distance)
    return best_mse_gain, best_split

class OrthogonalSplitGenerator:       
    @staticmethod
    def generate_candidates(X, y, parent_mse, features, geo_features, n, bbox, random_state):
        gain, (f, split) = orthogonal_evaluate_candidates(X, y, parent_mse, features, n)
        yield OrthogonalSplit(f, split)

class DiagonalSplitGenerator:
    @staticmethod
    def generate_candidates(X, y, parent_mse, features, geo_features, n, bbox, random_state):
        gain, (f1, f2, iint, islope) = diagonal_evaluate_candidates(X, y, parent_mse, n)
        yield DiagonalSplit(f1, f2, iint, islope)

class EllipseSplitGenerator:
    @staticmethod
    def generate_candidates(X, y, parent_mse, features, geo_features, n, bbox, random_state):
        gain, (f1, f2, xif1, xif2, xjf1, xjf2, distance) = ellipse_evaluate_candidates(X, y, parent_mse, n)
        yield EllipseSplit(f1, f2, np.array([xif1, xif2]), np.array([xjf1, xjf2]), distance, bbox)




