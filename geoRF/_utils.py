from . import _tree
from ._tree import evaluate_split
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from time import time
import json
import copy


def distgfs_f(pp, pid):
    #print(pid, pp)
    x = [i[1] for i in pp.items() if i[0].startswith('x')]
    candidate_split = pp['ind_to_split_func'](x)
    X = pp['X']
    y = pp['y']
    parent_mse = pp['parent_mse']
    n = pp['n']
    #logger.info(f"Iter: {pid}\t x:{x}, y:{y}, X:{X}")
    return evaluate_split(candidate_split, X, y, parent_mse, n)

def gfsopt_f(x, X, y, parent_mse, n, ind_to_split_func, pid):
    #print(pid)
    candidate_split = ind_to_split_func(x)
    return evaluate_split(candidate_split, X, y, parent_mse, n)

def plot_decision_boundary(X, y, yhat, f1=0, f2=1, steps=101, show_points=True, show_errors=False, vmin=None, vmax=None):
    mx = np.linspace(min(X[:,f1]), max(X[:,f1]), steps)
    my = np.linspace(min(X[:,f2]), max(X[:,f2]), steps)
    xx, yy = np.meshgrid(mx, my)
    # grid = np.column_stack((xx.ravel(), yy.ravel()))
    grid = np.zeros((steps**2,X.shape[1]))
    # fill grid with averages of other columns
    for i in range(0, X.shape[1]):
        # first define new column
        new_col = None
        if i == f1:
            new_col = xx.ravel()
        elif i == f2:
            new_col = yy.ravel()
        else:
            new_col = np.array([np.mean(X[:,i]) for _ in range(steps**2)])
        
        # then add new column to grid
        grid[:,i] = new_col
    if vmin is None:
        vmin = np.min(y)
        vmax = np.max(y)
    fig = plt.figure()
    plt.scatter(grid[:,f1], grid[:,f2], c=yhat(grid), zorder=1, vmin=vmin, vmax=vmax)
    plt.colorbar()
    if show_points:
        plt.scatter(X[:,f1], X[:, f2], c=y, s=0.5, vmin=vmin, vmax=vmax)
    if show_errors:
        errors= np.abs(y - yhat(X))
        plt.scatter(X[:,f1], X[:, f2], c=errors, s=0.5, cmap='YlOrRd')
        plt.colorbar()

def sk_tree(X, y, r=2):
    tree = DecisionTreeRegressor(max_depth=r)
    tree.fit(X, y)
    return tree

def calc_node_metrics(node_metrics, n, gtree, X, y, X_test, y_test, start, end=time()):
    node_metrics[n] = {}
    node_metrics[n] = _tree.calc_metrics(y, gtree.predict(X))
    node_metrics[n]['time'] = time()-start
    if X_test is not None:
        metrics_test = _tree.calc_metrics(y_test, gtree.predict(X_test), "test")
        node_metrics[n].update(metrics_test)
    gtree.set_node_metrics(node_metrics)

# CONTINUE_NONE: True for keep growing a node with None split. False for stop growing after one iteration with None split.
# RESTART: if >0, keep evaluating splits for a generator until n_RESTART or until it returns a not None split
# geo_features needs to be a number, is used in a range, so if 1 then 0,1 are geo features.
def simple_tree(X, y, generators, r=2, 
                geo_features=None, 
                CONTINUE_NONE=True, 
                n_RESTART=0, 
                X_test=None, y_test=None, 
                save_fig=False, 
                geosplits=None, min_area=0.0,
                early_stopping=None,
                min_gain=0.0
               ):
    start = time()
    trees = []
    metrics_r = {}
    node_metrics = {}
    n=1
    X = np.float32(X)
    y = np.float64(y)
    gtree = _tree.Tree(X, y)
    
    for i in range(r):
        for leaf in gtree.get_leafs():
            calc_node_metrics(node_metrics, n, gtree, X,y,X_test,y_test, start)
            if len(leaf.idx) < 2:
                continue
            elif (len(leaf.idx) == 2) & ((leaf.X[0]==leaf.X[1]).all()):
                continue
                
            leaf.grow(generators=generators, 
                      geo_features=geo_features, 
                      CONTINUE_NONE=CONTINUE_NONE, 
                      n_RESTART=n_RESTART, 
                      geosplits=geosplits, 
                      min_area=min_area,
                      min_gain=min_gain
                     )
            n+=1
        
        end = time()
        calc_node_metrics(node_metrics, n, gtree, X,y,X_test,y_test, start,end)
        # After each depth increase, calculate metrics
        n_leaves = gtree.get_n_leaves()
        metrics_r[n_leaves] = {}
        metrics_r[n_leaves] = _tree.calc_metrics(y, gtree.predict(X))
        metrics_r[n_leaves]['time'] = end-start
        metrics_r[n_leaves]['elli_area'] = gtree.get_avg_elli_area(i)
        metrics_r[n_leaves]['ortho_ratio'],metrics_r[n_leaves]['diag_ratio'],metrics_r[n_leaves]['elli_ratio'] = gtree.get_split_ratios()
        if X_test is not None:
            metrics_test = _tree.calc_metrics(y_test, gtree.predict(X_test), "test")
            metrics_r[n_leaves].update(metrics_test)
            if early_stopping:
                if (i>0) and (trees[-1][0] < metrics_test['maetest']):
                    gtree = trees[-1][1]
                    break
                else:
                    t = copy.deepcopy(gtree)
                    trees.append([metrics_test['maetest'],t])
        print(str(i+1),n_leaves,metrics_r[n_leaves])
        gtree.set_metrics(metrics_r)
        if save_fig:
            plot_decision_boundary(X,y,gtree.predict)
            plt.savefig(save_fig+f"_{i}.png")
    return gtree

def avg_ensemble(X, y, generators, r=2, 
                geo_features=None, 
                CONTINUE_NONE=True, 
                n_RESTART=0, 
                X_test=None, y_test=None, 
                save_fig=False, 
                geosplits=None, min_area=0.0,
                early_stopping=None,
                min_gain=0.0
               ):
    trees = []
    for i in range(3):
        t = simple_tree(X,y, generators, r, geo_features, CONTINUE_NONE,n_RESTART,X_test,y_test,save_fig,geosplits,min_area,early_stopping,min_gain)
        trees.append(t)
    y_hat = np.mean([t.predict(X) for t in trees], axis=0)
    metrics = {}
    metrics = _tree.calc_metrics(y,y_hat)
    y_hat_test = np.mean([t.predict(X_test) for t in trees], axis=0)
    metrics.update(_tree.calc_metrics(y_test,y_hat_test, "test"))
    print(metrics)
    return trees

    

def print_node_recurse(node, spacing):
    indent = ("|" + (" " * spacing)) * node.depth
    indent = indent[:-spacing] + "-" * spacing
    if node.is_leaf():
        print(indent, node.yhat, len(node.idx))
    else:
        print(indent, node.split, len(node.idx))
    if node.left:
        print_node_recurse(node.left)
    if node.right:
        print_node_recurse(node.right)

def print_tree(tree, spacing=3):
    print_node_recurse(tree.root)

def univariate_function(aik):
    f1 = (1/(1+aik))
    f2 = 3 * np.power(np.e, -50 * ((aik - 0.3)**2))
    f3 = 2 * np.power(np.e, -25 * ((aik - 0.7)**2))
    return f1 + f2 + f3

def make_regression_bivariate(n_samples):
    X = []
    s = []
    for i in range(n_samples):
        ai1, ai2 = np.random.uniform(0, 1, 2)
        si = (ai1, ai2)
        Xi1 = univariate_function(ai1)
        Xi2 = univariate_function(ai2)
        Xi = Xi1 * Xi2
        s.append(si)
        X.append(Xi)
    return (np.array(s), np.array(X))


