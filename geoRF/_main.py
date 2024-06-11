"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import resample, check_random_state
from sklearn.ensemble._forest import _get_n_samples_bootstrap, _generate_sample_indices, _generate_unsampled_indices
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from time import time
from . import _tree, _utils, _generators_numba, _generators_da, _generators_gfs,_generators_sklearn
import copy
import multiprocessing as mp
import joblib
from tqdm import tqdm

FEATURE_THRESHOLD = 1e-7

params = {
        "maxiter":100, 
        "regf":0
    }


GENERATORS = {
    "da": [
        _generators_sklearn.OrthogonalSplitGenerator,
        _generators_da.DiagonalSplitGenerator,
        _generators_da.EllipseSplitGenerator
    ],
    "gfs": [
        _generators_sklearn.OrthogonalSplitGenerator,
        _generators_gfs.DiagonalSplitGenerator,
        _generators_gfs.EllipseSplitGenerator
    ]
}

class GeoTreeRegressor(BaseEstimator,RegressorMixin):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.

    Examples
    --------
    >>> from scikit_geodect import GeoTreeRegressor
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = GeoTreeRegressor()
    >>> estimator.fit(X, y)
    GeoTreeRegressor()
    """
    def __init__(self, max_depth=None, min_samples_split= None, max_features = None, n_jobs=None, random_state=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split if min_samples_split else 2
        self.n_jobs = n_jobs
        self.is_fitted_ = False
        self.tree_ = None
        self.generators_ = None
        self.features_ = None
        self.n_features_in_ = -1
        self.random_state = random_state
        #print("min samples: "+str(self.min_samples_split))
    def fit_while(self, i, r):
        a = np.array([(len(leaf.idx) > self.min_samples_split) for leaf in self.tree_.get_leafs()])
        b = np.array([np.all(leaf.y==leaf.y[0]) or np.all(leaf.X==leaf.X[0]) for leaf in self.tree_.get_leafs()])
        if r>0:
            return ((i <= r) and a.any()) and (not b.all())
        else:
            return a.any() and (not b.all())

    def grow_leaf(self, leaf, geo_features, random_state):

        if self.max_features:
            #print(self.features_, geo_features)
            fs = copy.copy(self.features_)
            for gf in geo_features[1:]:
                fs.remove(gf)
            #print(fs)
            fs = resample(fs, n_samples=self.max_features, replace=False, random_state=random_state)
            gfs = []
            if any(item in geo_features for item in fs):
                fs = fs + geo_features
                fs = list(set(fs))
                gfs = geo_features
            #print(fs, gfs)
        else:
            fs = self.features_
            gfs = geo_features
        
        gens = self.generators_

        
        leaf.grow(generators=gens,
                    features = fs,
                    geo_features=gfs,
                    n_jobs=self.n_jobs,
                    random_state=random_state
                    )

    def fit(self, X, y,
                generators = "da",
                gen_params = params,
                features = None, 
                geo_features = [],
                print_metrics= False,
                X_test = None,
                y_test= None):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        self.is_fitted_ = True

        trees = []
        metrics_r = {}
        #node_metrics = {}
        #n=1
        self.tree_ = _tree.Tree(X.astype('float32'),y.astype('float64'), self.max_depth, self.min_samples_split)
        X_test = X_test.astype('float32') if X_test is not None else None
        y_test = y_test.astype('float64') if y_test is not None else None

        self.generators_ = GENERATORS[generators] if type(generators)==str else generators
        if type(generators)==str:
            self.generators_ = [self.generators_[0], self.generators_[1](**gen_params), self.generators_[2](**gen_params)]
       
        self.features_ = features if features else list(range(X.shape[1]))
        self.max_features = None if not self.max_features else min(self.max_features, len(self.features_)) if self.max_features>1 else \
            max(1, int(self.max_features*len(self.features_)))


        self.n_features_in_ = len(features) if features else X.shape[1]

        start = time()

        random_state = check_random_state(self.random_state)

        #print(self.tree_.max_depth)

        i=1
        while self.tree_.get_leafs_to_grow():

            self.tree_.curr_depth += 1

            leafs_to_grow = self.tree_.get_leafs_to_grow()


            joblib.Parallel(n_jobs=self.n_jobs,require='sharedmem')(
                joblib.delayed(self.grow_leaf)(leaf,
                                                geo_features, 
                                                random_state 
                                                )
                for leaf in leafs_to_grow
            )

            

            

            end = time()

            # After each depth increase, calculate metrics
            
            metrics_r[i] = {}
            metrics_r[i] = _tree.calc_metrics(self.tree_.y, self.predict(self.tree_.X))
            metrics_r[i]['time'] = end-start
            metrics_r[i]['leaves'] = len(self.tree_.get_leafs_not_split())
            metrics_r[i]['true_leaves'] = self.tree_.n_leaves
            metrics_r[i]['elli_area'] = self.tree_.get_avg_elli_area(i)
            metrics_r[i]['ortho_ratio'],\
                metrics_r[i]['diag_ratio'],\
                    metrics_r[i]['elli_ratio'] = self.tree_.get_split_ratios()
            
            if X_test is not None:
                
                metrics_test = _tree.calc_metrics(y_test, self.predict(X_test), "test")
                metrics_r[i].update(metrics_test)
                
                if early_stopping:
                    
                    if (i>0) and (trees[-1][0] < metrics_test['maetest']):
                        self.tree_ = trees[-1][1]
                        break
                    else:
                        t = copy.deepcopy(self.tree_)
                        trees.append([metrics_test['maetest'],t])
            
            if print_metrics:
                print(str(i),self.tree_.n_leaves,metrics_r[i])
            self.tree_.set_metrics(metrics_r)
            i+=1
            

        
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, 'is_fitted_')

        yhat = np.zeros(len(X))

        nodes_to_expand = [(self.tree_.root, np.arange(0, len(X)))]

        while nodes_to_expand:
            expand_next = []
            for node_to_expand, idx in nodes_to_expand:
                if node_to_expand.is_not_split():
                    yhat[idx] = node_to_expand.yhat
                else:
                    idx_left, idx_right = node_to_expand.get_left_right_idx(X[idx], idx)
                    expand_next.append((node_to_expand.left, idx_left))
                    expand_next.append((node_to_expand.right, idx_right))
            nodes_to_expand = expand_next
        
        return yhat

    def predict_at_depth(self, X, n):
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, 'is_fitted_')

        yhat = np.zeros(len(X))

        nodes_to_expand = [(self.tree_.root, np.arange(0, len(X)))]

        while nodes_to_expand:
            expand_next = []
            for node_to_expand, idx in nodes_to_expand:
                if node_to_expand.is_leaf() or (node_to_expand.depth==n):
                    yhat[idx] = node_to_expand.yhat
                else:
                    idx_left, idx_right = node_to_expand.get_left_right_idx(X[idx], idx)
                    expand_next.append((node_to_expand.left, idx_left))
                    expand_next.append((node_to_expand.right, idx_right))
            nodes_to_expand = expand_next
        
        return yhat

    def get_depth(self):
        return self.tree_.curr_depth

    def get_n_leaves(self):
        return len(self.tree_.get_leafs())

    @property
    def feature_importances_(self, normalize=True):
        importances = np.zeros(self.n_features_in_+1, dtype=np.float64)
        for node in self.tree_.get_nodes():
            if node.split:
                if hasattr(node.split,'feat'):
                    importances[node.split.feat] += node.gain
                else:
                    importances[-1] += node.gain
        importances /= self.tree_.X.shape[0]

        if normalize:
            normalizer = np.sum(importances)
            if normalizer > 0.0:
                importances /= normalizer
        return importances

    def print_splits_depth_first(self, node):
        if node.split:
            print(' '*node.depth + str(node.split) + f', squared_error= {node.mse}, samples= {len(node.idx)}, value= {node.yhat}')
            self.print_splits_depth_first(node.left)
            self.print_splits_depth_first(node.right)
        else:
            print(' '*node.depth +f"Leaf: pred {node.yhat}")

    def plot_tree(self):
        self.print_splits_depth_first(self.tree_.root)




@joblib.wrap_non_picklable_objects
def fit_estimator(X, y, gens, gen_params, 
                bs_size, f_index, max_features, geo_features, 
                max_depth, min_samples_split, random_state):
    #print(random_state)
    
    bs_indices = _generate_sample_indices(
            random_state, X.shape[0], bs_size
        )

    X_bs = X[bs_indices]
    y_bs = y[bs_indices]
    
    #self.features_used_.append(fs)

    #X_bs = X_bs[:,fs]

    t = GeoTreeRegressor(max_depth=max_depth,  
                        min_samples_split=min_samples_split, 
                        max_features=max_features, 
                        random_state=random_state)

    t.fit(X= X_bs,
        y= y_bs,
        generators= gens,
        gen_params= gen_params,
        features= f_index,
        geo_features= geo_features)

    return t

class GeoRFRegressor(BaseEstimator,RegressorMixin):
    """ A Random Forest regressor based on geospatial regression trees.

    Parameters
    ----------
    n_estimators : int, default=5
        The number of regression trees to include in the ensemble.

    max_depth : int, default=2
        The maximum depth each tree can grow.

    max_samples : float, default=None
        The size of each bootstrap sample of the data expressed as a proportion of the data.
        If None, then the bootstrap sample size is the same as the size of the dataset. 

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """
    def __init__(self, n_estimators=5, max_depth=None, min_samples_split=None, max_samples=None, max_features=2, oob_score=False, n_jobs=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_samples = max_samples
        self.max_features = max_features
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.is_fitted_ = False
        self.n_features_ = None
        self.generators_ = None
        self.geo_features_ = []
        self.features_used_ = []
        self.estimators_ = []
        self.oob_score_ = None
        self.oob_prediction_ = None

    def collect_result(self, result):
        self.estimators_.append(copy.copy(result))


    def fit(self, X, y, gens="best", gen_params = params, geo_features=[]):
        """A reference implementation of a fitting function for a regressor.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : {array-like, sparse matrix}, shape (n_samples, )
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """

        X = check_array(X, accept_sparse=True)

        self.n_features_ = X.shape[1]
        self.generators_ = gens
        self.geo_features_ = geo_features
    
        # bootstrap
        self.max_samples = int(X.shape[0] * self.max_samples) if self.max_samples else X.shape[0]

        # sampling features
        f_index = list(range(self.n_features_))

        random_state = check_random_state(self.random_state)
        rs = random_state.randint(np.iinfo(np.int32).max, size=self.n_estimators)
        
        self.estimators_ = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(fit_estimator)(X,
                                                                                y,
                                                                                self.generators_,
                                                                                gen_params, 
                                                                                self.max_samples,
                                                                                f_index,
                                                                                self.max_features,
                                                                                self.geo_features_,
                                                                                self.max_depth,
                                                                                self.min_samples_split,
                                                                                random_state=r) for _,r in zip(tqdm(range(self.n_estimators)),rs))
        
        self.features_used_ = [t.features_ for t in self.estimators_]

        if self.oob_score:
            self._set_oob_score_and_prediction(X, y)

        self.is_fitted_ = True

        return self


    def predict(self, X):
        """ A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, 'is_fitted_')

        # Input validation
        X = check_array(X, accept_sparse=False)

        yhat = np.zeros((len(X),self.n_estimators))
        
        for i,t in enumerate(self.estimators_):
            yhat[:,i] = t.predict(X)
        

        return yhat.mean(axis=1)

    def predict_at_depth(self, X, n):
        # Check is fit had been called
        check_is_fitted(self, 'is_fitted_')

        # Input validation
        X = check_array(X, accept_sparse=False)

        yhat = np.zeros((len(X),self.n_estimators))
        
        for i,t in enumerate(self.estimators_):
            yhat[:,i] = t.predict_at_depth(X, n)
        

        return yhat.mean(axis=1)


    def _get_oob_predictions(self, est, X):
        y_pred = est.predict(X)
        y_pred = np.array(y_pred, copy=False)

        return y_pred

    def _compute_oob_predictions(self, X, y):
        n_samples = y.shape[0]
        oob_pred_shape = (n_samples,)
        oob_pred = np.zeros(shape = oob_pred_shape, dtype=np.float64)
        n_oob_pred = np.zeros((n_samples,), dtype=np.int64)

        for est in self.estimators_:
            unsampled_indices = _generate_unsampled_indices(random_state = est.random_state,
                                n_samples = n_samples, 
                                n_samples_bootstrap = self.max_samples
                                )
            
            y_pred = self._get_oob_predictions(est, X[unsampled_indices, :])
            oob_pred[unsampled_indices] += y_pred
            n_oob_pred[unsampled_indices] += 1
        if (n_oob_pred == 0).any():
            n_oob_pred[n_oob_pred == 0] = 1
            
        oob_pred /= n_oob_pred

        return oob_pred

    def _set_oob_score_and_prediction(self, X, y):
        self.oob_prediction_ = self._compute_oob_predictions(X, y)
        self.oob_score_ = r2_score(y, self.oob_prediction_)


    @property
    def feature_importances_(self):
        """
        From sklearn.ensemble._forest.py
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.
        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The values of this array sum to 1, unless all trees are single node
            trees consisting of only the root node, in which case it will be an
            array of zeros.
        """
        check_is_fitted(self)

        all_importances = joblib.Parallel(n_jobs=self.n_jobs, prefer="threads")(
            joblib.delayed(getattr)(tree, "feature_importances_")
            for tree in self.estimators_
            if tree.tree_.node_count > 1
        )

        if not all_importances:
            return np.zeros(self.n_features_in_, dtype=np.float64)

        all_importances = np.mean(all_importances, axis=0, dtype=np.float64)
        return all_importances / np.sum(all_importances)
    