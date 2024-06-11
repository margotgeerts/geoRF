import numpy as np
from ._splits import *
from sklearn.tree import DecisionTreeRegressor


class OrthogonalSplitGenerator:
    def generate_candidates(X, y, parent_mse, features, geo_features, n, bbox, random_state):
        dt = DecisionTreeRegressor(max_depth=1, random_state=random_state)
        dt.fit(X[:,features],y)
        f = features[dt.tree_.feature[0]] if dt.tree_.feature[0] >= 0 else np.nan
        split = dt.tree_.threshold[0] if dt.tree_.feature[0] >= 0 else np.nan
        orthosplit = OrthogonalSplit(f, split) if ~np.isnan(f) else None
        yield orthosplit
