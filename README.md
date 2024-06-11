#  geoRF: a Geospatial Random Forest :deciduous_tree:

GeoRF is the Python-based code repository for the paper "geoRF: a Geospatial Random Forest". It is a bagged tree ensemble for geospatial data that introduces two split types specifically designed for geospatial features: a diagonal split and a Gaussian split. More information can be found in the paper: [link to be added] ().


## Installation

Clone the repository to use the geoRF model.

```bash
git clone https://github.com/margotgeerts/geoRF.git
```

## Usage

```python
from geoRF import GeoRFRegressor

# instantiate GeoRFRegressor
regr = GeoRFRegressor(n_estimators=100, n_jobs=-1)

# train the model, give training data and the column indices of the geospatial features
regr.fit(X_train, y_train, geo_features=[0,1])


```

Also see the [example](Example.ipynb).

## Cite us 

```bibtex
@article{geerts2024georf,
title={geoRF: a Geospatial Random Forest},
author={Geerts, Margot and {vanden Broucke}, Seppe and {De Weerdt}, Jochen},
  journal={},
  volume={},
  pages={},
  year={},
  publisher={}
```

## Contributions

This is an experimental implementation. If you find any errors, please let me know.

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.
