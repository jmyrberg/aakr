# aakr (Auto Associative Kernel Regression)

[![Build Status](https://travis-ci.com/jmyrberg/aakr.svg?branch=master)](https://travis-ci.com/jmyrberg/aakr) [![Documentation Status](https://readthedocs.org/projects/aakr/badge/?version=latest)](https://aakr.readthedocs.io/en/latest/?badge=latest)

![aakr cover](https://github.com/jmyrberg/aakr/blob/master/docs/cover.jpg?raw=true)


**aakr** is a Python implementation of the Auto-Associative Kernel Regression (AAKR). The algorithm is suitable for signal reconstruction, which can further be used for condition monitoring, anomaly detection etc.

Documentation is available at https://aakr.readthedocs.io.


## Installation

`pip install aakr`


## Quickstart

Given historical normal condition `X_nc` examples and new observations `X_obs` of size `n_samples x n_features`, what values we expect to see in normal conditions for the new observations? 

```python
from aakr import AAKR

# Create AAKR model
aakr = AAKR()

# Fit the model with normal condition examples
aakr.fit(X_nc)

# Ask for values expected to be seen in normal conditions
X_obs_nc = aakr.transform(X_obs)
```


## References

* [A modified Auto Associative Kernel Regression method for robust signal reconstruction in nuclear power plant components](https://www.researchgate.net/publication/292538769_A_modified_Auto_Associative_Kernel_Regression_method_for_robust_signal_reconstruction_in_nuclear_power_plant_components)

---
Jesse Myrberg (jesse.myrberg@gmail.com)