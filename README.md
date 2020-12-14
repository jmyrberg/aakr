# aakr

[![Build Status](https://travis-ci.com/jmyrberg/aakr.svg?branch=master)](https://travis-ci.com/jmyrberg/aakr)

Python implementation of the Auto-Associative Kernel Regression (AAKR). The algorithm is suitable for signal reconstruction, which further be used for e.g. condition monitoring or anomaly detection.


## Installation

`pip install aakr`


## Example usage

Give examples of normal conditions as pandas DataFrame or numpy array.

```python
from aakr import AAKR

aakr = AAKR()
aakr.fit(X_obs_nc)
```

Predict normal condition for given observations.

```python
X_nc = aakr.predict(X_obs)
```


## References

* [MTM algorithm by Martello and Toth](http://people.sc.fsu.edu/~jburkardt/f77_src/knapsack/knapsack.f) (Fortran)
* [A modified Auto Associative Kernel Regression method for robust signal reconstruction in nuclear power plant components](https://www.researchgate.net/publication/292538769_A_modified_Auto_Associative_Kernel_Regression_method_for_robust_signal_reconstruction_in_nuclear_power_plant_components)

---
Jesse Myrberg (jesse.myrberg@gmail.com)