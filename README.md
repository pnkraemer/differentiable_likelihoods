# Differentiable likelihoods for fast inversion of 'Likelihood-free' dynamical systems

This repository contains the python code that was used for the paper

Kersting, H., Kr\"amer, N., Schiegg, M., Daniel, C., Tiemann,M., and Hennig, P.   Differentiable likelihoods for fast inversion of ‘likelihood-free’ dynamical systems. In Proceedings of the 37th International Conference on Machine Learning, Vienna, Austria, PMLR 119, 2020.

and some related methods.


## Contents

Linear state space models, Kalman filter, probabilistic solvers for ODEs including different methods for generating measurements (see Kersting and Hennig, 2016), Markov Chain Monte Carlo simulation, Bayesian quadrature, Gaussian process regression, Clenshaw-Curtis quadrature and more.
<p align="center">
<img src="figures/hfn_unc.png" width="250px"><img src="figures/car_movement.png" width="250px"><img src="figures/res2bod.png" width="250px">
</p>

## Installation
In the root directory, run
```
pip install .
```
which allows avoiding the `sys.path.append` malarkey.

## Requirements

numpy, scipy, matplotlib

## Example
```python
from difflikelihoods import statespace
from difflikelihoods import odesolver
from difflikelihoods import ode
ibm = statespace.IBM(q=2, dim=1)
lin_ode = ode.LinearODE(t0=0.1, tmax=2.0, params=2.1, initval=0.9)
solver = odesolver.ODESolver(ibm, filtertype="kalman")
tsteps, means, stdevs = solver.solve(lin_ode, stepsize=0.01)
```
More examples are contained in the ```examples``` directory.

## Experiments

The experiments from the paper are in the ```experiments``` folder and sorted via `FigureN.ipynb`. 

## Cite as

Wherever relevant, please cite this work as
```
@article{kersting2020differentiable,
  title={Differentiable Likelihoods for Fast Inversion of'Likelihood-Free'Dynamical Systems},
  author={Kersting, Hans and Kr{\"a}mer, Nicholas and Schiegg, Martin and Daniel, Christian and Tiemann, Michael and Hennig, Philipp},
  journal={Proceedings of the 37th International Conference on Machine Learning, Vienna, Austria, PMLR 119},
  year={2020}
}

```
