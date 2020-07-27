# ODEFILTERS

This repository includes methods which allow to probabilistically solve ordinary differential equations as for instance presented by the papers below. This is a work in progress.


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

**Note:** Please do not use `python setup.py install`—for reasons.

## Requirements

dataclasses (Python 3.7), numpy, scipy, matplotlib, unittest

## Example
```python
from odefilters import statespace
from odefilters import odesolver
from odefilters import ode
ibm = statespace.IBM(q=2, dim=1)
lin_ode = ode.LinearODE(t0=0.1, tmax=2.0, params=2.1, initval=0.9)
solver = odesolver.ODESolver(ibm, filtertype="kalman")
tsteps, means, stdevs = solver.solve(lin_ode, stepsize=0.01)
```
More examples are contained in the ```examples``` directory.

## References

* Hairer, E., Norsett, S. P., Wanner, G. Solving Ordinary Differential Equations I: Nonstiff Problems. Springer, 2009.

* Kersting, H. and Hennig, P. Active Uncertainty Calibration in Bayesian ODE Solvers. Proceedings of the Thirty-Second Conference on Uncertainty in Artificial Intelligence, 2016

* Kersting, H., Sullivan, T., Hennig, P. Convergence Rates of Gaussian ODE Filters  https://arxiv.org/abs/1807.09737, 2018

* Rasmussen, C. & Williams, K., Gaussian Processes for Machine Learning, MIT Press, 2006

* Särkkä, S. Bayesian Filtering and Smoothing. Cambridge University Press, 2013.

* Särkkä, S., Solin, A. Applied Stochastic Differential Equations, Cambridge University Press, 2019

* Schober, M., Duvenaud, D. and Hennig, P. Probabilistic ODE Solvers with Runge-Kutta Means. https://arxiv.org/pdf/1406.2582.pdf

* Schober, M., Särkkä, S. and Hennig, P. A Probabilistic Model for the Numerical Solution of Initial Value Problems. Statistics and Computing, 2019.

* Tronarp, F., Kersting, H., Särkkä, S. and Hennig, P. Probabilistic Solutions to Ordinary Differential Equations as Non-Linear Bayesian FIltering: A New Perspective. arXiv:1810.03440, 2018.