## Content

This repository contains the experiments that reproduce the figures in the paper "Likelihood for 'Likelihood-Free' Dynamical Systems" (submission ID 
1580); see e.g. Figure4bottom_code for the code that produces the bottom plot 
of Figure 4. The code uses Gaussian ODE Filtering as a forward solver. The 
code for Gaussian ODE Filtering is to be found in the sub-repo odesolvers. 
Compared to previous ODE Filtering publications, the heart of the new
methods is the computation of the Jacobian estimator J. This computation
is contained in the file odesolvers/linearisation.py.

## Important Remarks

* The figures were edited afterwards to make them look nice for the paper.
* For Figure 4 bottom, random walk Monte Carlo (RWM) does not even find likelihood values of at least 10^{-300}. Hence, we added a line at 10^{-300} for RWM in the paper (see caption of Figure 4).

## Requirements

dataclasses (Python 3.7), numpy, scipy, matplotlib, unittest

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