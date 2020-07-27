"""

"""
import numpy as np
import scipy
import scipy.sparse as spsp
import scipy.sparse.linalg as spla
from odefilters import covariance as cov

import sys

def solve_precond(mat, arr):
    """
    """
    sol = np.linalg.solve(mat, arr)
    return sol



delta = 0.1
tsteps = np.arange(delta, 1+delta, delta)

covmat = cov.bmcov(tsteps, tsteps)
rhsvec = np.arange(len(tsteps))
sol = solve_precond(covmat, rhsvec)



# eigs, __ = np.linalg.eig(1000*covmat)
# print(eigs)





print("Stupid method:", np.linalg.norm(covmat.dot(sol) - rhsvec)/(np.sqrt(len(tsteps))))
print()