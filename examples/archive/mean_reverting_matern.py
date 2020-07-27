"""
mean_reverting_matern.py

We plot the trajectory of the mean of a 2d (nu=3/2)
Matern-type process with mean functions m1, m2, m3:
    m1(x) = (4, 3) (const.)
    m2(x) = x 
"""

import numpy as np
import matplotlib.pyplot as plt


x0 = np.array([1, 1])
goalpt = np.array([4, 3])
tvals = np.arange(2000 + 1)/200
print(tvals)
h = tvals[1] - tvals[0]
xvals = np.zeros((len(tvals), 2))
xvals[-1] = x0
for i in range(len(xvals)):
    x1, x2 = xvals[i-1]
    g1, g2 = 4, 2
    xvals[i][0] = x1 + h * (x2-g2)
    xvals[i][1] = x2 + h * (-3*(x1-g1) - 2*np.sqrt(3)*(x2-g2))# +np.sqrt(h)*np.random.randn()

# plt.style.use("bmh")
# plt.rc('font', size=30) # controls default text sizes
# plt.title("Mean of Matern Process")
# plt.plot(xvals[:, 0], xvals[:, 1], linewidth=6, label="Trajectory")
# plt.plot(xvals[-1, 0], xvals[-1, 1], 'o', label="End at t=%r" % tvals[-1])
# plt.plot(xvals[0, 0], xvals[0, 1], 'o', label="Start at t=%r" % tvals[0])
# plt.legend()
# plt.show()

plt.style.use("bmh")
plt.rc('font', size=30) # controls default text sizes
plt.plot(tvals, xvals[:, 0], linewidth=6, label="x0")
plt.plot(tvals, xvals[:, 1], linewidth=6, label="x1")
plt.legend()
plt.show()