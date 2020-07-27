"""

"""
import numpy as np
from odefilters import ode
from odefilters import odesolver 
from odefilters import statespace
import matplotlib.pyplot as plt
from matplotlib import animation
# Solve ODE
stsp = statespace.IBM(q=1, dim=2)
solver = odesolver.ODESolver(ssm=stsp, filtertype="kalman")
fhn = ode.FitzHughNagumo(t0=0., tmax=2., params=[0.2, 0.2, 3.0], initval=np.array([-1.0, 1.0]), initval_unc=1e-04)
t, m, s = solver.solve(fhn, stepsize=0.02)

# Extract relevant trajectories
mtraj1 = odesolver.get_trajectory(m, 0, 0)
munc1 = odesolver.get_trajectory(s, 0, 0)
mtraj2 = odesolver.get_trajectory(m, 1, 0)
munc2 = odesolver.get_trajectory(s, 1, 0)

# Plot solution
# plt.title("FitzHugh-Nagumo Model with Uncertainty")
# plt.plot(mtraj1, mtraj2, color="darkslategray", linewidth=1.5, label="Trajectory")
# plt.plot(mtraj1[0], mtraj2[0], 'o', label="Starting point at t=t0")
# plt.plot(mtraj1[-1], mtraj2[-1], 'o', label="Endpoint at t=tmax")
# plt.fill_between(mtraj1, mtraj2 + 3*munc2, mtraj2 - 3*munc2, color="darkslategray", alpha=0.125)
# plt.fill_betweenx(mtraj2, mtraj1 + 3*munc1, mtraj1 - 3*munc1, color="darkslategray", alpha=0.125)
# plt.legend()
# plt.show()


from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata, u1data, u2data = [], [], [], []
plt.title("FitzHugh-Nagumo Model")
ln, = plt.plot([], [], '-', color='darkslategray')

def init():
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 1.5)
    ln.set_data(xdata, ydata)
    return ln,

def update(frame):
    xdata.append(mtraj1[frame])
    ydata.append(mtraj2[frame])
    u1data.append(mtraj2[frame] - 6*munc2[frame])
    u2data.append(mtraj2[frame] + 6*munc2[frame])
    ln.set_data(xdata, ydata)
    # plt.remove()
    ax.remove()
    p = plt.fill_between(xdata, u1data, u2data, color='darkslategray', alpha=0.002)
    return ln, p

ani = FuncAnimation(fig, update, frames=len(mtraj1)-2,
                    init_func=init, interval=15, blit=True)
ani.save("./animations/FHN_uncertainty.mp4", fps=30, dpi=350) 