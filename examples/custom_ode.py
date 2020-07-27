"""
custom_ode.py

We demonstrate two ways of creating custom ODE objects:
    * RHS as objects (advantage: change parameters as you like)
    * RHS as functions (advantage: simple)

Both are illustrated via creating the differential equations
for the sigmoid logistic growth function.

"""

from odefilters import ode


class SigmoidLogGrowthRHS():
    """
    RHS of Sigmoid logistic growth function ODE
    as an object with parameters as instance attributes

    It takes more lines of code than the function version below
    but is much more flexible with respect to changing parameters:
    see main below.
    """
    def __init__(self, r, k):
        """
        Initialise SLG with r and k
        """
        self.r = r
        self.k = k

    def __repr__(self):
        """
        Represent SLG object and its parameters.
        """
        return "SigmoidLogGrowthRHS(r=%r, k=%r)" % (self.r, self.k)

    def modeval(self, t, x):
        """
        Evaluate RHS.
        """
        return self.r * x * (1. - x / self.k)


def slg_rhs(t, x, r=2., k=2.):
    """
    RHS evaluation of sigmoid logistic growth as
    a plain function.

    It takes much fewer lines of code than the object version
    but is less flexible with respect to changing parameters.
    """
    return r * x * (1. - x / k)


if __name__ == "__main__":

    print("\n\n-------------------------------------------")
    print("Custom ODE via RHS as function:")
    print("-------------------------------------------")
    custom_ode_fct = ode.CustomODE(t0=0.0, tmax=1.0, modeval=slg_rhs,
                                   initval=[-1., 1.], initval_unc=0.0)
    print(custom_ode_fct)

    print("\n\n-------------------------------------------")
    print("Custom ODE via RHS as object:")
    print("-------------------------------------------")
    slgrhs = SigmoidLogGrowthRHS(r=1., k=2.)
    custom_ode_obj = ode.CustomODE(t0=0.0, tmax=1.0, modeval=slgrhs.modeval,
                                   initval=[-1., 1.], initval_unc=0.0)
    print(custom_ode_obj)

    slgrhs.r = 200.
    print("\n\n-------------------------------------------")
    print("Custom ODE via RHS as object,")
    print("after changing r to r=%u"% slgrhs.r)
    print("and without touching the ODE object itself:")
    print("-------------------------------------------")
    print(custom_ode_obj, "\n")

# END OF FILE
