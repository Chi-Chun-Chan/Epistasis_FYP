""" An alternative way of scoring a parameter set, based on steady states and
the Jacobian alone. """

from typing import Dict, List, Tuple

import numpy as np
from numba import njit  # type: ignore

Vec3 = Tuple[float, float, float]


def acdc_score(par: Dict[str, float], S: float) -> float:
    """ Score for the coexistence of a DC and an AC state, as
    described in Section 6 of the notes.pdf. """
    equils = ss(par=par, S=S)

    # For now, we just score whether we have exactly three equilibria
    return 3.0 * np.abs(len(equils) - 3)


def dc_score(par: Dict[str, float], S: float) -> float:
    """ DC score, as described in Section 6 of the notes.pdf. """
    equils = ss(par=par, S=S)

    if len(equils) != 1:
        return 3.0 * np.abs(len(equils) - 1)

    j = jac(y=equils[0], par=par, S=S)
    lambdas = np.linalg.eigvals(j)  # type: ignore
    max_real = np.max(np.real(lambdas))
    return logistic(max_real)


def ac_score(par: Dict[str, float], S: float) -> float:
    """ AC score, as described in Section 6 of the notes.pdf. """

    # Find equilibria
    equils = ss(par=par, S=S)

    # If we don't have exactly one equilibrium, the score
    # is three times the absolute difference of the number
    # of equilibria from one
    if len(equils) != 1:
        return 3.0 * np.abs(len(equils) - 1)

    # Calculate the eigenvalues of the Jacobian at the equilibrium
    j = jac(y=equils[0], par=par, S=S)
    lambdas = np.linalg.eigvals(j)  # type: ignore

    if np.all(np.imag(lambdas) == 0):
        return 2.5

    # Else, we sort eigenvalues by imaginary part
    lambdas_sorted_ir = sorted([(np.imag(z), np.real(z)) for z in lambdas])

    # The complex eigenvalue with positive imag part is the last
    # It has the form lambda_+ = alpha + omega*i
    alpha = lambdas_sorted_ir[2][1]
    omega = lambdas_sorted_ir[2][0]

    # Score as defined
    return logistic(-alpha) + 2 * logistic(-omega)


@njit
def logistic(x: float) -> float:
    """ Logistic function. """
    return 1 / (1 + np.exp(-x))


def score(par: Dict[str, float]) -> float:
    """ Score a parametrisation, specified by a parameter dict,
    evaluated at four levels of external signal.
    We want pure DC behaviour at S=1 and S=10'000,
    pure AC behaviour at S=100,
    and a coexistence of both at S=1'000. """

    d_dc_1 = dc_score(par=par, S=1)
    d_ac = ac_score(par=par, S=100)
    d_acdc = acdc_score(par=par, S=1000)
    d_dc_2 = dc_score(par=par, S=10000)
    return d_dc_1 + d_ac + d_acdc + d_dc_2


def ss(par: Dict[str, float], S: float) -> List[Vec3]:
    """ Find all steady states of the system given a parametrisation
    and a signal concentration S."""

    def xbar_of_zbar(zbar):
        """ From eq (1) """
        c1 = par["alpha_X"] + par["beta_X"] * S
        c2 = 1 + S
        c3 = 1 / par["z_X"]
        return c1 / (c2 + (c3 * zbar)**par["n_ZX"])

    def ybar_of_xbar(xbar):
        """ From eq (2) """
        c4 = par["alpha_Y"] + par["beta_Y"] * S
        c5 = 1 + S
        c6 = 1 / par["x_Y"]
        return (1/par["delta_Y"]) * c4 / \
         (c5 + (c6*xbar)**par["n_XY"])

    def zbar_of_xbar_and_ybar(xbar, ybar):
        """ From eq (3) """
        return (1/par["delta_Z"]) / \
         (1 + (xbar/par["x_Z"])**par["n_XZ"] + (ybar/par["y_Z"])**par["n_YZ"])

    def f(zbar):
        """ Zero crossings of this function indicate equilibrium. """
        return -zbar + zbar_of_xbar_and_ybar(
            xbar=xbar_of_zbar(zbar=zbar),
            ybar=ybar_of_xbar(xbar=xbar_of_zbar(zbar)))

    zs = np.logspace(start=-8, stop=3, num=10000)
    pos = f(zs) > 0
    npos = ~pos
    idx = ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]

    zbars = zs[idx]
    xbars = [xbar_of_zbar(zbar) for zbar in zbars]  #xbar_of_zbar(zbars)
    ybars = [ybar_of_xbar(xbar) for xbar in xbars]  #ybar_of_xbar(xbars)
    return list(zip(xbars, ybars, zbars))


def jac(y: Vec3, par: Dict[str, float], S: float):
    """The Jacobian of the system. Works for both
    model topologies (2I and 1I)"""
    X, Y, Z = y

    c1 = par["alpha_X"] + par["beta_X"] * S
    c2 = 1 + S
    c3 = 1 / par["z_X"]
    n_ZX = par["n_ZX"]
    j13 = -n_ZX*c1*c3**n_ZX*Z**(n_ZX-1) / \
     (c2 + (c3*Z)**(n_ZX)) ** 2

    c4 = par["alpha_Y"] + par["beta_Y"] * S
    c5 = 1 + S
    c6 = 1 / par["x_Y"]
    n_XY = par["n_XY"]
    j21 = -n_XY*c4*c6**n_XY*X**(n_XY-1) / \
     (c5 + (c6*X)**n_XY) ** 2

    c7 = 1 + (Y / par["y_Z"])**par["n_YZ"]
    c8 = 1 / par["x_Z"]
    j31 = -par["n_XZ"]*(c8*X)**par["n_XZ"] / \
     (X*((c8*X)**par["n_XZ"] + c7)**2)

    c9 = 1 + (X / par["x_Z"])**par["n_XZ"]
    c10 = 1 / par["y_Z"]
    j32 = -par["n_YZ"]*(c10*Y)**par["n_YZ"] / \
     (Y*((c10*Y)**par["n_YZ"] + c9)**2)

    return np.array([[-1, 0, j13], [j21, -par["delta_Y"], 0],
                     [j31, j32, -par["delta_Z"]]])
