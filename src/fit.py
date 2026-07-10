"""Recover (theta, M, X) from the point cloud.

Un-rotate the data for a trial (theta, X); then u is t and, since t > 0, the
model collapses to v = e^(M u) sin(0.3 u). That leaves one residual and three
unknowns. sin(0.3 t) makes the surface bumpy in theta, so we try many starts
and keep the best.
"""

import numpy as np
from scipy.optimize import least_squares

from .model import unrotate, OMEGA

# Ranges given in the problem.
THETA_BOUNDS = (np.deg2rad(0.0), np.deg2rad(50.0))
M_BOUNDS = (-0.05, 0.05)
X_BOUNDS = (0.0, 100.0)


def load_xy(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def residuals(params, x, y):
    theta, M, X = params
    u, v = unrotate(x, y, theta, X)
    return v - np.exp(M * np.abs(u)) * np.sin(OMEGA * u)


def fit(x, y, n_theta=25, n_x=19, m_seeds=(-0.02, 0.0, 0.02)):
    lb = [THETA_BOUNDS[0], M_BOUNDS[0], X_BOUNDS[0]]
    ub = [THETA_BOUNDS[1], M_BOUNDS[1], X_BOUNDS[1]]

    best, best_cost = None, np.inf
    for theta0 in np.linspace(*THETA_BOUNDS, n_theta):
        for x0 in np.linspace(X_BOUNDS[0] + 1, X_BOUNDS[1] - 1, n_x):
            for m0 in m_seeds:
                res = least_squares(
                    residuals, [theta0, m0, x0], args=(x, y),
                    bounds=(lb, ub), method="trf",
                )
                cost = float(np.sum(res.fun ** 2))
                if cost < best_cost:
                    best_cost, best = cost, res

    theta, M, X = best.x
    u, _ = unrotate(x, y, theta, X)
    return {
        "theta_rad": float(theta),
        "theta_deg": float(np.rad2deg(theta)),
        "M": float(M),
        "X": float(X),
        "rms_residual": float(np.sqrt(np.mean(best.fun ** 2))),
        "cost": best_cost,
        "t_min": float(u.min()),   # should sit inside [6, 60]
        "t_max": float(u.max()),
    }
