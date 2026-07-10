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


LB = [THETA_BOUNDS[0], M_BOUNDS[0], X_BOUNDS[0]]
UB = [THETA_BOUNDS[1], M_BOUNDS[1], X_BOUNDS[1]]


def _theta_cost(x, y, theta, x_seeds, m0=0.0):
    """Cheapest cost achievable at a fixed theta (M, X free), plus that (M, X)."""
    best_c, best_p = np.inf, None
    for x0 in x_seeds:
        r = least_squares(
            lambda p: residuals([theta, p[0], p[1]], x, y), [m0, x0],
            bounds=([M_BOUNDS[0], X_BOUNDS[0]], [M_BOUNDS[1], X_BOUNDS[1]]),
            method="trf",
        )
        c = float(np.sum(r.fun ** 2))
        if c < best_c:
            best_c, best_p = c, r.x
    return best_c, best_p


def fit(x, y, n_theta=51, n_x=5, n_refine=3, m_seeds=(-0.02, 0.0, 0.02)):
    """Coarse-to-fine. Scan theta (M, X free) to find the promising basins,
    then run full 3-parameter refinements only from the best few."""
    x_seeds = np.linspace(X_BOUNDS[0] + 5, X_BOUNDS[1] - 5, n_x)

    # Coarse: rank every theta by its best achievable cost.
    thetas = np.linspace(*THETA_BOUNDS, n_theta)
    scored = [(theta, *_theta_cost(x, y, theta, x_seeds)) for theta in thetas]
    scored.sort(key=lambda s: s[1])

    # Fine: full 3-parameter fits seeded from the top theta basins.
    best, best_cost = None, np.inf
    for theta0, _, (m_seed, x0) in scored[:n_refine]:
        for m0 in (m_seed, *m_seeds):
            res = least_squares(
                residuals, [theta0, m0, x0], args=(x, y),
                bounds=(LB, UB), method="trf",
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
