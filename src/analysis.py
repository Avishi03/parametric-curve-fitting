"""How trustworthy is the recovered (theta, M, X)?

Three questions a point estimate alone can't answer:

  1. How precise is it?     -> analytic error bars from the fit Jacobian.
  2. Is it the only answer?  -> scan theta; show a single global minimum.
  3. Does it survive noise?  -> inject noise, refit many times, watch the spread.

Run from the repo root:  python -m src.analysis   (writes figures to assets/)
"""

import os

import numpy as np
from scipy.optimize import least_squares

from .fit import load_xy, fit, residuals, THETA_BOUNDS, M_BOUNDS, X_BOUNDS
from .model import forward, unrotate, OMEGA

OUT = "assets"
PARAM_NAMES = ("theta_deg", "M", "X")


def _refit_local(x, y, init):
    """One local least-squares from a good starting point (no multistart)."""
    lb = [THETA_BOUNDS[0], M_BOUNDS[0], X_BOUNDS[0]]
    ub = [THETA_BOUNDS[1], M_BOUNDS[1], X_BOUNDS[1]]
    return least_squares(residuals, init, args=(x, y), bounds=(lb, ub), method="trf")


def theta_from_pca(x, y):
    """Estimate theta from geometry alone, with no optimization.

    The curve runs along a 'spine' in the theta direction (u = t grows linearly
    while v only oscillates), so the dominant principal axis of the centred
    point cloud points along theta. Gives theta to a degree or two instantly.
    """
    P = np.c_[x - np.mean(x), y - np.mean(y)]
    _, V = np.linalg.eigh(P.T @ P)
    pc1 = V[:, -1]                      # eigenvector of the largest eigenvalue
    return float(np.degrees(np.arctan2(pc1[1], pc1[0])) % 180.0)


def l1_score(x, y, params):
    """Mean L1 distance between each data point and the predicted curve.

    Each point's recovered t is its un-rotated u; the predicted point is
    forward(u). This mirrors the assignment's L1 scoring metric.
    """
    theta, M, X = params["theta_rad"], params["M"], params["X"]
    u, _ = unrotate(x, y, theta, X)
    xp, yp = forward(u, theta, M, X)
    l1 = np.abs(xp - x) + np.abs(yp - y)
    return float(l1.mean()), float(l1.max())


def analytic_errors(res):
    """Standard errors from the least-squares covariance sigma^2 (J^T J)^-1.

    Returns 1-sigma errors in the SAME order as res.x: (theta_rad, M, X),
    plus theta's error converted to degrees.
    """
    J = res.jac
    m, n = J.shape
    dof = max(m - n, 1)
    sigma2 = float(np.sum(res.fun ** 2) / dof)      # residual variance
    cov = sigma2 * np.linalg.inv(J.T @ J)
    se = np.sqrt(np.diag(cov))                       # se[0]=theta_rad, [1]=M, [2]=X
    return {
        "theta_deg": float(np.rad2deg(se[0])),
        "M": float(se[1]),
        "X": float(se[2]),
        "sigma_resid": float(np.sqrt(sigma2)),
    }


def theta_profile(x, y, n=181):
    """Best achievable cost as a function of theta (M, X free at each theta).

    A single deep minimum over the whole theta range == the answer is unique.
    """
    thetas = np.linspace(*THETA_BOUNDS, n)
    costs = np.empty_like(thetas)
    for i, th in enumerate(thetas):
        # fix theta, let M and X adapt from a neutral start
        def resid_fixed(p):
            return residuals([th, p[0], p[1]], x, y)
        r = least_squares(
            resid_fixed, [0.0, 55.0],
            bounds=([M_BOUNDS[0], X_BOUNDS[0]], [M_BOUNDS[1], X_BOUNDS[1]]),
            method="trf",
        )
        costs[i] = float(np.sum(r.fun ** 2))
    return np.rad2deg(thetas), costs


def bootstrap(x, y, params, sigma_xy, n_boot=200, seed=0):
    """Refit on data perturbed by Gaussian noise of size sigma_xy in (x, y).

    Starts each refit from the known optimum, so it's fast and stays in the
    right basin. Returns an (n_boot, 3) array of (theta_deg, M, X).
    """
    rng = np.random.default_rng(seed)
    init = [params["theta_rad"], params["M"], params["X"]]
    out = np.empty((n_boot, 3))
    for b in range(n_boot):
        xb = x + rng.normal(0.0, sigma_xy, size=x.shape)
        yb = y + rng.normal(0.0, sigma_xy, size=y.shape)
        r = _refit_local(xb, yb, init)
        th, M, X = r.x
        out[b] = (np.rad2deg(th), M, X)
    return out


def robustness_sweep(x, y, params, noise_levels, n_boot=150):
    """Parameter scatter (std) at several noise levels. Returns dict per param."""
    stds = {k: [] for k in PARAM_NAMES}
    for s in noise_levels:
        samples = bootstrap(x, y, params, sigma_xy=s, n_boot=n_boot)
        for j, k in enumerate(PARAM_NAMES):
            stds[k].append(float(samples[:, j].std()))
    return stds


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT, exist_ok=True)
    x, y = load_xy("data/xy_data.csv")
    p = fit(x, y)
    res = _refit_local(x, y, [p["theta_rad"], p["M"], p["X"]])
    se = analytic_errors(res)

    print("Geometry sanity check (no optimization)")
    print("  theta from PCA principal axis = %.2f deg  (fitted: %.2f)"
          % (theta_from_pca(x, y), p["theta_deg"]))

    print("Recovered parameters with 1-sigma analytic error bars")
    print("  theta = %.4f  +/- %.2e  deg" % (p["theta_deg"], se["theta_deg"]))
    print("  M     = %.6f  +/- %.2e" % (p["M"], se["M"]))
    print("  X     = %.4f  +/- %.2e" % (p["X"], se["X"]))
    print("  residual sigma = %.3e  (data is essentially exact)" % se["sigma_resid"])

    l1_mean, l1_max = l1_score(x, y, p)
    print("L1 score (assignment metric)")
    print("  mean L1 = %.3e   max L1 = %.3e  (data vs predicted curve)"
          % (l1_mean, l1_max))

    # --- Identifiability figure: cost vs theta -------------------------------
    th_deg, costs = theta_profile(x, y)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(th_deg, costs, c="#1f77b4")
    ax.axvline(p["theta_deg"], c="#d62728", ls="--", lw=1,
               label="global min = %.2f deg" % p["theta_deg"])
    ax.set_xlabel(r"$\theta$ (deg)"); ax.set_ylabel("best fit cost (log scale)")
    ax.set_title("Identifiability: one sharp global minimum in " + r"$\theta$")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "identifiability.png"), dpi=150)
    plt.close(fig)

    # --- Robustness figure: parameter uncertainty vs injected noise ----------
    noise = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    stds = robustness_sweep(x, y, p, noise)
    print("\nRobustness to injected (x, y) noise  (1-sigma parameter scatter)")
    print("  noise |  d.theta(deg)   dM          dX")
    for i, s in enumerate(noise):
        print("  %5.2f | %10.4f  %10.6f  %8.4f"
              % (s, stds["theta_deg"][i], stds["M"][i], stds["X"][i]))

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for j, k in enumerate(PARAM_NAMES):
        ax[j].plot(noise, stds[k], "o-", c="#2ca02c")
        ax[j].set_xlabel("injected noise sigma (x,y units)")
        ax[j].set_ylabel("std of recovered " + k)
        ax[j].set_title(k)
    fig.suptitle("Robustness: recovered-parameter scatter grows smoothly with noise")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "robustness.png"), dpi=150)
    plt.close(fig)

    print("\nwrote identifiability.png and robustness.png to %s/" % OUT)


if __name__ == "__main__":
    main()
