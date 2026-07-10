"""Generate the proof-of-fit plots in assets/.

Runs the fitter on data/xy_data.csv and saves three figures:
  fit_overlay.png  - data cloud with the recovered curve on top
  residuals.png    - per-point residual vs recovered t (should be ~1e-6)
  recovered_t.png  - recovered t values, expected to sit inside [6, 60]

Run from the repo root:  python scripts/make_plots.py
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # no display needed, just write files
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fit import load_xy, fit
from src.model import forward, unrotate, OMEGA

DATA = "data/xy_data.csv"
OUT = "assets"


def main():
    os.makedirs(OUT, exist_ok=True)
    x, y = load_xy(DATA)
    r = fit(x, y)
    theta, M, X = r["theta_rad"], r["M"], r["X"]
    print("theta = %.4f deg   M = %.5f   X = %.4f" % (r["theta_deg"], M, X))
    print("RMS residual = %.2e   t range = [%.2f, %.2f]"
          % (r["rms_residual"], r["t_min"], r["t_max"]))

    # 1. Fit overlay: data points + the recovered curve on a dense t grid.
    t_grid = np.linspace(r["t_min"], r["t_max"], 2000)
    xc, yc = forward(t_grid, theta, M, X)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=6, c="#1f77b4", alpha=0.5, label="data (1500 pts)")
    ax.plot(xc, yc, c="#d62728", lw=1.5, label="recovered curve")
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(r"Fit overlay  ($\theta$=%.2f$\degree$, M=%.4f, X=%.2f)"
                 % (r["theta_deg"], M, X))
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fit_overlay.png"), dpi=150)
    plt.close(fig)

    # 2. Residuals vs recovered t.
    u, v = unrotate(x, y, theta, X)
    resid = v - np.exp(M * u) * np.sin(OMEGA * u)
    order = np.argsort(u)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(u[order], resid[order], c="#2ca02c", lw=0.8)
    ax.axhline(0, c="k", lw=0.5)
    ax.set_xlabel("recovered t"); ax.set_ylabel("residual")
    ax.set_title("Residuals (RMS = %.2e)" % r["rms_residual"])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "residuals.png"), dpi=150)
    plt.close(fig)

    # 3. Recovered t values against the stated [6, 60] window.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(u, bins=60, color="#9467bd", alpha=0.8)
    ax.axvline(6, c="r", ls="--", lw=1, label="stated bounds 6..60")
    ax.axvline(60, c="r", ls="--", lw=1)
    ax.set_xlabel("recovered t"); ax.set_ylabel("count")
    ax.set_title("Recovered t range: [%.2f, %.2f]" % (r["t_min"], r["t_max"]))
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "recovered_t.png"), dpi=150)
    plt.close(fig)

    print("wrote 3 plots to %s/" % OUT)


if __name__ == "__main__":
    main()
