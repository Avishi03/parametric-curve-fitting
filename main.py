"""Recover (theta, M, X) from data/xy_data.csv and print the result.

    python main.py                # fit and print
    python main.py --plots        # also (re)generate the plots in assets/
    python main.py --data foo.csv # use a different point cloud
"""

import argparse

from src.fit import load_xy, fit


def main():
    ap = argparse.ArgumentParser(description="Recover theta, M, X from a point cloud.")
    ap.add_argument("--data", default="data/xy_data.csv", help="CSV of x,y points")
    ap.add_argument("--plots", action="store_true", help="also write plots to assets/")
    args = ap.parse_args()

    x, y = load_xy(args.data)
    r = fit(x, y)

    print("Recovered parameters")
    print("  theta = %.4f deg  (%.6f rad)" % (r["theta_deg"], r["theta_rad"]))
    print("  M     = %.6f" % r["M"])
    print("  X     = %.4f" % r["X"])
    print("Fit quality")
    print("  RMS residual = %.3e" % r["rms_residual"])
    print("  recovered t  = [%.2f, %.2f]  (stated: 6 < t < 60)"
          % (r["t_min"], r["t_max"]))

    if args.plots:
        from scripts.make_plots import main as make_plots
        make_plots()


if __name__ == "__main__":
    main()
