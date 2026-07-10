"""The fitter must recover the parameters it was given."""

import numpy as np

from src.model import forward
from src.fit import residuals, fit, load_xy


def test_residual_zero_at_truth():
    # Data generated from known params has ~zero residual at those params.
    t = np.linspace(6, 60, 400)
    theta, M, X = np.deg2rad(30.0), 0.03, 55.0
    x, y = forward(t, theta, M, X)
    r = residuals([theta, M, X], x, y)
    assert np.max(np.abs(r)) < 1e-9


def test_fit_recovers_synthetic():
    # Round trip: make data from known params, fit, get them back.
    theta, M, X = np.deg2rad(22.0), -0.015, 40.0
    t = np.linspace(6, 60, 600)
    x, y = forward(t, theta, M, X)
    r = fit(x, y)
    assert abs(r["theta_deg"] - 22.0) < 1e-3
    assert abs(r["M"] - (-0.015)) < 1e-5
    assert abs(r["X"] - 40.0) < 1e-3


def test_fit_on_real_data():
    # The provided dataset must give the known clean answer, with the
    # recovered t-range inside the stated [6, 60] window.
    x, y = load_xy("data/xy_data.csv")
    r = fit(x, y)
    assert abs(r["theta_deg"] - 30.0) < 1e-2
    assert abs(r["M"] - 0.03) < 1e-4
    assert abs(r["X"] - 55.0) < 1e-2
    assert 6.0 <= r["t_min"] and r["t_max"] <= 60.0
