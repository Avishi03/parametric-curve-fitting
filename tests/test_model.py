"""The forward model and its inverse must be consistent."""

import numpy as np

from src.model import forward, unrotate, shape


def test_unrotate_inverts_forward():
    # For any params, un-rotating a forward-generated point recovers (u, v),
    # and u is exactly t.
    t = np.linspace(6, 60, 100)
    theta, M, X = np.deg2rad(30.0), 0.03, 55.0
    x, y = forward(t, theta, M, X)
    u, v = unrotate(x, y, theta, X)

    _, v_true = shape(t, M)
    assert np.allclose(u, t, atol=1e-9)
    assert np.allclose(v, v_true, atol=1e-9)


def test_shape_matches_definition():
    t = np.array([6.0, 20.0, 59.0])
    M = 0.03
    u, v = shape(t, M)
    assert np.allclose(u, t)
    assert np.allclose(v, np.exp(M * t) * np.sin(0.3 * t))
