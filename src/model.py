"""The curve and its inverse.

    x = t*cos(th) - e^(M|t|)*sin(0.3t)*sin(th) + X
    y = 42 + t*sin(th) + e^(M|t|)*sin(0.3t)*cos(th)

It's just the curve (u, v) = (t, e^(M|t|)*sin(0.3t)) rotated by th about (X, 42).
That rotation is the whole trick behind the fit (docs/derivation.md).
"""

import numpy as np

Y_CENTER = 42.0
OMEGA = 0.3


def shape(t, M):
    """The pre-rotation curve: u = t, v = e^(M|t|) sin(0.3 t)."""
    t = np.asarray(t, float)
    return t, np.exp(M * np.abs(t)) * np.sin(OMEGA * t)


def forward(t, theta, M, X):
    """Point(s) (x, y) on the curve at parameter t."""
    u, v = shape(t, M)
    x = u * np.cos(theta) - v * np.sin(theta) + X
    y = Y_CENTER + u * np.sin(theta) + v * np.cos(theta)
    return x, y


def unrotate(x, y, theta, X):
    """Rotate (x, y) back to (u, v). With the right theta/X, u is exactly t."""
    dx, dy = np.asarray(x, float) - X, np.asarray(y, float) - Y_CENTER
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    return u, v
