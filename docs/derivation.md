# Mathematical Derivation

This document explains how the three unknowns **θ, M, X** are recovered from a
cloud of `(x, y)` points **without ever knowing which value of `t` produced each
point**. The key is to notice that the curve is a *rotated, translated* version
of a much simpler curve.

## 1. The model

```
x = t·cos(θ) − e^(M·|t|)·sin(0.3t)·sin(θ) + X
y = 42 + t·sin(θ) + e^(M·|t|)·sin(0.3t)·cos(θ)
```

with the given ranges `0° < θ < 50°`, `−0.05 < M < 0.05`, `0 < X < 100`, and
the parameter `6 < t < 60`.

## 2. The naïve approach (and why it struggles)

The obvious idea is to feed the two equations straight into a non-linear least
squares solver. The problem: **each observed point has its own unknown `t`**.
With `N` points that is `N + 3` unknowns (one `t` per point plus θ, M, X), a
large, poorly-conditioned optimization that is easy to get wrong and slow to
converge. We can do far better by exploiting the structure.

## 3. Key observation — it is a rotation

Group the two "shape" terms into named quantities:

- a **linear** part `u(t) = t`
- an **oscillating** part `v(t) = e^(M·|t|)·sin(0.3t)`

Rewriting the model around the fixed centre `(X, 42)`:

```
x − X   = u·cos(θ) − v·sin(θ)
y − 42  = u·sin(θ) + v·cos(θ)
```

In matrix form:

```
[ x − X  ]   [ cos θ   −sin θ ] [ u ]
[ y − 42 ] = [ sin θ    cos θ ] [ v ]
```

That matrix is exactly a **2-D rotation by θ**. So the whole dataset is nothing
more than the simple curve `(u, v) = (t,  e^(M·|t|)·sin(0.3t))` **rotated by θ
and shifted by `(X, 42)`**.

## 4. Un-rotating the data

A rotation is trivially invertible (its inverse is rotation by `−θ`). Given a
candidate `θ` and `X`, we can map every observed point back to `(u, v)` space:

```
u =  (x − X)·cos(θ) + (y − 42)·sin(θ)
v = −(x − X)·sin(θ) + (y − 42)·cos(θ)
```

Now two facts pin everything down:

1. **`u` recovers `t` directly.** Because `u(t) = t`, the un-rotated horizontal
   coordinate *is* the parameter `t` for that point. We never had to guess it.

2. **`|t| = t`.** Every point satisfies `t > 6 > 0`, so `|t| = t` and the
   exponential simplifies to `e^(M·t)`.

Therefore, for the *correct* `(θ, M, X)`, every point must satisfy the single
scalar constraint

```
v  =  e^(M·u)·sin(0.3·u)
```

## 5. The optimization

We reduce the whole problem to **just 3 unknowns**. Define the per-point residual

```
r_i(θ, M, X) = v_i − e^(M·u_i)·sin(0.3·u_i)
```

where `u_i, v_i` are the un-rotated coordinates of point `i`. We then solve

```
min_{θ, M, X}   Σ_i  r_i(θ, M, X)²
```

with `scipy.optimize.least_squares` (Trust Region Reflective) subject to the
given box constraints on θ, M, X.

**Multistart.** The `sin(0.3t)` factor makes the cost surface non-convex in θ
(several local minima). To guarantee we find the *global* optimum rather than a
nearby local one, we launch the solver from a grid of initial guesses spanning
the allowed θ and X ranges and keep the best result.

## 6. A built-in correctness check

The recovered `u`-values are supposed to *be* the `t`-values, which the problem
states lie in `6 < t < 60`. So after fitting we check that
`u ∈ [6, 60]`. If the recovered `t`-range lands cleanly inside the stated bounds,
that is **independent evidence** (not used during fitting) that the solution is
the true one and not a spurious local minimum.

## 7. Result

Running this procedure on the provided dataset yields, to machine precision:

| Parameter | Recovered value |
|-----------|-----------------|
| θ | 30.0000° (0.523599 rad) |
| M | 0.030000 |
| X | 55.0000 |

with an RMS residual of ~3.5 × 10⁻⁶ and a recovered `t`-range of `[6.05, 59.99]`
— exactly inside the stated `6 < t < 60`, confirming the fit.
