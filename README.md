# Parametric Curve Fitting — Recovering θ, M, X

> Flam AI / R&D Assignment — estimating the unknown parameters of a rotated,
> exponentially-modulated spiral from a cloud of `(x, y)` points.

## Problem

Given a list of points that lie on the curve below (for `6 < t < 60`), recover
the three unknown parameters **θ, M, X**.

```
x = t·cos(θ) − e^(M·|t|)·sin(0.3t)·sin(θ) + X
y = 42 + t·sin(θ) + e^(M·|t|)·sin(0.3t)·cos(θ)
```

**Search ranges**

| Param | Range |
|-------|-------|
| θ | 0° < θ < 50° |
| M | −0.05 < M < 0.05 |
| X | 0 < X < 100 |
| t | 6 < t < 60 |

## Results

_To be filled in as the solution is built (see commit history)._

## Repository layout

```
data/            input points (xy_data.csv)
src/             model, fitting, and validation code
docs/            mathematical derivation
assets/          plots (fit overlay, residuals, recovered-t)
tests/           sanity tests
main.py          CLI entry point
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash);  use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

## Usage

_To be documented._

---

_Work in progress — this README evolves alongside the commit history._
