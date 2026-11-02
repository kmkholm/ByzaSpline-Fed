"""B-spline geometry used by the server (numpy, float64): basis, Gram matrix, consensus knot grid,
re-interpolation (L2 projection) onto a new grid, Cholesky whitening, Krum-type scores, and the
Gaussian-mechanism privacy accountant.

Convention (identical to the client layer in models.py): degree k (k = 3: cubic), uniform interior grid of G
intervals on [lo, hi], extended by k knots on each side, so a knot vector has G + 2k + 1 entries and an edge
carries P = G + k coefficients. Basis functions use half-open intervals [t_j, t_{j+1}) and vanish outside
[t_0, t_end); the functional distance is the L2 distance over that support.
"""
import numpy as np


def uniform_knots(G=8, k=3, lo=-2.0, hi=2.0):
    h = (hi - lo) / G
    return lo + h * np.arange(-k, G + k + 1, dtype=np.float64)


def batched_basis(x, g, k):
    """x: (b, in), g: (in, L) knot vectors -> (b, in, L - k - 1). Works unchanged on numpy arrays and torch tensors
    (the client layer calls it with tensors)."""
    x = x[..., None]
    B = ((x >= g[:, :-1]) & (x < g[:, 1:])) * 1.0
    for d in range(1, k + 1):
        B = ((x - g[:, :-(d + 1)]) / (g[:, d:-1] - g[:, :-(d + 1)]) * B[..., :-1]
             + (g[:, d + 1:] - x) / (g[:, d + 1:] - g[:, 1:-d]) * B[..., 1:])
    return B


def basis(x, t, k):
    """Single knot vector t: x (n,) -> (n, len(t) - k - 1)."""
    x = np.asarray(x, np.float64)
    return batched_basis(x[:, None], np.asarray(t, np.float64)[None, :], k)[:, 0, :]


def _quad_points(t, n_per):
    gx, gw = np.polynomial.legendre.leggauss(n_per)
    xs, ws = [], []
    for a, b in zip(t[:-1], t[1:]):
        if b > a:
            xs.append(0.5 * (b - a) * gx + 0.5 * (a + b))
            ws.append(0.5 * (b - a) * gw)
    return np.concatenate(xs), np.concatenate(ws)


def gram(t, k):
    """G_pq = int B_p(u) B_q(u) du over the support of the knot vector. Gauss-Legendre with k + 1 nodes per knot
    interval integrates the degree-2k products exactly."""
    xs, ws = _quad_points(np.asarray(t, np.float64), k + 1)
    Bv = basis(xs, t, k)
    return (Bv * ws[:, None]).T @ Bv


def consensus_knots(T, min_gap_frac=1e-3):
    """T: (K, L) knot vectors of one input from K clients -> coordinate-wise median, made strictly increasing.
    The median of non-decreasing vectors is non-decreasing; ties (repeated knots) are separated by a minimum gap of
    min_gap_frac times the median knot spacing, so every basis function keeps a non-empty support."""
    t = np.median(np.asarray(T, np.float64), axis=0)
    gap = min_gap_frac * max(np.median(np.diff(t)), 1e-12)
    for j in range(1, len(t)):
        if t[j] < t[j - 1] + gap:
            t[j] = t[j - 1] + gap
    return t


def refit(C, t_old, t_new, k, n_per=8):
    """L2 projection of splines with coefficients C (..., P_old) on t_old onto the basis of t_new, over the support of
    t_new. Returns (coefficients (..., P_new), relative L2 re-interpolation error). Exact (error 0) when t_new = t_old."""
    xs, ws = _quad_points(np.asarray(t_new, np.float64), n_per)
    sw = np.sqrt(ws)[:, None]
    A_new = basis(xs, t_new, k) * sw
    Fv = (np.asarray(C, np.float64) @ basis(xs, t_old, k).T) * sw[:, 0]          # (..., n)
    flat = Fv.reshape(-1, Fv.shape[-1]).T                                        # (n, m)
    sol, *_ = np.linalg.lstsq(A_new, flat, rcond=None)                           # (P_new, m)
    resid = A_new @ sol - flat
    den = np.linalg.norm(flat)
    err = float(np.linalg.norm(resid) / den) if den > 0 else 0.0
    return sol.T.reshape(*Fv.shape[:-1], A_new.shape[1]), err


def refit_layer(C, grid_old, grid_new, k):
    """C: (out, in, P) spline coefficients of one KAN layer; grid_*: (in, L). Re-interpolates input by input."""
    out = np.empty((C.shape[0], C.shape[1], grid_new.shape[1] - k - 1))
    errs = []
    for i in range(C.shape[1]):
        if np.array_equal(grid_old[i], grid_new[i]):
            out[:, i] = C[:, i]
            continue
        out[:, i], e = refit(C[:, i], grid_old[i], grid_new[i], k)
        errs.append(e)
    return out, (float(np.mean(errs)) if errs else 0.0)


def layer_cholesky(grid, k, identity=False):
    """Per-input factors L_i with G_i = L_i L_i^T (G = I when identity=True). Shape (in, P, P)."""
    P = grid.shape[1] - k - 1
    if identity:
        return np.broadcast_to(np.eye(P), (grid.shape[0], P, P)).copy()
    return np.stack([np.linalg.cholesky(gram(g, k) + 1e-12 * np.eye(P)) for g in grid])


def whiten(C, Ls):
    """Row-vector coefficients c -> c L, so that ||(c_i - c_j) L||^2 = (c_i - c_j) G (c_i - c_j)^T: the Gram-weighted
    functional distance of Eq (2) is the Euclidean distance of whitened coefficients."""
    return np.einsum("oip,ipq->oiq", C, Ls)


def top_eig(Gm):
    """Largest eigenvalue / eigenvector of a Gram matrix; the eigenvector is signed to have a positive sum."""
    w, V = np.linalg.eigh(Gm)
    v = V[:, -1]
    return float(w[-1]), (v if v.sum() >= 0 else -v)


def sq_dists(X):
    """Pairwise squared Euclidean distances of the rows of X (K, d)."""
    n = (X * X).sum(1)
    D = n[:, None] + n[None, :] - 2.0 * X @ X.T
    np.fill_diagonal(D, 0.0)
    return np.maximum(D, 0.0)


def krum_scores(D, f):
    """Sum of squared distances to the K - f - 2 nearest other clients (at least 1)."""
    K = len(D)
    m = max(1, K - f - 2)
    s = np.empty(K)
    for i in range(K):
        s[i] = np.sort(np.delete(D[i], i))[:m].sum()
    return s


# ------------------------------------------------------------------ privacy accounting (Gaussian mechanism)
def gaussian_epsilon(z, rounds, delta=1e-5):
    """(eps, delta) after `rounds` adaptive compositions of a Gaussian mechanism with noise multiplier z
    (noise std / L2 sensitivity), via Renyi DP: eps = min_a [rounds * a / (2 z^2) + log(1/delta) / (a - 1)]."""
    a = np.concatenate([np.linspace(1.01, 10, 900), np.linspace(10, 5000, 5000)])
    return float(np.min(rounds * a / (2.0 * z * z) + np.log(1.0 / delta) / (a - 1.0)))


def noise_multiplier_for(eps, rounds, delta=1e-5):
    lo, hi = 1e-3, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if gaussian_epsilon(mid, rounds, delta) <= eps else (mid, hi)
    return hi
