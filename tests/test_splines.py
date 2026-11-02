"""numpy-only checks of the server-side spline geometry (no torch):  python tests/test_splines.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "byzaspline_fed"))
import splines as S  # noqa: E402


def cox_de_boor(x, t, j, k):
    if k == 0:
        return 1.0 if t[j] <= x < t[j + 1] else 0.0
    a = 0.0 if t[j + k] == t[j] else (x - t[j]) / (t[j + k] - t[j]) * cox_de_boor(x, t, j, k - 1)
    b = 0.0 if t[j + k + 1] == t[j + 1] else (t[j + k + 1] - x) / (t[j + k + 1] - t[j + 1]) * cox_de_boor(x, t, j + 1, k - 1)
    return a + b


def main():
    rng = np.random.default_rng(0)
    G, k = 8, 3
    t = S.uniform_knots(G, k)
    P = G + k
    assert len(t) == G + 2 * k + 1
    # 1) vectorised basis == recursive Cox-de Boor, on a uniform and a random non-uniform grid
    tn = np.sort(np.r_[t[:k + 1], rng.uniform(-2, 2, G - 1), t[-(k + 1):]])
    for knots in (t, tn):
        xs = rng.uniform(knots[0], knots[-1], 300)
        Bv = S.basis(xs, knots, k)
        ref = np.array([[cox_de_boor(x, knots, j, k) for j in range(P)] for x in xs])
        assert np.abs(Bv - ref).max() < 1e-12, "basis mismatch"
    # 2) partition of unity on the interior [t_k, t_{G+k})
    xi = np.linspace(t[k], t[G + k] - 1e-9, 500)
    assert np.abs(S.basis(xi, t, k).sum(1) - 1).max() < 1e-12
    # 3) Gram: symmetric positive definite and equal to dense numerical integration
    Gm = S.gram(t, k)
    assert np.allclose(Gm, Gm.T) and np.linalg.eigvalsh(Gm).min() > 0
    xd = np.linspace(t[0], t[-1], 400001)
    Bd = S.basis(xd[:-1], t, k)
    G_num = (Bd.T @ Bd) * (xd[1] - xd[0])
    assert np.abs(Gm - G_num).max() < 1e-4, np.abs(Gm - G_num).max()
    # 4) functional distance: c G c = int (sum c_p B_p)^2
    c = rng.normal(size=P)
    f2 = ((Bd @ c) ** 2).sum() * (xd[1] - xd[0])
    assert abs(c @ Gm @ c - f2) / f2 < 1e-4
    # 5) whitening reproduces the Gram-weighted distance
    Ls = S.layer_cholesky(np.stack([t, tn]), k)
    C1, C2 = rng.normal(size=(5, 2, P)), rng.normal(size=(5, 2, P))
    d_direct = sum((C1[o, i] - C2[o, i]) @ S.gram([t, tn][i], k) @ (C1[o, i] - C2[o, i]) for o in range(5) for i in range(2))
    d_white = ((S.whiten(C1, Ls) - S.whiten(C2, Ls)) ** 2).sum()
    assert abs(d_direct - d_white) / d_direct < 1e-9
    # 6) re-interpolation: identity on the same grid; small error for a smooth function on a shifted grid
    Cr, e = S.refit(C1[:, 0], t, t, k)
    assert e < 1e-12 and np.abs(Cr - C1[:, 0]).max() < 1e-9
    t_shift = t + 0.05
    smooth = np.linalg.lstsq(S.basis(xi, t, k), np.sin(xi), rcond=None)[0]
    _, e2 = S.refit(smooth[None], t, t_shift, k)
    assert e2 < 0.05, e2
    # 7) consensus knots: strictly increasing even with repeated knots
    T = np.stack([t, t, tn, tn, tn])
    T[:, 5] = T[:, 6]
    tc = S.consensus_knots(T)
    assert np.all(np.diff(tc) > 0)
    # 8) attack direction: the top eigenvector maximises the Rayleigh quotient; Eq (1) is solved by
    #    d = sqrt(gamma / lambda_max) v_max with |d|^2 = gamma / lambda_max
    lam, v = S.top_eig(Gm)
    for _ in range(200):
        u = rng.normal(size=P)
        assert u @ Gm @ u / (u @ u) <= lam + 1e-12
    gamma = 0.7
    d = np.sqrt(gamma / lam) * v
    assert abs(d @ Gm @ d - gamma) < 1e-12
    cond = lam / np.linalg.eigvalsh(Gm).min()
    # 9) Krum scores pick the central point
    X = np.r_[rng.normal(size=(9, 4)), rng.normal(size=(3, 4)) + 8]
    s = S.krum_scores(S.sq_dists(X), 3)
    assert set(np.argsort(s)[:9]) == set(range(9))
    # 10) privacy accountant: monotone in z; noise multiplier for eps=8.5 reproduces eps
    assert S.gaussian_epsilon(2, 100) > S.gaussian_epsilon(4, 100)
    z = S.noise_multiplier_for(8.5, 100)
    assert abs(S.gaussian_epsilon(z, 100) - 8.5) < 1e-3
    print(f"all spline tests passed | cubic G=8 Gram: lambda_max={lam:.4f}, cond={cond:.1f}, "
          f"mean eig={np.trace(Gm) / P:.4f} | z(eps=8.5, R=100)={z:.3f}")


if __name__ == "__main__":
    main()
