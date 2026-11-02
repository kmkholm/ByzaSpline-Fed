"""Byzantine attacks. Every malicious client first trains honestly on its own data (on flipped labels for 'label'),
which gives the attackers a *partial-knowledge* estimate of the honest update: they never see honest uploads.
They know the aggregation rule and the broadcast global model (including its knot grid), and they collude.

  label       data poisoning: y -> C - 1 - y (classification), RUL -> 125 - RUL
  sign        upload  global - 3 * own update
  gauss       upload  global + N(0, (3 * rms of own update)^2) per tensor
  alie        'a little is enough' (Baruch et al., 2019): mean - z * std of the attackers' updates, all parameters
  functional  functional spline poisoning, Eq (1): per edge e, d_e = sqrt(gamma_e / lambda_max) v_max with
              gamma_e = kappa * (estimated honest functional update energy of e); added to the attacker's own
              update re-interpolated on the global grid. Signs: last layer +1 into the target class and -1 elsewhere
              (classification; Normal is the target, i.e. faults are pushed towards 'healthy'), +1 for RUL
              (late prediction); hidden layers: a fixed random sign per edge shared by the colluders.
  functional_adaptive   the same direction, but each perturbation is shrunk towards the attacker's own honest head by
              the factor r / D, where D is the functional distance of the poisoned head to the colluders' mean honest
              head and r the median distance of their honest heads to that mean (an attacker that knows the rule and
              tries to stay inside the honest functional radius)
  encoder     encoder-only attack: spline head left honest; encoder parameters set to global - 3 * own encoder update
"""
import numpy as np
from scipy.stats import norm

import splines as S
from aggregators import layers_of

ATTACKS = ("none", "label", "sign", "gauss", "alie", "functional", "functional_adaptive", "encoder")


def _update(u, glob, keys):
    return {kk: u[kk] - glob[kk] for kk in keys}


def alie_z(K, f):
    s = int(np.floor(K / 2 + 1)) - f
    return float(norm.ppf((K - s) / K))


def functional_poison(benign, glob, k, kappa, task, target, signs, adaptive=False):
    layers = layers_of(glob)
    out = [dict(u) for u in benign]
    for L in layers:
        g = glob[f"{L}.grid"]
        Cg = glob[f"{L}.spline_weight"]
        # attackers' honest updates, re-interpolated onto the public global grid
        Ca = [S.refit_layer(u[f"{L}.spline_weight"], u[f"{L}.grid"], g, k)[0] for u in benign]
        dC = np.stack([c - Cg for c in Ca])                                    # (f, out, in, P)
        n_in = g.shape[0]
        Gs = [S.gram(g[i], k) for i in range(n_in)]
        eig = [S.top_eig(Gm) for Gm in Gs]
        lam = np.array([e[0] for e in eig])                                     # (in,)
        V = np.stack([e[1] for e in eig])                                       # (in, P)
        energy = np.stack([np.einsum("aop,pq,aoq->ao", dC[:, :, i], Gs[i], dC[:, :, i]) for i in range(n_in)], -1)
        gamma = kappa * energy.mean(0)                                          # (out, in)
        if L == layers[-1] and task == "cls":
            sg = -np.ones_like(gamma)
            sg[target] = 1.0
        elif L == layers[-1] and task == "rul":
            sg = np.ones_like(gamma)
        else:
            sg = signs[L]
        d = (np.sqrt(gamma / lam[None, :]) * sg)[..., None] * V[None, :, :]   # (out, in, P)
        for a, u in enumerate(out):
            u[f"{L}.grid"] = g.copy()
            u[f"{L}.spline_weight"] = Ca[a] + d
    if adaptive:
        # scale the perturbation of every attacker so that its whitened head distance to the attackers' mean honest
        # head equals the median distance of the attackers' own honest heads to that mean
        Ls = {L: S.layer_cholesky(glob[f"{L}.grid"], k) for L in layers}

        def wv(u):
            return np.concatenate([S.whiten(u[f"{L}.spline_weight"], Ls[L]).ravel() for L in layers])

        honest_heads = []
        for u in benign:
            h = {}
            for L in layers:
                h[f"{L}.spline_weight"] = S.refit_layer(u[f"{L}.spline_weight"], u[f"{L}.grid"], glob[f"{L}.grid"], k)[0]
            honest_heads.append(wv(h))
        H = np.stack(honest_heads)
        mu = H.mean(0)
        radius = np.median(np.linalg.norm(H - mu, axis=1)) if len(H) > 1 else np.linalg.norm(H[0] - wv(glob))
        for a, u in enumerate(out):
            base = {f"{L}.spline_weight": out[a][f"{L}.spline_weight"] for L in layers}
            cur = np.linalg.norm(wv(base) - mu)
            if cur > radius > 0:
                t = radius / cur
                for L in layers:
                    Ca_L = S.refit_layer(benign[a][f"{L}.spline_weight"], benign[a][f"{L}.grid"], glob[f"{L}.grid"], k)[0]
                    u[f"{L}.spline_weight"] = Ca_L + t * (u[f"{L}.spline_weight"] - Ca_L)
    return out


def apply(attack, benign, glob, ctx):
    """benign: list of the malicious clients' honestly trained states. ctx: dict(param_keys, enc_keys, K, f, k,
    kappa, task, target, signs, rng). Returns the uploaded malicious states."""
    if attack in ("none", "label"):
        return benign
    pk = ctx["param_keys"]
    ups = [_update(u, glob, pk) for u in benign]
    out = [dict(u) for u in benign]
    if attack == "sign":
        for o, d in zip(out, ups):
            for kk in pk:
                o[kk] = glob[kk] - 3.0 * d[kk]
    elif attack == "gauss":
        for o, d in zip(out, ups):
            for kk in pk:
                rms = np.sqrt(np.mean(d[kk] ** 2)) + 1e-12
                o[kk] = glob[kk] + ctx["rng"].normal(0.0, 3.0 * rms, size=d[kk].shape)
    elif attack == "alie":
        z = alie_z(ctx["K"], ctx["f"])
        for kk in pk:
            D = np.stack([d[kk] for d in ups])
            sd = D.std(0) if len(D) > 1 else np.abs(D[0])
            v = glob[kk] + D.mean(0) - z * sd
            for o in out:
                o[kk] = v
    elif attack in ("functional", "functional_adaptive"):
        out = functional_poison(benign, glob, ctx["k"], ctx["kappa"], ctx["task"], ctx["target"], ctx["signs"],
                                adaptive=attack == "functional_adaptive")
    elif attack == "encoder":
        for o, d in zip(out, ups):
            for kk in ctx["enc_keys"]:
                o[kk] = glob[kk] - 3.0 * d[kk]
    else:
        raise ValueError(attack)
    return out
