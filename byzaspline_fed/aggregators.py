"""Server-side aggregation rules. Uploads are full client states (numpy float64 dicts, same keys).

Off-the-shelf rules act on the raw uploaded vectors (knot vectors and coefficients included):
  fedavg   sample-weighted mean                          krum     Krum (one client)          mkrum   Multi-Krum (K - f clients)
  trim     coordinate-wise trimmed mean (f per side)     median   coordinate-wise median
  foolsgold  FoolsGold (Fung et al., 2020): weights from the cosine similarity of the clients' cumulative update
             histories (pardoning and logit scaling as in the reference implementation); weighted mean of the uploads
Spline-aware family (consensus knot grid = coordinate-wise median, every head re-interpolated onto it):
  spline_avg     weighted mean of all aligned clients (no robustness)
  byza           ByzaSpline-Fed: Krum-type scores from Gram-weighted functional distances of the KAN head,
                 the K - f clients with the smallest scores are averaged (all parameters)
  byza_joint     ByzaSpline-Fed with joint scoring: head functional score + Euclidean score of all non-spline
                 parameters (encoder, base weights, norms), each divided by its median, then summed
  byza_GI        ablation: G = I (Euclidean distance between aligned coefficients)
  byza_noalign   ablation: no consensus grid / re-interpolation (raw coefficients compared and averaged)
"""
import numpy as np

import splines as S

ROBUST_SPLINE = ("spline_avg", "byza", "byza_joint", "byza_GI", "byza_noalign")
METHODS = ("fedavg", "krum", "mkrum", "trim", "median", "foolsgold") + ROBUST_SPLINE


def foolsgold_weights(H):
    """H: (K, d) cumulative update histories -> client weights in [0, 1]."""
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    cs = Hn @ Hn.T - np.eye(len(H))
    maxcs = cs.max(1)
    for i in range(len(H)):
        for j in range(len(H)):
            if i != j and maxcs[i] < maxcs[j]:
                cs[i, j] *= maxcs[i] / maxcs[j]
    wv = np.clip(1.0 - cs.max(1), 0.0, 1.0)
    if wv.max() <= 0:
        return np.ones(len(H))
    wv = wv / wv.max()
    wv[wv == 1.0] = 0.99
    with np.errstate(divide="ignore"):
        wv = np.log(wv / (1.0 - wv)) + 0.5
    wv[np.isposinf(wv) | (wv > 1)] = 1.0
    wv[wv < 0] = 0.0                                   # includes -inf (a client with weight 0)
    return wv if wv.sum() > 0 else np.ones(len(H))


def layers_of(state):
    return sorted({k[: -len(".spline_weight")] for k in state if k.endswith(".spline_weight")})


def spline_keys(layers):
    return {f"{L}.{s}" for L in layers for s in ("grid", "spline_weight")}


def wmean(arrs, w):
    w = np.asarray(w, np.float64)
    w = w / w.sum()
    return sum(wi * a for wi, a in zip(w, arrs))


def flat(state, keys):
    return np.concatenate([state[k].ravel() for k in keys])


def align(ups, layers, k, grids=None):
    """Consensus knot grids (or the given ones) and every client's coefficients re-interpolated onto them.
    Returns grids {L: (in, Lk)}, aligned [{L: (out, in, P)}], mean relative re-interpolation error."""
    if grids is None:
        grids = {L: np.stack([S.consensus_knots(np.stack([u[f"{L}.grid"][i] for u in ups]))
                              for i in range(ups[0][f"{L}.grid"].shape[0])]) for L in layers}
    aligned, errs = [], []
    for u in ups:
        a = {}
        for L in layers:
            a[L], e = S.refit_layer(u[f"{L}.spline_weight"], u[f"{L}.grid"], grids[L], k)
            errs.append(e)
        aligned.append(a)
    return grids, aligned, float(np.mean(errs)) if errs else 0.0


def head_vectors(aligned, grids, layers, k, identity=False):
    """Whitened, flattened head coefficients: their Euclidean distances are the functional distances D(i, j)."""
    Ls = {L: S.layer_cholesky(grids[L], k, identity) for L in layers}
    return np.stack([np.concatenate([S.whiten(a[L], Ls[L]).ravel() for L in layers]) for a in aligned])


def aggregate(method, ups, weights, f, k, keys, history=None):
    """Returns (new state, info). `keys`: the float keys to aggregate. info['selected'] lists kept clients
    for the selection rules (None otherwise). `history`: (K, d) cumulative updates (FoolsGold only)."""
    K = len(ups)
    layers = layers_of(ups[0])
    info = {"selected": None, "reinterp_err": None}
    if method == "fedavg":
        return {kk: wmean([u[kk] for u in ups], weights) for kk in keys}, info
    if method == "foolsgold":
        fw = foolsgold_weights(history)
        info["fg_weights"] = fw.round(3).tolist()
        return {kk: wmean([u[kk] for u in ups], fw) for kk in keys}, info
    if method in ("krum", "mkrum"):
        s = S.krum_scores(S.sq_dists(np.stack([flat(u, keys) for u in ups])), f)
        sel = [int(np.argmin(s))] if method == "krum" else sorted(np.argsort(s)[: K - f].tolist())
        info["selected"] = sel
        return {kk: wmean([ups[i][kk] for i in sel], [weights[i] for i in sel]) for kk in keys}, info
    if method in ("trim", "median"):
        out = {}
        for kk in keys:
            X = np.stack([u[kk] for u in ups])
            if method == "median":
                out[kk] = np.median(X, axis=0)
            else:
                Xs = np.sort(X, axis=0)
                out[kk] = Xs[f: K - f].mean(axis=0) if K - 2 * f > 0 else np.median(X, axis=0)
        return out, info
    if method not in ROBUST_SPLINE:
        raise ValueError(method)
    skeys = spline_keys(layers)
    other = [kk for kk in keys if kk not in skeys]
    if method == "byza_noalign":
        grids = {L: wmean([u[f"{L}.grid"] for u in ups], weights) for L in layers}
        aligned = [{L: u[f"{L}.spline_weight"] for L in layers} for u in ups]
    else:
        grids, aligned, info["reinterp_err"] = align(ups, layers, k)
    sel = list(range(K))
    if method != "spline_avg" and f > 0:
        Z = head_vectors(aligned, grids, layers, k, identity=(method == "byza_GI"))
        s = S.krum_scores(S.sq_dists(Z), f)
        if method == "byza_joint":
            se = S.krum_scores(S.sq_dists(np.stack([flat(u, other) for u in ups])), f)
            s = s / max(np.median(s), 1e-30) + se / max(np.median(se), 1e-30)
        sel = sorted(np.argsort(s)[: K - f].tolist())
    if method != "spline_avg":
        info["selected"] = sel
    w = [weights[i] for i in sel]
    out = {kk: wmean([ups[i][kk] for i in sel], w) for kk in other}
    for L in layers:
        out[f"{L}.grid"] = grids[L]
        out[f"{L}.spline_weight"] = wmean([aligned[i][L] for i in sel], w)
    return out, info


def diagnostics(ups, mal, k, keys):
    """Distance diagnostics for one round: median distance malicious->honest divided by median honest->honest,
    under the raw Euclidean metric (all parameters) and the functional metric (aligned head)."""
    if not mal or len(mal) == len(ups):
        return None
    layers = layers_of(ups[0])
    hon = [i for i in range(len(ups)) if i not in mal]
    De = S.sq_dists(np.stack([flat(u, keys) for u in ups]))
    grids, aligned, _ = align(ups, layers, k)
    Df = S.sq_dists(head_vectors(aligned, grids, layers, k))
    Dh = S.sq_dists(np.stack([np.concatenate([a[L].ravel() for L in layers]) for a in aligned]))

    def ratio(D):
        hh = [D[i, j] for i in hon for j in hon if i < j]
        mh = [D[i, j] for i in mal for j in hon]
        return float(np.sqrt(np.median(mh) / max(np.median(hh), 1e-30)))

    return {"euclid_all": ratio(De), "euclid_head_aligned": ratio(Dh), "functional_head": ratio(Df)}
