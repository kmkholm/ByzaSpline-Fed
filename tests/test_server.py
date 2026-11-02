"""numpy-only checks of the aggregation rules and attacks on synthetic client states (no torch, no data):
python tests/test_server.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "byzaspline_fed"))
import attacks as A  # noqa: E402
import splines as S  # noqa: E402
from aggregators import METHODS, aggregate, diagnostics  # noqa: E402


def make_states(K, rng, k=3, G=8):
    t = S.uniform_knots(G, k)
    glob = {"enc.w": rng.normal(size=(50,)), "kan1.grid": np.tile(t, (6, 1)), "kan1.spline_weight": rng.normal(size=(4, 6, G + k)),
            "kan1.base_weight": rng.normal(size=(4, 6)), "kan2.grid": np.tile(t, (4, 1)),
            "kan2.spline_weight": rng.normal(size=(3, 4, G + k)), "kan2.base_weight": rng.normal(size=(3, 4))}
    ups = []
    for _ in range(K):
        u = {kk: v + 0.05 * rng.normal(size=v.shape) for kk, v in glob.items() if not kk.endswith("grid")}
        for L in ("kan1", "kan2"):                     # client-adapted knot vectors (still increasing)
            g = glob[f"{L}.grid"] + 0.02 * rng.normal(size=glob[f"{L}.grid"].shape)
            u[f"{L}.grid"] = np.sort(g, axis=1)
        ups.append(u)
    return glob, ups


def main():
    rng = np.random.default_rng(1)
    K, f, k = 12, 4, 3
    glob, ups = make_states(K, rng)
    keys = list(glob)
    pk = [kk for kk in keys if not kk.endswith("grid")]
    w = [100 + i for i in range(K)]
    # every rule runs and keeps shapes; spline-aware grids are strictly increasing
    hist = np.stack([np.concatenate([(u[kk] - glob[kk]).ravel() for kk in pk]) for u in ups])
    for m in METHODS:
        out, info = aggregate(m, ups, w, f, k, keys, history=hist)
        assert all(out[kk].shape == glob[kk].shape for kk in keys), m
        for L in ("kan1", "kan2"):
            assert np.all(np.diff(out[f"{L}.grid"], axis=1) > 0), m
    # reduction property: f = 0 -> ByzaSpline-Fed == spline-aware FedAvg
    a, _ = aggregate("byza", ups, w, 0, k, keys)
    b, _ = aggregate("spline_avg", ups, w, 0, k, keys)
    assert all(np.allclose(a[kk], b[kk]) for kk in keys)
    # with f > 0 and honest clients only, f clients are still discarded
    _, info = aggregate("byza", ups, w, f, k, keys)
    assert len(info["selected"]) == K - f
    # attacks
    signs = {L: rng.choice([-1.0, 1.0], size=glob[f"{L}.spline_weight"].shape[:2]) for L in ("kan1", "kan2")}
    ctx = dict(param_keys=pk, enc_keys=["enc.w"], K=K, f=f, k=k, kappa=4.0, task="cls", target=0, signs=signs, rng=rng)
    mal = list(range(f))
    for att in A.ATTACKS:
        if att in ("none", "label"):
            continue
        bad = A.apply(att, [ups[i] for i in mal], glob, ctx)
        assert all(set(b) == set(glob) and all(b[kk].shape == glob[kk].shape for kk in keys) for b in bad), att
        poisoned = list(ups)
        for i, u in zip(mal, bad):
            poisoned[i] = u
        dg = diagnostics(poisoned, mal, k, keys)
        out, info = aggregate("byza", poisoned, w, f, k, keys)
        rec = len(set(mal) - set(info["selected"])) / f
        print(f"  {att:20s} distance ratio euclid_all={dg['euclid_all']:.2f} head_aligned={dg['euclid_head_aligned']:.2f} "
              f"functional={dg['functional_head']:.2f} | byza recall {rec:.2f}")
    # FoolsGold down-weights clients with identical histories (sybils)
    from aggregators import foolsgold_weights
    Hs = rng.normal(size=(12, 40))
    Hs[:3] = Hs[0] + 1e-3 * rng.normal(size=(3, 40))
    fw = foolsgold_weights(Hs)
    assert fw[:3].max() < fw[3:].min(), fw
    # encoder-only attack leaves the spline head untouched
    bad = A.apply("encoder", [ups[0]], glob, dict(ctx, f=1))
    assert np.array_equal(bad[0]["kan1.spline_weight"], ups[0]["kan1.spline_weight"])
    assert not np.array_equal(bad[0]["enc.w"], ups[0]["enc.w"])
    print("all server tests passed")


if __name__ == "__main__":
    main()
