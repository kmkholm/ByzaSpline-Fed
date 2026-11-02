"""Tables, statistics and figures, computed only from results/*/*.json.

python analysis.py            -> results/analysis/{S1_Data.csv, tables.txt, tables.tex, Fig*.png/.tif}

Statistics (per dataset, functional poisoning, paired by seed): mean, s.d., 95 % t-interval, paired difference to
ByzaSpline-Fed with its 95 % t-interval and Hodges-Lehmann estimate, paired t-test p with Holm adjustment over the
comparisons of one dataset (primary), and the exact two-sided Wilcoxon p (with 5 seeds its minimum is 0.0625).
"""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("BZSF_RESULTS", os.path.join(os.path.dirname(HERE), "results"))
OUT = os.path.join(RESULTS, "analysis")
DS = ("cwru", "paderborn", "mimii", "cmapss")
DS_LABEL = {"cwru": "CWRU", "paderborn": "Paderborn", "mimii": "MIMII (fan)", "cmapss": "C-MAPSS"}
METRIC = {"cwru": "accuracy (%)", "paderborn": "accuracy (%)", "mimii": "AUC", "cmapss": "RMSE (cycles)"}
HIGHER = {"cwru": True, "paderborn": True, "mimii": True, "cmapss": False}
DEF = ("fedavg", "krum", "mkrum", "trim", "median", "foolsgold", "spline_avg", "byza", "byza_joint")
LABEL = {"fedavg": "FedAvg", "krum": "Krum", "mkrum": "Multi-Krum", "trim": "Trimmed mean", "median": "Median",
         "foolsgold": "FoolsGold",
         "spline_avg": "Spline-aware FedAvg", "byza": "ByzaSpline-Fed", "byza_joint": "ByzaSpline-Fed (joint)",
         "byza_GI": "w/o Gram weighting (G = I)", "byza_noalign": "w/o grid alignment", "central": "Pooled data"}
COLOR = {"fedavg": "#7f7f7f", "krum": "#1f77b4", "mkrum": "#17becf", "trim": "#2ca02c", "median": "#9467bd",
         "foolsgold": "#8c564b", "spline_avg": "#bcbd22", "byza": "#d62728", "byza_joint": "#ff7f0e"}
MARK = {"fedavg": "o", "krum": "s", "mkrum": "D", "trim": "^", "median": "v", "foolsgold": "h", "spline_avg": "P",
        "byza": "*", "byza_joint": "X"}


def scale(ds, v):
    return 100.0 * v if ds in ("cwru", "paderborn") else v


def load():
    rows = []
    for p in glob.glob(os.path.join(RESULTS, "*", "*.json")):
        exp = os.path.basename(os.path.dirname(p))
        if exp in ("analysis", "smoke"):
            continue
        r = json.load(open(p))
        c = r["config"]
        tag = c.get("tag", "")
        fin = r["final"]
        rows.append(dict(exp=exp, dataset=c["dataset"], method=c["method"], attack=c.get("attack", "none"),
                         n_mal=c.get("n_mal", 0), f_assumed=c.get("f_assumed", 0), tag=tag, seed=r["seed"],
                         primary=scale(c["dataset"], fin["primary"]), macro_f1=fin.get("macro_f1"),
                         phm08=fin.get("score"), detection_recall=r.get("detection_recall"),
                         detection_precision=r.get("detection_precision"),
                         time_agg_s=r.get("time_agg_s_per_round"), time_local_s=r.get("time_local_s_per_round"),
                         reinterp_err=r.get("reinterp_err_mean"), dp_epsilon=r.get("dp_epsilon"), gpu=r.get("gpu"),
                         _r=r))
    return rows


def pick(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def by_seed(rs):
    return {r["seed"]: r["primary"] for r in rs}


def msd(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return "n/a"
    return f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else f"{v.mean():.2f}"


def tci(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return (np.nan, np.nan)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return (v.mean() - h, v.mean() + h)


def hodges_lehmann(d):
    d = np.asarray(d, float)
    w = [(d[i] + d[j]) / 2 for i in range(len(d)) for j in range(i, len(d))]
    return float(np.median(w))


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def main_table(rows, out, tag="_k256"):
    lines = [f"MAIN TABLE ({tag}): clean vs functional poisoning (main f), mean ± s.d. over seeds", ""]
    tex = []
    for ds in DS:
        cen = [r["primary"] for r in pick(rows, exp="main", dataset=ds, method="central")]
        lines.append(f"## {DS_LABEL[ds]} [{METRIC[ds]}]  pooled-data reference: {msd(cen)} (n={len(cen)})")
        ref = by_seed(pick(rows, exp="main", dataset=ds, method="byza", attack="functional", tag=tag))
        comps, block = [], []
        for m in DEF + ("byza_GI", "byza_noalign"):
            clean = [r["primary"] for r in pick(rows, exp="main", dataset=ds, method=m, attack="none")]
            att = pick(rows, exp="main", dataset=ds, method=m, attack="functional", tag=tag)
            av = [r["primary"] for r in att]
            if not av and not clean:
                continue
            rec = [r["detection_recall"] for r in att if r["detection_recall"] is not None]
            line = dict(m=m, clean=msd(clean), att=msd(av), ci=tci(av), n=len(av),
                        rec=f"{np.mean(rec):.2f}" if rec else "-")
            if m != "byza" and av:
                s = by_seed(att)
                common = sorted(set(s) & set(ref))
                d = np.array([ref[k] - s[k] for k in common])
                if not HIGHER[ds]:
                    d = -d                                             # positive = ByzaSpline-Fed better
                if len(d) >= 2 and np.any(d != 0):        # paired t-test (primary); exact Wilcoxon reported alongside
                    p = float(stats.ttest_1samp(d, 0.0).pvalue) if np.std(d) > 0 else 0.0
                    line["p_wilcoxon"] = float(stats.wilcoxon(d, alternative="two-sided", method="exact").pvalue)
                else:
                    p = 1.0
                line.update(diff=f"{d.mean():+.2f} [{tci(d)[0]:+.2f}, {tci(d)[1]:+.2f}]", hl=f"{hodges_lehmann(d):+.2f}", p=p)
                comps.append(line)
            block.append(line)
        if comps:
            for c, pa in zip(comps, holm([c["p"] for c in comps])):
                c["p_holm"] = pa
        lines.append(f"{'method':28s} {'clean':>16s} {'functional':>16s} {'95% CI':>18s} {'Byza-other [95% CI]':>26s} "
                     f"{'HL':>7s} {'p':>7s} {'p_Holm':>7s} {'recall':>6s}")
        for b in block:
            ci = f"[{b['ci'][0]:.2f}, {b['ci'][1]:.2f}]" if not np.isnan(b["ci"][0]) else ""
            lines.append(f"{LABEL[b['m']]:28s} {b['clean']:>16s} {b['att']:>16s} {ci:>18s} {b.get('diff', ''):>26s} "
                         f"{b.get('hl', ''):>7s} {b.get('p', float('nan')):7.4f} {b.get('p_holm', float('nan')):7.4f} {b['rec']:>6s}"
                         f"  pW={b.get('p_wilcoxon', float('nan')):.4f}")
            tex.append(f"{DS_LABEL[ds]} & {LABEL[b['m']]} & {b['clean']} & {b['att']} & {b.get('diff', '--')} & "
                       f"{b.get('p_holm', float('nan')):.3f} \\\\")
        lines.append("")
    out["txt"] += lines
    out["tex"] += [f"% main table rows ({tag}): dataset & method & clean & functional & difference [95% CI] & Holm p"] + tex


def sweep_table(rows, out, exp, key, title):
    lines = [title]
    groups = defaultdict(list)
    for r in pick(rows, exp=exp):
        groups[(r["dataset"], r[key], r["method"], r["tag"])].append(r["primary"])
    for k in sorted(groups, key=str):
        lines.append(f"  {k}: {msd(groups[k])} (n={len(groups[k])})")
    out["txt"] += lines + [""]


def overhead(rows, out):
    lines = ["SERVER AGGREGATION TIME per round (s) and local training time per round (s), CWRU main runs"]
    for m in DEF:
        rs = pick(rows, exp="main", dataset="cwru", method=m)
        if rs:
            ta = [r["time_agg_s"] for r in rs if r["time_agg_s"] is not None]
            tl = [r["time_local_s"] for r in rs if r["time_local_s"] is not None]
            lines.append(f"  {LABEL[m]:28s} agg {np.mean(ta):.3f}  local {np.mean(tl):.2f}  gpu {rs[0]['gpu']}")
    out["txt"] += lines + [""]


def distances(rows, out):
    lines = ["DISTANCE DIAGNOSTICS: median distance malicious->honest / honest->honest (1 = indistinguishable)"]
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for d in r["_r"].get("distance_diagnostics", []):
            for k in ("euclid_all", "euclid_head_aligned", "functional_head"):
                g[(r["dataset"], r["attack"], r["tag"])][k].append(d[k])
    for k in sorted(g, key=str):
        lines.append(f"  {k}: " + "  ".join(f"{m}={np.mean(v):.2f}" for m, v in g[k].items()))
    out["txt"] += lines + [""]


def figures(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10})

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
        fig.savefig(os.path.join(OUT, name + ".tif"), dpi=300, pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)

    # convergence under strong functional poisoning (kappa = 256, all 50 rounds)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, ds in zip(axes, DS):
        for m in DEF:
            cs = [r["_r"]["curve"] for r in pick(rows, exp="main", dataset=ds, method=m, attack="functional", tag="_k256")]
            if not cs:
                continue
            x = [c["round"] for c in cs[0]]
            Y = np.array([[scale(ds, c["test"]) for c in cv] for cv in cs if len(cv) == len(x)])
            mu, sd = Y.mean(0), Y.std(0)
            ax.plot(x, mu, color=COLOR[m], marker=MARK[m], markevery=4, ms=4, lw=2.2 if m.startswith("byza") else 1.2,
                    label=LABEL[m])
            ax.fill_between(x, mu - sd, mu + sd, color=COLOR[m], alpha=0.12)
        ax.set_xlabel("Communication round")
        ax.set_ylabel(METRIC[ds])
        ax.set_title(DS_LABEL[ds])
        ax.grid(alpha=0.3)
        if ds == "cmapss":
            ax.set_ylim(15, 40)
    axes[0].legend(fontsize=7, ncol=2)
    save(fig, "convergence_functional")

    # Byzantine fraction sweep (C-MAPSS, K = 20, kappa = 256)
    fig, axes = plt.subplots(1, 1, figsize=(5.2, 3.6))
    axes = [axes]
    for ax, ds in zip(axes, ("cmapss",)):
        for m in DEF:
            xs, mu, sd = [], [], []
            for f in (2, 4, 6, 8):
                rs = pick(rows, exp="fraction" if f != 6 else "main", dataset=ds, method=m, attack="functional", n_mal=f,
                          tag="_k256")
                if rs:
                    v = [r["primary"] for r in rs]
                    xs.append(100 * f / 20)
                    mu.append(np.mean(v))
                    sd.append(np.std(v, ddof=1) if len(v) > 1 else 0)
            if xs:
                ax.errorbar(xs, mu, yerr=sd, color=COLOR[m], marker=MARK[m], capsize=3, lw=1.4, label=LABEL[m])
        ax.set_xlabel("Byzantine clients (% of K = 20)")
        ax.set_ylabel(METRIC[ds] + " (clipped at 40)")
        ax.set_title(DS_LABEL[ds])
        ax.set_ylim(18, 40)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    save(fig, "fraction_sweep")

    # (a-d) encoder-only attack on the four datasets, (e) ALIE on CWRU, (f) functional vs adaptive functional on C-MAPSS
    fig, axes = plt.subplots(1, 6, figsize=(20, 3.6))
    enc_m = ("fedavg", "krum", "trim", "byza", "byza_joint")
    for ax, ds in zip(axes[:4], DS):
        for j, m in enumerate(enc_m):
            v = [r["primary"] for r in pick(rows, exp="attacks", dataset=ds, method=m, attack="encoder")]
            ax.bar(j, np.mean(v) if v else np.nan, 0.7, yerr=np.std(v, ddof=1) if len(v) > 1 else 0, color=COLOR[m],
                   label=LABEL[m], capsize=2)
        ax.set_xticks(range(len(enc_m)))
        ax.set_xticklabels([LABEL[m].replace("ByzaSpline-Fed", "BSF") for m in enc_m], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(METRIC[ds])
        ax.set_title(f"{DS_LABEL[ds]}: encoder-only")
        ax.grid(alpha=0.3, axis="y")
    cm = ("fedavg", "krum", "trim", "median", "foolsgold", "byza", "byza_joint")
    for ax, ds, atts, xl in ((axes[4], "cwru", (("attacks", "alie", ""),), ["ALIE"]),
                             (axes[5], "cmapss", (("main", "functional", "_k256"), ("attacks", "functional_adaptive", "_k256")),
                              ["Functional", "Adaptive"])):
        w = 0.8 / len(cm)
        for j, m in enumerate(cm):
            mu, sd = [], []
            for exp, a, tg in atts:
                v = [r["primary"] for r in pick(rows, exp=exp, dataset=ds, method=m, attack=a, tag=tg)]
                mu.append(np.mean(v) if v else np.nan)
                sd.append(np.std(v, ddof=1) if len(v) > 1 else 0)
            ax.bar(np.arange(len(atts)) + (j - len(cm) / 2 + 0.5) * w, mu, w, yerr=sd, color=COLOR[m], label=LABEL[m],
                   capsize=1.5, error_kw={"lw": 0.6})
        ax.set_xticks(range(len(atts)))
        ax.set_xticklabels(xl, fontsize=9)
        ax.set_ylabel(METRIC[ds])
        ax.set_title(DS_LABEL[ds])
        ax.grid(alpha=0.3, axis="y")
    axes[5].legend(fontsize=6.5, ncol=2, loc="upper left")
    save(fig, "attack_types")

    # mean confusion matrices, CWRU, functional poisoning: Krum vs ByzaSpline-Fed
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, m in zip(axes, ("krum", "byza")):
        cms = [np.array(r["_r"]["final"]["confusion"], float) for r in pick(rows, exp="main", dataset="cwru", method=m,
                                                                             attack="functional", tag="_k256")]
        if not cms:
            continue
        cm = np.mean(cms, 0)
        cm = cm / cm.sum(1, keepdims=True).clip(min=1)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > 0.5 else "black")
        lab = ["N", "IR7", "IR14", "IR21", "B7", "B14", "B21", "OR7", "OR14", "OR21"]
        ax.set_xticks(range(10))
        ax.set_xticklabels(lab, rotation=90, fontsize=7)
        ax.set_yticks(range(10))
        ax.set_yticklabels(lab, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(LABEL[m])
    save(fig, "confusion_cwru_functional")


def spline_figure():
    """Learned edge functions of the output layer (CWRU, seed 0): clean ByzaSpline-Fed vs Krum and ByzaSpline-Fed under
    functional poisoning."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import splines as S
    files = {"clean": "main/cmapss_byza_none_m0_fa6_s0_state.npz",
             "Krum, attacked": "main/cmapss_krum_functional_m6_k256_s0_state.npz",
             "ByzaSpline-Fed, attacked": "main/cmapss_byza_functional_m6_k256_s0_state.npz"}
    st = {k: np.load(os.path.join(RESULTS, v)) for k, v in files.items() if os.path.exists(os.path.join(RESULTS, v))}
    if len(st) < 2:
        return
    fig, axes = plt.subplots(1, 4, figsize=(13, 3))
    cols = {"clean": "#2ca02c", "Krum, attacked": "#1f77b4", "ByzaSpline-Fed, attacked": "#d62728"}
    for ax, i in zip(axes, range(4)):
        for name, z in st.items():
            g = z["kan2.grid"][i]
            u = np.linspace(g[3], g[-4] - 1e-6, 300)
            phi = S.basis(u, g, 3) @ z["kan2.spline_weight"][0, i]
            ax.plot(u, phi, color=cols[name], lw=1.8, label=name)
        ax.set_xlabel(f"input {i} of output layer")
        ax.set_ylabel("spline part of phi (to RUL output)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "spline_edges_cmapss.png"), dpi=300)
    fig.savefig(os.path.join(OUT, "spline_edges_cmapss.tif"), dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load()
    with open(os.path.join(OUT, "S1_Data.csv"), "w", newline="") as fh:
        cols = [k for k in rows[0] if not k.startswith("_")]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["exp"], r["dataset"], r["method"], r["attack"], r["n_mal"], r["tag"], r["seed"])):
            w.writerow({k: r[k] for k in cols})
    out = {"txt": [f"{len(rows)} runs"], "tex": []}
    main_table(rows, out, "_k256")
    main_table(rows, out, "_k16")
    sweep_table(rows, out, "attacks", "attack", "OTHER ATTACKS (main f)")
    sweep_table(rows, out, "fraction", "n_mal", "BYZANTINE FRACTION SWEEP")
    sweep_table(rows, out, "sens", "tag", "SENSITIVITY (CWRU, f = 4)")
    sweep_table(rows, out, "noniid", "tag", "NON-IID (CWRU)")
    sweep_table(rows, out, "dp", "tag", "DIFFERENTIAL PRIVACY (CWRU)")
    overhead(rows, out)
    distances(rows, out)
    open(os.path.join(OUT, "tables.txt"), "w", encoding="utf-8").write("\n".join(out["txt"]))
    open(os.path.join(OUT, "tables.tex"), "w", encoding="utf-8").write("\n".join(out["tex"]))
    figures(rows)
    spline_figure()
    print("\n".join(out["txt"]))


if __name__ == "__main__":
    main()
