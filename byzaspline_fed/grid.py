"""Experiment grid (R = 50 rounds, E = 1 local epoch; defaults of run.py). Each run is a fresh process;
finished runs are skipped, so the queue can be stopped and restarted at any time.

python grid.py --list                 # job counts and ETA per tier
python grid.py --workers 1            # everything, in tier order

Attack strength: a 20-round calibration (results/calib) showed that functional poisoning with kappa <= 64 left FedAvg at
its clean level on CWRU, Paderborn and MIMII; on C-MAPSS kappa = 256 raised the RMSE from 22.6 to 64.8. The main
comparison therefore uses two strengths: kappa = 16 (Euclidean distance ratio ~1.6, inside the honest spread) and
kappa = 256 (strong). Byzantine counts: K = 12 -> f = 4 (33.3 %); K = 20 -> f = 6 (30 %).
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("BZSF_RESULTS", os.path.join(os.path.dirname(HERE), "results"))
LOG = os.path.join(RESULTS, "queue_log.txt")
MIN = {"cwru": 3.7, "paderborn": 2.5, "mimii": 2.7, "cmapss": 6.0}          # minutes per 50-round run (RTX A4000 laptop)
DS = ("cwru", "paderborn", "mimii", "cmapss")
MAIN_F = {"cwru": 4, "paderborn": 4, "mimii": 4, "cmapss": 6}
DEF = ("fedavg", "krum", "mkrum", "trim", "median", "foolsgold", "spline_avg", "byza", "byza_joint")
SEEDS = range(3)


def J(exp, ds, method, seed, attack="none", n_mal=0, extra=()):
    return (exp, ds, method, seed, attack, n_mal, tuple(extra))


def K(kap):
    return ("--kappa", str(kap), "--tag", f"_k{kap}")


def tier1():   # main comparison, strong functional poisoning (kappa = 256)
    return [J("main", ds, m, s, "functional", MAIN_F[ds], K(256) + (("--save_model",) if s == 0 and m in ("fedavg", "krum", "byza") else ()))
            for s in SEEDS for ds in DS for m in DEF]


def tier2():   # clean training with the server tolerance of the main setting, and the pooled-data reference
    jobs = []
    for s in SEEDS:
        for ds in DS:
            f = MAIN_F[ds]
            if s == 0:
                jobs.append(J("main", ds, "central", s))
            for m in ("fedavg", "trim", "median", "spline_avg", "byza"):
                save = ("--save_model",) if (s == 0 and m == "byza") else ()
                jobs.append(J("main", ds, m, s, "none", 0, ("--f_assumed", str(f), "--tag", f"_fa{f}") + save))
    return jobs


def tier3():   # main comparison, stealthy functional poisoning (kappa = 16)
    return [J("main", ds, m, s, "functional", MAIN_F[ds], K(16)) for s in SEEDS for ds in ("cwru", "cmapss")
            for m in ("fedavg", "krum", "trim", "median", "byza", "byza_joint")]


def tier4():   # encoder-only attack (all datasets), ALIE (CWRU), adaptive functional attack (C-MAPSS, kappa = 256)
    jobs = []
    for s in range(2):
        for ds in ("cwru", "cmapss"):
            for m in ("trim", "byza", "byza_joint"):
                jobs.append(J("attacks", ds, m, s, "encoder", MAIN_F[ds]))
        for m in ("fedavg", "trim", "byza", "byza_joint"):
            jobs.append(J("attacks", "cwru", m, s, "alie", 4))
        for m in ("trim", "byza", "byza_joint"):
            jobs.append(J("attacks", "cmapss", m, s, "functional_adaptive", 6, K(256)))
    return jobs


def tier5():   # ablation and Byzantine-count sweep on C-MAPSS (kappa = 256)
    jobs = [J("main", "cmapss", m, s, "functional", 6, K(256)) for s in range(2) for m in ("byza_GI", "byza_noalign")]
    for s in range(2):
        for f in (2, 8):
            for m in ("fedavg", "trim", "byza"):
                jobs.append(J("fraction", "cmapss", m, s, "functional", f, K(256)))
    return jobs


def tier6():   # attack-strength sweep (kappa = 64, 1024) on CWRU and C-MAPSS, 2 seeds
    return [J("sens", ds, m, s, "functional", MAIN_F[ds], K(kap)) for s in range(2) for ds in ("cwru", "cmapss")
            for kap in (64, 1024) for m in ("fedavg", "krum", "trim", "byza")]


def tier7():   # server tolerance (C-MAPSS), non-IID severity without attack (CWRU), differential privacy (CWRU, clean)
    jobs = []
    for s in SEEDS:
        for fa in (3, 8):
            jobs.append(J("sens", "cmapss", "byza", s, "functional", 6, ("--kappa", "256", "--f_assumed", str(fa), "--tag", f"_k256_fa{fa}")))
        for al in (0.1, 1.0):
            for m in ("spline_avg", "byza"):
                jobs.append(J("noniid", "cwru", m, s, "none", 0, ("--alpha", str(al), "--f_assumed", "4", "--tag", f"_a{al}")))
    for s in range(2):
        for z, tag in (("4.627", "_eps8.5"), ("1.124", "_eps50")):           # splines.noise_multiplier_for(eps, 50)
            jobs.append(J("dp", "cwru", "byza", s, "none", 0, ("--dp_z", z, "--f_assumed", "4", "--tag", tag)))
    return jobs


def tier8():   # two more seeds for the strong-attack main comparison
    return [J("main", ds, m, s, "functional", MAIN_F[ds], K(256)) for s in (3, 4) for ds in DS for m in DEF]


TIERS = {1: tier1, 2: tier2, 3: tier3, 4: tier4, 5: tier5}


def name_of(job):
    exp, ds, m, s, att, f, extra = job
    tag = extra[extra.index("--tag") + 1] if "--tag" in extra else ""
    if m == "central":
        return exp, f"{ds}_central{tag}_s{s}"
    return exp, f"{ds}_{m}_{att}_m{f}{tag}_s{s}"


def minutes(job):
    return 2.5 if job[2] == "central" else MIN[job[1]]


def done(job):
    exp, name = name_of(job)
    return os.path.exists(os.path.join(RESULTS, exp, name + ".json"))


def log(msg):
    os.makedirs(RESULTS, exist_ok=True)
    line = time.strftime("%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def execute(job):
    exp, ds, m, s, att, f, extra = job
    if done(job):
        return
    cmd = [sys.executable, os.path.join(HERE, "run.py"), "--dataset", ds, "--method", m, "--seed", str(s),
           "--attack", att, "--n_mal", str(f), "--exp", exp] + list(extra)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    log(f"{name_of(job)[1]} rc={r.returncode} {time.time() - t0:.0f}s | {tail}")
    if r.returncode != 0:
        err = r.stderr.strip().splitlines()
        log("  stderr: " + (err[-1] if err else "(none)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--tiers", default="1,2,3,4,5")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    tiers = [int(t) for t in a.tiers.split(",")]
    jobs = []
    for t in tiers:
        tj = [j for j in TIERS[t]() if j not in jobs]
        todo = [j for j in tj if not done(j)]
        if a.list:
            print(f"tier {t}: {len(tj)} runs, {len(todo)} to do, ~{sum(map(minutes, todo)) / 60:.1f} GPU-hours")
        jobs += tj
    todo = [j for j in jobs if not done(j)]
    if a.list:
        print(f"total: {len(todo)} runs to do, ~{sum(map(minutes, todo)) / 60:.1f} GPU-hours sequential")
        return
    log(f"queue start: {len(todo)} runs, {a.workers} workers, ~{sum(map(minutes, todo)) / 60 / a.workers:.1f} h")
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(execute, todo))
    log("queue finished")


if __name__ == "__main__":
    main()
