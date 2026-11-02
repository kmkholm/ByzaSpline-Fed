"""One experiment -> results/<exp>/<name>.json (existing results are skipped).

python run.py --dataset cwru --method byza --attack functional --n_mal 4 --seed 0
methods: central | fedavg krum mkrum trim median | spline_avg byza byza_joint byza_GI byza_noalign
attacks: none label sign gauss alie functional functional_adaptive encoder
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D                                       # noqa: E402
import splines as S                                    # noqa: E402
from aggregators import METHODS                        # noqa: E402
from attacks import ATTACKS                            # noqa: E402
from fl import run_central, run_federated              # noqa: E402
from models import Net                                 # noqa: E402

RESULTS = os.environ.get("BZSF_RESULTS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))


def run_name(a):
    if a.method == "central":
        return f"{a.dataset}_central{a.tag}_s{a.seed}"
    return f"{a.dataset}_{a.method}_{a.attack}_m{a.n_mal}{a.tag}_s{a.seed}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(D.TASK))
    ap.add_argument("--method", required=True, choices=("central",) + METHODS)
    ap.add_argument("--attack", default="none", choices=ATTACKS)
    ap.add_argument("--n_mal", type=int, default=0, help="number of Byzantine clients")
    ap.add_argument("--f_assumed", type=int, default=-1, help="server tolerance f (-1: = n_mal)")
    ap.add_argument("--kappa", type=float, default=4.0, help="functional attack: harm target relative to honest energy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.5, help="CWRU Dirichlet concentration (<= 0: IID)")
    ap.add_argument("--spline_k", type=int, default=3, help="B-spline degree (3 = cubic)")
    ap.add_argument("--grid_G", type=int, default=8)
    ap.add_argument("--no_grid_update", action="store_true")
    ap.add_argument("--grid_every", type=int, default=10, help="knot adaptation every g rounds")
    ap.add_argument("--dp_z", type=float, default=0.0, help="Gaussian noise multiplier on the aggregate (0 = off)")
    ap.add_argument("--dp_clip", type=float, default=1.0)
    ap.add_argument("--central_epochs", type=int, default=60)
    ap.add_argument("--exp", default="main")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save_model", action="store_true")
    a = ap.parse_args(argv)
    if a.f_assumed < 0:
        a.f_assumed = a.n_mal
    dp = a.dp_z > 0

    name = run_name(a)
    out_dir = os.path.join(RESULTS, a.exp)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + ".json")
    if os.path.exists(path):
        print("exists", path)
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c_in, length, patch, batch = D.SHAPES[a.dataset]
    task = D.TASK[a.dataset]
    clients = D.load(a.dataset, alpha=a.alpha) if a.dataset == "cwru" else D.load(a.dataset)
    for c in clients:
        for kk in list(c):
            if kk.startswith(("x_", "y_")):
                c[kk] = torch.as_tensor(np.asarray(c[kk])).to(dev)
    K = len(clients)
    assert a.f_assumed < K / 2, "the rule needs f < K/2"
    if a.method in ("krum", "mkrum"):
        assert K >= 2 * a.f_assumed + 3, "Krum needs K >= 2f + 3"
    cfg = dict(dataset=a.dataset, method=a.method, attack=a.attack, n_mal=a.n_mal, f_assumed=a.f_assumed, kappa=a.kappa,
               task=task, n_out=D.N_OUT[a.dataset], rounds=a.rounds, local_epochs=a.local_epochs, batch=batch, lr=a.lr,
               wd=1e-4, spline_k=a.spline_k, grid_G=a.grid_G, grid_update=not (a.no_grid_update or dp), grid_every=a.grid_every, dp_z=a.dp_z,
               dp_clip=a.dp_clip, alpha=a.alpha if a.dataset == "cwru" else None, central_epochs=a.central_epochs,
               eval_every=5, device=dev, clients=K, malicious_fraction=a.n_mal / K, tag=a.tag)

    def make_model():
        return Net(task, D.N_OUT[a.dataset], c_in, length, patch=patch, G=a.grid_G, k=a.spline_k,
                   norm="group" if dp else "bn")

    print(f"== {name}  ({K} clients, {a.n_mal} Byzantine, task {task})", flush=True)
    t0 = time.time()
    if a.method == "central":
        res, state = run_central(make_model, clients, cfg, a.seed)
    else:
        res, state = run_federated(make_model, clients, cfg, a.seed)
    if dp:
        res["dp_epsilon"] = S.gaussian_epsilon(a.dp_z, a.rounds, 1e-5)
        res["dp_delta"] = 1e-5
    res.update(config={k: v for k, v in cfg.items() if k != "device"}, seed=a.seed, name=name,
               wall_s=round(time.time() - t0, 1), torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", python=platform.python_version())
    if a.save_model and state is not None:
        np.savez_compressed(os.path.join(out_dir, name + "_state.npz"), **state)
    with open(path + ".tmp", "w") as f:
        json.dump(res, f)
    os.replace(path + ".tmp", path)
    print(f"done {name}: primary {res['final']['primary']:.4f} in {res['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
