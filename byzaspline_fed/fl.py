"""Federated training with Byzantine clients. All K clients participate in every round (full participation, the
setting of Byzantine-robust aggregation). The coordinator has no clean validation data, so the reported model is
the last-round global model (no model selection)."""
import time

import numpy as np
import torch
import torch.nn.functional as F

import attacks as A
from aggregators import aggregate, diagnostics, layers_of
from metrics import evaluate


def loss_fn(task, out, y, x):
    if task == "cls":
        return F.cross_entropy(out, y)
    if task == "rul":                                   # the network predicts RUL / 125
        return F.huber_loss(out, y / 125.0, delta=13.0 / 125.0)
    return F.mse_loss(out, x)


def local_train(model, x, y, cfg, gen):
    n, bs = len(x), cfg["batch"]
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    model.train()
    for _ in range(cfg["local_epochs"]):
        perm = torch.randperm(n, generator=gen).to(x.device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            loss = loss_fn(cfg["task"], model(x[idx]), y[idx], x[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def to_np(sd):
    return {k: v.detach().cpu().double().numpy() for k, v in sd.items() if v.dtype.is_floating_point}


def load_np(model, glob):
    sd = model.state_dict()
    model.load_state_dict({k: (torch.from_numpy(glob[k]).to(sd[k]) if k in glob else sd[k]) for k in sd})


def flipped(task, y, n_out):
    if task == "cls":
        return n_out - 1 - y
    if task == "rul":
        return 125.0 - y
    raise ValueError("label flipping is undefined for unsupervised anomaly detection")


def run_federated(make_model, clients, cfg, seed, log=print):
    dev = cfg["device"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    model = make_model().to(dev)
    glob = to_np(model.state_dict())
    keys = list(glob)
    param_keys = [k for k, _ in model.named_parameters()]
    enc_keys = [k for k in param_keys if k.startswith("enc.")]
    layers = layers_of(glob)
    K, k_deg = len(clients), cfg["spline_k"]
    f_true, f_srv = cfg["n_mal"], cfg["f_assumed"]
    mal = sorted(np.random.default_rng(seed + 999).choice(K, f_true, replace=False).tolist()) if f_true else []
    arng = np.random.default_rng(seed + 4242)
    signs = {L: arng.choice([-1.0, 1.0], size=glob[f"{L}.spline_weight"].shape[:2]) for L in layers}
    actx = dict(param_keys=param_keys, enc_keys=enc_keys, K=K, f=f_true, k=k_deg, kappa=cfg["kappa"], task=cfg["task"],
                target=0, signs=signs, rng=arng)
    flip = {i: flipped(cfg["task"], clients[i]["y_tr"], cfg["n_out"]) for i in mal} if cfg["attack"] == "label" else {}
    weights = [len(c["y_tr"]) for c in clients]
    dp = cfg.get("dp_z", 0) > 0
    curve, det, diag, t_local, t_agg, reint, fgw = [], [], [], [], [], [], []
    hist = None
    t0 = time.time()
    for rnd in range(1, cfg["rounds"] + 1):
        ups = []
        tl = time.time()
        for i in range(K):
            load_np(model, glob)
            x, y = clients[i]["x_tr"], flip.get(i, clients[i]["y_tr"])
            local_train(model, x, y, cfg, gen)
            if cfg["grid_update"] and (rnd - 1) % cfg["grid_every"] == 0:   # knot adaptation in rounds 1, 1 + g, ...
                sub = torch.randperm(len(x), generator=gen)[:512].to(x.device)
                model.update_grids(x[sub])
            ups.append(to_np(model.state_dict()))
        t_local.append(time.time() - tl)
        if mal and cfg["attack"] not in ("none", "label"):
            bad = A.apply(cfg["attack"], [ups[i] for i in mal], glob, actx)
            for i, u in zip(mal, bad):
                ups[i] = u
        if dp:   # clip every client's update (all aggregated keys) to norm S before aggregation
            for u in ups:
                nrm = np.sqrt(sum(((u[kk] - glob[kk]) ** 2).sum() for kk in param_keys))
                c = min(1.0, cfg["dp_clip"] / (nrm + 1e-12))
                for kk in param_keys:
                    u[kk] = glob[kk] + c * (u[kk] - glob[kk])
        if mal and (rnd <= 10 or rnd % 10 == 0):
            dg = diagnostics(ups, mal, k_deg, keys)
            if dg:
                diag.append(dict(round=rnd, **dg))
        ta = time.time()
        if cfg["method"] == "foolsgold":
            upd = np.stack([np.concatenate([(u[kk] - glob[kk]).ravel() for kk in param_keys]) for u in ups])
            hist = upd if hist is None else hist + upd
        new, info = aggregate(cfg["method"], ups, weights, f_srv, k_deg, keys, history=hist)
        t_agg.append(time.time() - ta)
        if "fg_weights" in info:
            fgw.append(info["fg_weights"])
        if dp:
            # the aggregate of clipped updates lies in a ball of radius S whatever the data-dependent selection,
            # so its L2 sensitivity is 2S; Gaussian noise of std z * 2S per coordinate (parameters only)
            for kk in param_keys:
                new[kk] = new[kk] + rng.normal(0.0, cfg["dp_z"] * 2.0 * cfg["dp_clip"], size=new[kk].shape)
        if info.get("reinterp_err") is not None:
            reint.append(info["reinterp_err"])
        if info["selected"] is not None:
            rej = sorted(set(range(K)) - set(info["selected"]))
            det.append({"round": rnd, "rejected": rej, "tp": len(set(rej) & set(mal))})
        glob = {**glob, **new}
        if rnd % cfg["eval_every"] == 0 or rnd == cfg["rounds"]:
            tst = evaluate_global(model, glob, clients, cfg)
            curve.append({"round": rnd, "test": tst["primary"], "time_s": round(time.time() - t0, 1)})
            log(f"  r{rnd:3d} test {tst['primary']:.4f} ({time.time() - t0:.0f}s)")
    final = evaluate_global(model, glob, clients, cfg, full=True)
    out = {"final": final, "curve": curve, "malicious": mal, "detection": det, "distance_diagnostics": diag,
           "time_local_s_per_round": float(np.mean(t_local)), "time_agg_s_per_round": float(np.mean(t_agg)),
           "reinterp_err_mean": float(np.mean(reint)) if reint else None,
           "n_params": int(sum(v.size for kk, v in glob.items() if kk in param_keys)),
           "train_time_s": round(time.time() - t0, 1)}
    if fgw and mal:
        W = np.array(fgw)
        hon = [i for i in range(K) if i not in mal]
        out["foolsgold_weight_malicious"] = float(W[:, mal].mean())
        out["foolsgold_weight_honest"] = float(W[:, hon].mean())
    if det and mal:
        rej_tot = sum(len(d["rejected"]) for d in det)
        tp = sum(d["tp"] for d in det)
        out["detection_recall"] = tp / (len(mal) * len(det))
        out["detection_precision"] = tp / rej_tot if rej_tot else None
    return out, glob


@torch.no_grad()
def evaluate_global(model, state, clients, cfg, full=False):
    load_np(model, state)
    model.eval()
    preds, ys, groups = [], [], []
    for c in clients:
        x, y = c["x_te"], c["y_te"]
        if len(y) == 0:
            continue
        outs = []
        for j in range(0, len(x), 256):
            o = model(x[j:j + 256])
            if cfg["task"] == "ae":
                o = (o - x[j:j + 256]).pow(2).mean(dim=(1, 2))
            elif cfg["task"] == "rul":
                o = o * 125.0
            outs.append(o.float())
        preds.append(torch.nan_to_num(torch.cat(outs), nan=0.0, posinf=1e6, neginf=-1e6).cpu())
        ys.append(y.cpu())
        groups += [c["meta"]] * len(y)
    return evaluate(cfg["task"], torch.cat(preds), torch.cat(ys), groups, split="te", full=full, dataset=cfg["dataset"])


def run_central(make_model, clients, cfg, seed, log=print):
    """Pooled-data reference: one model trained on the union of all clients' training data; the epoch with the best
    pooled validation score is kept (a single data owner may hold validation data)."""
    dev = cfg["device"]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    model = make_model().to(dev)
    x = torch.cat([c["x_tr"] for c in clients])
    y = torch.cat([c["y_tr"] for c in clients])
    xv = torch.cat([c["x_va"] for c in clients if len(c["y_va"])])
    yv = torch.cat([c["y_va"] for c in clients if len(c["y_va"])])
    c2 = dict(cfg, local_epochs=1)
    best, best_state = None, None
    for ep in range(cfg["central_epochs"]):
        local_train(model, x, y, c2, gen)
        if cfg["grid_update"]:
            model.update_grids(x[torch.randperm(len(x), generator=gen)[:512].to(x.device)])
        if (ep + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                o = torch.cat([model(xv[j:j + 256]) for j in range(0, len(xv), 256)])
                v = -float(loss_fn(cfg["task"], o, yv, xv)) if cfg["task"] != "cls" else float((o.argmax(1) == yv).float().mean())
            if best is None or v > best:
                best, best_state = v, to_np(model.state_dict())
            log(f"  ep{ep + 1} val {v:.4f}")
    final = evaluate_global(model, best_state, clients, cfg, full=True)
    return {"final": final, "train_time_s": round(time.time() - t0, 1)}, best_state
