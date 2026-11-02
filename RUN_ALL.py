"""Run the whole ByzaSpline-Fed study on a GPU machine with one command:

    python RUN_ALL.py              (inside the unzipped server kit; any conda env with PyTorch >= 1.12, Python >= 3.8)

Steps: install missing pip packages -> unit tests -> client partitions -> experiment grid (tiers 1-7, finished runs are
skipped, so the command can simply be started again after an interruption) -> tables/figures -> results zip.
Options:  --workers N (default 4 concurrent runs on the GPU)   --tiers 1,4 (subset)   --no-zip
Output:   results/ (one JSON per run, queue_log.txt, analysis/)  and  byzaspline_results.zip  to send back.
"""
import argparse
import importlib
import os
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "byzaspline_fed")
NEEDED = {"numpy": "numpy", "scipy": "scipy", "sklearn": "scikit-learn", "matplotlib": "matplotlib", "torch": None}


def sh(args, env=None):
    print(">>", " ".join(args), flush=True)
    r = subprocess.run(args, env=env)
    if r.returncode != 0:
        sys.exit(f"step failed: {' '.join(args)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tiers", default="1,4,2,3,5,6,7")
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()

    for mod, pipname in NEEDED.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            if pipname is None:
                sys.exit("PyTorch is missing in this environment: activate the env that has torch (e.g. conda activate vlm)")
            sh([sys.executable, "-m", "pip", "install", pipname])
    import torch
    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

    env = dict(os.environ)
    env.setdefault("BZSF_DATA", os.path.join(HERE, "data"))
    env.setdefault("BZSF_RESULTS", os.path.join(HERE, "results"))
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    if not os.path.isdir(os.path.join(env["BZSF_DATA"], "CWRU")):
        sys.exit(f"data folder not found: {env['BZSF_DATA']} (needs CWRU/ and cache/)")

    sh([sys.executable, os.path.join(HERE, "tests", "test_splines.py")], env)
    sh([sys.executable, os.path.join(HERE, "tests", "test_server.py")], env)
    sh([sys.executable, os.path.join(PKG, "data.py"), "cwru", "paderborn", "mimii", "cmapss"], env)
    # smoke test: 1 round per dataset through attack, alignment, scoring and evaluation (results/smoke, not analysed)
    for ds in ("cwru", "paderborn", "mimii", "cmapss"):
        sh([sys.executable, os.path.join(PKG, "run.py"), "--dataset", ds, "--method", "byza_joint", "--attack", "functional",
            "--n_mal", "4" if ds != "cmapss" else "6", "--rounds", "1", "--exp", "smoke"], env)
    sh([sys.executable, os.path.join(PKG, "run.py"), "--dataset", "cwru", "--method", "foolsgold", "--attack", "encoder",
        "--n_mal", "4", "--rounds", "1", "--exp", "smoke"], env)
    sh([sys.executable, os.path.join(PKG, "run.py"), "--dataset", "cwru", "--method", "central", "--central_epochs", "2",
        "--exp", "smoke"], env)
    sh([sys.executable, os.path.join(PKG, "grid.py"), "--list", "--tiers", a.tiers], env)
    for t in a.tiers.split(","):                     # tier by tier, so the main table is complete first
        sh([sys.executable, os.path.join(PKG, "grid.py"), "--tiers", t, "--workers", str(a.workers)], env)
        r = subprocess.run([sys.executable, os.path.join(PKG, "analysis.py")], env=env)   # never stops the queue
        print("analysis rc", r.returncode, flush=True)
    if not a.no_zip:
        out = os.path.join(HERE, "byzaspline_results.zip")
        res = env["BZSF_RESULTS"]
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(res):
                for fn in files:
                    p = os.path.join(root, fn)
                    z.write(p, os.path.relpath(p, os.path.dirname(res)))
            for root, _, files in os.walk(os.path.join(HERE, "partitions")):
                for fn in files:
                    p = os.path.join(root, fn)
                    z.write(p, os.path.relpath(p, HERE))
        print("\nDONE. Send back:", out, flush=True)


if __name__ == "__main__":
    main()
