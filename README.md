# ByzaSpline-Fed

Byzantine-robust federated learning for models with B-spline Kolmogorov–Arnold (KAN) heads.
The server compares clients by the Gram-weighted functional distance between their spline heads on a
consensus knot grid instead of the Euclidean distance between raw coefficients, discards the clients with
the largest Krum-type scores, and averages the rest. The repository also contains the functional
spline-poisoning attack used to evaluate the rule, together with reference aggregation rules and attacks.

## Layout

```
byzaspline_fed/
  data.py        client partitions for CWRU, Paderborn, MIMII (fan) and C-MAPSS (recording/engine-level splits)
  models.py      Mamba encoder (pure-PyTorch selective scan) + B-spline KAN head with client-side knot adaptation
  splines.py     B-spline basis, Gram matrix, consensus knots, re-interpolation, whitening, Krum scores, DP accountant
  aggregators.py FedAvg, Krum, Multi-Krum, trimmed mean, median, FoolsGold, spline-aware FedAvg, ByzaSpline-Fed
  attacks.py     label flipping, sign flipping, Gaussian, ALIE, functional spline poisoning (+ adaptive), encoder-only
  fl.py          federated loop (full participation, last-round model, detection / timing / distance logs)
  run.py         one run -> results/<exp>/<name>.json
  grid.py        experiment grid with ETA and parallel workers
  analysis.py    tables (mean, s.d., CI, paired tests) and figures from the result files
partitions/      client assignment of every recording / engine / clip (JSON)
tests/           numpy-only tests of the server-side code
RUN_ALL.py       tests -> partitions -> smoke runs -> grid -> analysis, in one command
```

## Data

Set `BZSF_DATA` to a folder that contains

- `CWRU/` with the 12 kHz drive-end and normal-baseline `.mat` files of the Case Western Reserve University
  bearing data centre (the 10-class partition is built from them on first use), and
- `cache/` with the cached Paderborn, MIMII-fan and C-MAPSS client partitions (`paderborn_clients.npz`,
  `cmapss_clients.npz`, `mimii_fan_logmel.npz`), or the raw Paderborn, MIMII and C-MAPSS files, from which
  `data.py` builds them.

All four datasets are public; see the references in `data.py`.

## Usage

```bash
pip install -r requirements.txt
export BZSF_DATA=/path/to/data            # folder with CWRU/ and cache/
python tests/test_splines.py && python tests/test_server.py
python byzaspline_fed/data.py cwru paderborn mimii cmapss
python byzaspline_fed/run.py --dataset cwru --method byza --attack functional --n_mal 4 --kappa 256 --seed 0
python byzaspline_fed/grid.py --list
python byzaspline_fed/grid.py --workers 1
python byzaspline_fed/analysis.py
```

Methods: `central fedavg krum mkrum trim median foolsgold spline_avg byza byza_joint byza_GI byza_noalign`.
Attacks: `none label sign gauss alie functional functional_adaptive encoder`.
Every run writes one JSON file with its full configuration, seed, per-round test metric, Byzantine
identities, detection recall / precision and timing, so the queue can be stopped and restarted at any time.

Defaults: 50 rounds, one local epoch, all clients every round, Adam (lr 1e-3), cubic B-splines on 8
intervals, knot adaptation every 10 rounds. Byzantine clients: 4 of 12 (CWRU, Paderborn, MIMII), 6 of 20
(C-MAPSS); the functional attack strength is `--kappa` (16 = stealthy, 256 = strong).

## License

MIT.
