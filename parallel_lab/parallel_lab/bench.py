"""Benchmark real socket all-reduce, fit an alpha-beta cost model, validate on held-out points."""
import json, sys, time
import numpy as np
from .launch import run


def _one(g, algo, size, reps):
    x0 = np.random.default_rng(g.rank).standard_normal(size).astype(np.float32)
    exp = sum(np.random.default_rng(r).standard_normal(size).astype(np.float32) for r in range(g.world))
    ts = []
    for _ in range(reps):
        x = x0.copy(); g.barrier(); t = time.perf_counter(); g.allreduce(x, algo); ts.append(time.perf_counter() - t)
    return float(np.median(ts)), float(np.abs(x - exp).max() / (np.abs(exp).max() + 1e-9))


def measure(world, algo, size, reps=5):
    r = run(world, _one, algo, size, reps)
    return max(t for t, _ in r), max(e for _, e in r)


def model_terms(algo, n, nbytes):
    """Return (latency_steps, bandwidth_bytes) so that T = alpha*steps + bytes/beta."""
    if algo == "ring": return 2 * (n - 1), 2 * (n - 1) / n * nbytes
    if algo == "star": return 2 * (n - 1), 2 * (n - 1) * nbytes          # root serializes
    if algo == "rd": return int(np.log2(n)), int(np.log2(n)) * nbytes
    raise ValueError(algo)


def fit(rows):
    A = np.array([[model_terms(r["algo"], r["n"], r["bytes"])[0], model_terms(r["algo"], r["n"], r["bytes"])[1]] for r in rows], float)
    y = np.array([r["t"] for r in rows])
    # relative least squares so small messages are not ignored
    w = 1 / y
    sol, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    alpha, inv_beta = sol
    return float(max(alpha, 1e-7)), float(1 / max(inv_beta, 1e-12))


def predict(alpha, beta, algo, n, nbytes):
    s, b = model_terms(algo, n, nbytes); return alpha * s + b / beta


def main(out="results/bench.json"):
    import os; os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for algo in ["ring", "star", "rd"]:
        for n in [2, 4, 8]:
            for size in [1 << 8, 1 << 12, 1 << 16, 1 << 18, 1 << 20, 1 << 22]:
                t, err = measure(n, algo, size)
                rows.append(dict(algo=algo, n=n, bytes=size * 4, t=t, rel_err=err))
                print(f"{algo:5s} n={n} {size*4:>9d}B  {t*1e3:9.3f} ms  rel_err={err:.1e}", flush=True)
    # fit on n in {2,4}, validate on n=8 (unseen world size)
    res = {"rows": rows, "fits": {}, "validation": {}}
    for algo in ["ring", "star", "rd"]:
        tr = [r for r in rows if r["algo"] == algo and r["n"] in (2, 4)]
        te = [r for r in rows if r["algo"] == algo and r["n"] == 8]
        a, b = fit(tr)
        mape = float(np.mean([abs(predict(a, b, algo, r["n"], r["bytes"]) - r["t"]) / r["t"] for r in te]))
        res["fits"][algo] = dict(alpha_s=a, beta_Bps=b)
        res["validation"][algo] = dict(heldout="n=8", mape=mape)
        print(f"{algo}: alpha={a*1e6:.1f} us  beta={b/1e9:.2f} GB/s  held-out n=8 MAPE={mape*100:.0f}%")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
