"""Real data-parallel training of a 2-layer MLP. Gradients are averaged with the socket ring all-reduce.
Check: N-rank result must equal single-process full-batch training."""
import time, sys
import numpy as np
from .launch import run


def make_data(n=4096, d=64, k=10, seed=0):
    r = np.random.default_rng(seed); X = r.standard_normal((n, d)).astype(np.float32)
    W = r.standard_normal((d, k)).astype(np.float32); y = (X @ W + 0.3 * r.standard_normal((n, k))).argmax(1)
    return X, y


def init(d, h, k, seed=1):
    r = np.random.default_rng(seed)
    return [r.standard_normal((d, h)).astype(np.float32) * 0.1, np.zeros(h, np.float32), r.standard_normal((h, k)).astype(np.float32) * 0.1, np.zeros(k, np.float32)]


def grads(P, X, y, total):
    W1, b1, W2, b2 = P; z = X @ W1 + b1; a = np.maximum(z, 0); lg = a @ W2 + b2
    lg -= lg.max(1, keepdims=True); p = np.exp(lg); p /= p.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-9).sum() / total
    dl = p; dl[np.arange(len(y)), y] -= 1; dl /= total
    dW2 = a.T @ dl; db2 = dl.sum(0); da = dl @ W2.T; da[z <= 0] = 0
    return loss, [X.T @ da, da.sum(0), dW2, db2]


def train(g, steps, h, algo):
    X, y = make_data(); n = len(y); P = init(X.shape[1], h, 10)
    lo, hi = g.rank * n // g.world, (g.rank + 1) * n // g.world
    sizes = [p.size for p in P]; t0 = time.perf_counter(); comm = 0.0; losses = []
    for _ in range(steps):
        loss, G = grads(P, X[lo:hi], y[lo:hi], n)           # each rank: partial sum / global batch
        flat = np.concatenate([np.append(gr.ravel(), loss) for gr in G]).astype(np.float32) if False else np.concatenate([gr.ravel() for gr in G] + [np.array([loss], np.float32)])
        t = time.perf_counter(); g.allreduce(flat, algo); comm += time.perf_counter() - t
        off = 0
        for p, s in zip(P, sizes):
            p -= 0.5 * flat[off:off + s].reshape(p.shape); off += s
        losses.append(float(flat[-1]))
    return dict(params=[p.copy() for p in P], losses=losses, wall=time.perf_counter() - t0, comm=comm)


def main(steps=60, h=1024):
    base = run(1, train, steps, h, "ring")[0]
    print(f"1 rank : loss {base['losses'][0]:.3f} -> {base['losses'][-1]:.3f}, wall {base['wall']:.2f}s")
    out = {"steps": steps, "hidden": h, "single": dict(wall=base["wall"], loss=base["losses"][-1]), "runs": []}
    nparams = sum(p.size for p in base["params"])
    for n in [2, 4]:
        r = run(n, train, steps, h, "ring")
        diff = max(float(np.abs(a - b).max()) for a, b in zip(r[0]["params"], base["params"]))
        same = all(all(np.array_equal(a, b) for a, b in zip(r[0]["params"], x["params"])) for x in r)
        w = max(x["wall"] for x in r); c = max(x["comm"] for x in r)
        print(f"{n} ranks: loss {r[0]['losses'][-1]:.3f}, max |param diff| vs 1 rank = {diff:.2e}, ranks identical={same}, wall {w:.2f}s (comm {c:.2f}s), speedup {base['wall']/w:.2f}x, params {nparams}")
        out["runs"].append(dict(n=n, max_param_diff=diff, ranks_identical=same, wall=w, comm=c, speedup=base["wall"] / w, loss=r[0]["losses"][-1]))
    import json, os; os.makedirs("results", exist_ok=True); json.dump(out, open("results/train_dp.json", "w"), indent=1)


if __name__ == "__main__":
    main()
