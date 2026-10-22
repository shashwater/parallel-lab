import numpy as np, os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from parallel_lab.launch import run
from parallel_lab.planner import Model, Cluster, search, Plan, evaluate


def _ar(g, algo, size):
    x = np.random.default_rng(g.rank).standard_normal(size).astype(np.float32); g.allreduce(x, algo); return x


def _expect(n, size): return sum(np.random.default_rng(r).standard_normal(size).astype(np.float32) for r in range(n))


def test_ring_any_world_and_odd_sizes():
    for n in (2, 3, 5):
        for size in (1, 7, 1000, 100003):
            out = run(n, _ar, "ring", size)
            for o in out: np.testing.assert_allclose(o, _expect(n, size), rtol=1e-5, atol=1e-5)
            assert all(np.array_equal(out[0], o) for o in out), "ranks must agree bit-for-bit"


def test_star_and_rd_match_ring():
    for algo, n in (("star", 3), ("rd", 4), ("rd", 8)):
        out = run(n, _ar, algo, 4097)
        for o in out: np.testing.assert_allclose(o, _expect(n, 4097), rtol=1e-5, atol=1e-5)


def test_dp_training_matches_single_process():
    from parallel_lab.train_dp import train
    base = run(1, train, 15, 64, "ring")[0]
    r = run(3, train, 15, 64, "ring")
    for a, b in zip(r[0]["params"], base["params"]): np.testing.assert_allclose(a, b, atol=5e-4)
    assert r[0]["losses"][-1] < r[0]["losses"][0]


def test_planner_memory_and_degrees():
    mo = Model(96, 12288, 96); cl = Cluster(1536)
    best = search(mo, cl, 2304, top=3)
    assert best and all(p.dp * p.tp * p.pp == 1536 and p.mem_gb <= 80 for p in best)
    assert evaluate(mo, cl, 2304, Plan(1536, 1, 1, 1, 1, True, False)).mem_gb > 80   # pure DP of 175B cannot fit


if __name__ == "__main__":   # plain runner (also works under pytest)
    import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for name, fn in list(globals().items()):
        if name.startswith("test_"): fn(); print("PASS", name)
