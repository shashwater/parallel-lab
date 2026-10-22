import multiprocessing as mp, random
from .collectives import Group


def _worker(rank, world, port, fn, args, q):
    g = Group(rank, world, port)
    try:
        q.put((rank, fn(g, *args)))
    finally:
        g.close()


def run(world, fn, *args):
    """Run fn(group, *args) on `world` processes; return list of per-rank results."""
    port = random.randint(20000, 50000)
    ctx = mp.get_context("fork"); q = ctx.Queue()
    ps = [ctx.Process(target=_worker, args=(r, world, port, fn, args, q)) for r in range(world)]
    for p in ps: p.start()
    res = dict(q.get(timeout=300) for _ in ps)
    for p in ps: p.join()
    return [res[r] for r in range(world)]
