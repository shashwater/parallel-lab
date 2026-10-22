"""Analytical planner for 3D parallelism (data x tensor x pipeline) of a GPT-style model.
All hardware numbers are inputs (assumptions), not measurements. The collective cost model uses the
same alpha-beta form fitted on the real socket benchmarks in bench.py."""
from dataclasses import dataclass
import itertools, math


@dataclass
class Model:
    layers: int; hidden: int; heads: int; seq: int = 2048; vocab: int = 51200
    @property
    def params(self): return 12 * self.layers * self.hidden ** 2 * (1 + 13 / (12 * self.hidden)) + (self.vocab + self.seq) * self.hidden


@dataclass
class Cluster:
    gpus: int; gpus_per_node: int = 8; mem_gb: float = 80.0
    peak_tflops: float = 312.0            # A100 fp16/bf16 dense peak
    gemm_eff: float = 0.55                # asymptotic fraction of peak a GEMM sustains (calibrated)
    eff_h0: float = 0.0                   # if >0: eff(h) = gemm_eff * h/(h+eff_h0), bigger hidden size -> bigger GEMMs -> higher efficiency
    intra_busbw: float = 230e9            # B/s NVLink all-reduce bus bandwidth per GPU
    inter_bw_per_gpu: float = 20e9        # B/s per GPU over the node fabric (8x200Gb IB / 8 GPUs, derated)
    alpha_intra: float = 10e-6; alpha_inter: float = 30e-6


@dataclass
class Plan:
    dp: int; tp: int; pp: int; micro_b: int; m: int; recompute: bool; zero1: bool
    step_s: float = 0; mem_gb: float = 0; tflops_per_gpu: float = 0; breakdown: dict = None
    def key(self): return (self.dp, self.tp, self.pp, self.micro_b)


def ring_time(nbytes, n, busbw, alpha):
    return 0.0 if n <= 1 else 2 * (n - 1) * alpha + 2 * (n - 1) / n * nbytes / busbw


def model_flops(mo: Model, B, recompute=True):
    l, h, s, V = mo.layers, mo.hidden, mo.seq, mo.vocab
    f = 72 * B * s * l * h * h * (1 + s / (6 * h) + V / (12 * l * h))   # fwd+bwd, no recompute
    return f * (4 / 3) if recompute else f                              # recompute adds one forward (= +1/3)


def memory_gb(mo: Model, p: Plan):
    P = mo.params / (p.tp * p.pp)
    weights_grads = 4 * P                                    # fp16 weights + fp16 grads
    opt = 12 * P / (p.dp if p.zero1 else 1)                  # fp32 master + Adam m,v
    l_stage = mo.layers / p.pp; s, h, b, a = mo.seq, mo.hidden, p.micro_b, mo.heads
    if p.recompute: act = 2 * s * b * h * l_stage + 34 * s * b * h / p.tp        # stored checkpoints + one live layer
    else: act = l_stage * (34 * s * b * h + 5 * a * s * s * b) / p.tp          # approx. per-layer activation bytes
    act *= min(p.pp, p.m)                                    # 1F1B keeps up to pp microbatches in flight
    return (weights_grads + opt + act) / 1e9


def evaluate(mo: Model, cl: Cluster, B: int, p: Plan, dp_overlap=0.0):
    N = p.dp * p.tp * p.pp
    flops = model_flops(mo, B, p.recompute)
    eff = cl.gemm_eff * mo.hidden / (mo.hidden + cl.eff_h0) if cl.eff_h0 else cl.gemm_eff
    compute = flops / (N * cl.peak_tflops * 1e12 * eff)
    bubble = compute * (p.pp - 1) / p.m
    msg = p.micro_b * mo.seq * mo.hidden * 2
    n_ar = 6 if p.recompute else 4
    tp_comm = p.m * (mo.layers / p.pp) * n_ar * ring_time(msg, p.tp, cl.intra_busbw, cl.alpha_intra)
    pp_comm = 0 if p.pp == 1 else p.m * 2 * (msg / p.tp) / cl.inter_bw_per_gpu * 1 + (p.pp - 1) * cl.alpha_inter * p.m * 2
    dp_bytes = 2 * mo.params / (p.tp * p.pp)
    dp_bw = cl.inter_bw_per_gpu if p.dp * p.tp * p.pp > cl.gpus_per_node and p.dp > 1 else cl.intra_busbw
    dp_comm = ring_time(dp_bytes, p.dp, dp_bw, cl.alpha_inter) * (1 - dp_overlap)
    p.step_s = compute + bubble + tp_comm + pp_comm + dp_comm
    p.mem_gb = memory_gb(mo, p)
    p.tflops_per_gpu = flops / p.step_s / N / 1e12
    p.breakdown = dict(compute=compute, bubble=bubble, tp=tp_comm, pp=pp_comm, dp=dp_comm)
    return p


def divisors(n): return [d for d in range(1, n + 1) if n % d == 0]


def search(mo: Model, cl: Cluster, B: int, top=5, allow_recompute=(True, False), dp_overlap=0.0, allow_zero1=(True, False)):
    out = []
    for tp in divisors(cl.gpus_per_node):
        if mo.heads % tp: continue
        for pp in divisors(mo.layers):
            if cl.gpus % (tp * pp): continue
            dp = cl.gpus // (tp * pp)
            for mb in (1, 2, 4, 8):
                if B % (dp * mb): continue
                m = B // (dp * mb)
                if m < 1: continue
                for rc in allow_recompute:
                    for z in allow_zero1:
                        p = evaluate(mo, cl, B, Plan(dp, tp, pp, mb, m, rc, z), dp_overlap)
                        if p.mem_gb <= cl.mem_gb * 0.92: out.append(p)   # leave headroom for fragmentation/workspace
    out.sort(key=lambda p: p.step_s)
    return out[:top] if top else out


def show(p: Plan, B, mo):
    b = p.breakdown
    return (f"dp={p.dp} tp={p.tp} pp={p.pp} micro_b={p.micro_b} m={p.m} recompute={p.recompute} zero1={p.zero1} | "
            f"step {p.step_s:.2f}s  {p.tflops_per_gpu:.0f} TFLOP/s/GPU  mem {p.mem_gb:.0f} GB | "
            f"compute {b['compute']:.2f} bubble {b['bubble']:.2f} tp {b['tp']:.2f} pp {b['pp']:.2f} dp {b['dp']:.2f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", type=int, default=96); ap.add_argument("--hidden", type=int, default=12288)
    ap.add_argument("--heads", type=int, default=96); ap.add_argument("--gpus", type=int, default=1536); ap.add_argument("--batch", type=int, default=2304)
    a = ap.parse_args(); mo = Model(a.layers, a.hidden, a.heads); cl = Cluster(a.gpus)
    print(f"params {mo.params/1e9:.1f}B"); [print(show(p, a.batch, mo)) for p in search(mo, cl, a.batch)]
