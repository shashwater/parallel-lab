"""Check the planner against Table 1 of Narayanan et al., 'Efficient Large-Scale Language Model Training on GPU Clusters
Using Megatron-LM' (arXiv:2104.04473): published TFLOP/s per GPU for 10 configs, A100, seq 2048, vocab 51200, recompute on.
Protocol: fit on the 5 smallest models (<=39B), test on the 5 largest (76B..1T, up to 26x bigger). No ZeRO (the paper did not use it)."""
import numpy as np, json, os
from .planner import *
# (params_B, heads, hidden, layers, tp, pp, gpus, batch, achieved_tflops_per_gpu)
T1 = [(1.7,24,2304,24,1,1,32,512,137),(3.6,32,3072,30,2,1,64,512,138),(7.5,32,4096,36,4,1,128,512,142),(18.4,48,6144,40,8,1,256,1024,135),
      (39.1,64,8192,48,8,2,512,1536,138),(76.1,80,10240,60,8,4,1024,1792,140),(145.6,96,12288,80,8,8,1536,2304,148),
      (310.1,128,16384,96,8,16,1920,2160,155),(529.6,128,20480,105,8,35,2520,2520,163),(1008.0,160,25600,128,8,64,3072,3072,163)]


def cluster(r, a, h0): return Cluster(r[6], gemm_eff=a, eff_h0=h0)


def predict(row, a, h0, overlap, mb=1):
    pb, h_, hid, l, tp, pp, n, B, tf = row
    dp = n // (tp * pp); m = B // (dp * mb)
    return evaluate(Model(l, hid, h_), cluster(row, a, h0), B, Plan(dp, tp, pp, mb, m, True, False), overlap)


def mape(rows, a, h0, ov): return float(np.mean([abs(predict(r, a, h0, ov).tflops_per_gpu - r[8]) / r[8] for r in rows]))


def main():
    train, test = T1[:5], T1[5:]
    # baseline: one constant GEMM efficiency
    a0 = min(np.linspace(0.3, 1.0, 141), key=lambda a: mape(train, a, 0, 0.0))
    # model: size-dependent GEMM efficiency + fraction of DP all-reduce hidden behind compute (3 fitted numbers)
    grid = [(a, h0, ov) for a in np.linspace(0.45, 0.9, 46) for h0 in (0, 500, 1000, 2000, 3000, 4000, 6000, 8000) for ov in (0.0, 0.25, 0.5, 0.75)]
    a, h0, ov = min(grid, key=lambda g: mape(train, *g))
    print(f"baseline  (1 param: eff={a0:.2f}):            train MAPE {100*mape(train,a0,0,0.0):.1f}%  test MAPE {100*mape(test,a0,0,0.0):.1f}%")
    print(f"size-aware (a={a:.2f}, h0={h0}, overlap={ov}): train MAPE {100*mape(train,a,h0,ov):.1f}%  test MAPE {100*mape(test,a,h0,ov):.1f}%")
    rows = []; match = 0; gaps = []
    for r in T1:
        p = predict(r, a, h0, ov); e = (p.tflops_per_gpu - r[8]) / r[8]
        allp = search(Model(r[3], r[2], r[1]), cluster(r, a, h0), r[7], top=0, allow_recompute=(True,), allow_zero1=(False,), dp_overlap=ov)
        top = allp[0]; ok = (top.tp, top.pp) == (r[4], r[5]); match += ok
        best_by = {}
        for q in allp: best_by[(q.tp, q.pp)] = min(best_by.get((q.tp, q.pp), 1e9), q.step_s)
        gap = best_by[(r[4], r[5])] / top.step_s - 1; gaps.append(gap)
        rank = sorted(best_by.values()).index(best_by[(r[4], r[5])]) + 1
        rows.append(dict(params_B=r[0], published=r[8], predicted=round(p.tflops_per_gpu, 1), err_pct=round(100 * e, 1), mem_gb=round(p.mem_gb, 1), published_tp_pp=(r[4], r[5]), planner_best_tp_pp=(top.tp, top.pp), paper_cfg_gap_to_best_pct=round(100 * gap, 1), paper_cfg_rank=rank))
        print(f"{r[0]:7.1f}B  published {r[8]:3d}  predicted {p.tflops_per_gpu:6.1f}  err {100*e:+5.1f}%  mem {p.mem_gb:5.1f}GB  paper (tp,pp)=({r[4]},{r[5]})  planner best=({top.tp},{top.pp})  paper cfg {100*gap:.1f}% slower than planner best (rank {rank})")
    print(f"planner's top (tp,pp) equals the paper's on {match}/10 configs; the paper's config is always memory-feasible and within {100*max(gaps):.1f}% of the planner's best modeled step time")
    os.makedirs("results", exist_ok=True)
    json.dump(dict(fit_on="first 5 (<=39B)", test_on="last 5 (76B-1T)", baseline=dict(eff=float(a0), train_mape=mape(train, a0, 0, 0.0), test_mape=mape(test, a0, 0, 0.0)),
                   size_aware=dict(a=float(a), h0=h0, overlap=ov, train_mape=mape(train, a, h0, ov), test_mape=mape(test, a, h0, ov)), top_pick_matches=match, paper_cfg_max_gap_pct=100 * max(gaps), rows=rows), open("results/megatron_validation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
