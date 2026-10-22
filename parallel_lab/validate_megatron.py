"""Check the planner against Table 1 of Narayanan et al., 'Efficient Large-Scale Language Model Training on GPU Clusters
Using Megatron-LM' (arXiv:2104.04473): published TFLOP/s per GPU for 10 configs, A100, seq 2048, vocab 51200, recompute on."""
import numpy as np, json
from .planner import *
# (params_B, heads, hidden, layers, tp, pp, gpus, batch, achieved_tflops_per_gpu)
T1 = [(1.7,24,2304,24,1,1,32,512,137),(3.6,32,3072,30,2,1,64,512,138),(7.5,32,4096,36,4,1,128,512,142),(18.4,48,6144,40,8,1,256,1024,135),
      (39.1,64,8192,48,8,2,512,1536,138),(76.1,80,10240,60,8,4,1024,1792,140),(145.6,96,12288,80,8,8,1536,2304,148),
      (310.1,128,16384,96,8,16,1920,2160,155),(529.6,128,20480,105,8,35,2520,2520,163),(1008.0,160,25600,128,8,64,3072,3072,163)]


def predict(row, eff, mb=1, overlap=0.0):
    pb, h_, hid, l, tp, pp, n, B, tf = row
    mo = Model(l, hid, h_); cl = Cluster(n, gemm_eff=eff); dp = n // (tp * pp); m = B // (dp * mb)
    return evaluate(mo, cl, B, Plan(dp, tp, pp, mb, m, True, False), overlap)


def main():
    cal = T1[:3]   # calibrate ONE scalar (GEMM efficiency) on the three smallest configs
    best = min(np.linspace(0.3, 1.0, 141), key=lambda e: np.mean([abs(predict(r, e).tflops_per_gpu - r[8]) / r[8] for r in cal]))
    print(f"calibrated gemm_eff={best:.3f} on first 3 rows")
    errs = []; rows = []
    for r in T1:
        p = predict(r, best); e = (p.tflops_per_gpu - r[8]) / r[8]; errs.append(e)
        # does the unconstrained search reproduce the published (tp,pp)?
        mo = Model(r[3], r[2], r[1]); top = search(mo, Cluster(r[6], gemm_eff=best), r[7], top=1, allow_recompute=(True,), allow_zero1=(False,))[0]
        rows.append(dict(params_B=r[0], published=r[8], predicted=round(p.tflops_per_gpu, 1), err_pct=round(100 * e, 1), mem_gb=round(p.mem_gb, 1), published_tp_pp=(r[4], r[5]), planner_best_tp_pp=(top.tp, top.pp)))
        print(f"{r[0]:7.1f}B  published {r[8]:3d}  predicted {p.tflops_per_gpu:6.1f}  err {100*e:+5.1f}%  mem {p.mem_gb:5.1f}GB  paper (tp,pp)=({r[4]},{r[5]})  planner best=({top.tp},{top.pp})")
    held = errs[3:]
    print(f"held-out (7 configs) MAPE {100*np.mean(np.abs(held)):.1f}%, max {100*np.max(np.abs(held)):.1f}%;  all-10 MAPE {100*np.mean(np.abs(errs)):.1f}%")
    import os; os.makedirs("results", exist_ok=True); json.dump(dict(gemm_eff=best, rows=rows, heldout_mape=float(np.mean(np.abs(held)))), open("results/megatron_validation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
