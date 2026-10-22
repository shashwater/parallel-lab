# parallel-lab

Real collectives over TCP sockets, real data-parallel training on top of them, and an analytical planner for
data x tensor x pipeline parallelism that is checked against published Megatron-LM numbers.

Everything runs on a CPU box (tested on 2 cores, loopback sockets). GPU hardware numbers in the planner are inputs
(A100-class defaults), not measurements.

## What is in here
| piece | what it does | how it is checked |
|---|---|---|
| `collectives.py` | all-reduce over a full TCP mesh: ring (reduce-scatter + all-gather), star (gather/broadcast at rank 0), recursive doubling | sums match a reference on world sizes 2,3,4,5,8 and odd sizes; every rank ends bit-identical |
| `bench.py` | times each algorithm, fits an alpha-beta cost model, validates it on an unseen world size | `results/bench.json` |
| `train_dp.py` | trains an MLP with N processes, averaging gradients with the ring all-reduce | final weights match single-process training (max diff ~8e-5, float32 reduction order); loss 3.46 -> 0.165 |
| `planner.py` | enumerates (dp, tp, pp, micro-batch, recompute, ZeRO-1), filters by memory, ranks by modeled step time | `validate_megatron.py` vs Table 1 of arXiv:2104.04473 |

```
python3 -m parallel_lab.bench                       # ~20 s
OPENBLAS_NUM_THREADS=1 python3 -m parallel_lab.train_dp
python3 -m parallel_lab.validate_megatron
python3 -m parallel_lab.planner --layers 96 --hidden 12288 --heads 96 --gpus 1536 --batch 2304
PYTHONPATH=. python3 tests/test_all.py
```

## Results (real runs)

**Data-parallel training on 2 CPU cores** (77k-param MLP, 60 steps, one BLAS thread per process):
1 rank 3.14 s; 2 ranks 1.74 s (**1.80x**, comm 0.12 s); 4 ranks 1.95 s (1.61x, oversubscribed: 4 processes on 2 cores).
Weights identical across ranks, within 8e-5 of single-process training.

**All-reduce algorithm choice** (16 MiB, 8 ranks on 2 cores): ring 69.6 ms, star 83.4 ms, recursive doubling 99.1 ms.
At 1 KiB, star wins (0.15 ms vs ring 4.9 ms): ring pays 2(N-1) latency steps. The crossover is why the planner prices collectives per message size.
Honest limit: loopback on 2 cores is not a network. The alpha-beta fit predicted the unseen 8-rank runs with 30-67% error
(see `results/bench.json`), because 8 processes time-slice 2 cores. I use the fit to show the method, not to claim network numbers.

**Planner vs Megatron-LM Table 1** (10 configs, 1.7B to 1T parameters, 32 to 3072 A100s). One scalar (GEMM efficiency 0.52)
calibrated on the 3 smallest rows only. Predicted TFLOP/s per GPU on the other 7 rows: **12.8% mean absolute error, 18% max**
(it under-predicts large models, since real GEMM efficiency rises with size and I hold it constant).
Its top (tp, pp) pick equals the paper's on 3 of 10 configs (76B, 145B, 1T) and uses tp=8 for the large models like the paper.
It differs on others, e.g. it prefers a smaller pipeline depth than the paper for 310B and 530B (my bubble/memory model is simpler).
See `results/megatron_validation.json`.

## How it works (read this before an interview)
**Ring all-reduce.** Split the tensor into N chunks. N-1 steps of reduce-scatter: each rank sends one chunk to its right neighbor and adds the chunk it receives from the left, so after N-1 steps rank r owns the full sum of one chunk. N-1 steps of all-gather circulate the finished chunks. Each rank sends 2(N-1)/N of the tensor total, independent of N, which is why it is bandwidth-optimal. Cost: 2(N-1)alpha + 2(N-1)/N * S/beta.
**Why recursive doubling?** log2(N) steps instead of 2(N-1), so it wins for small messages, but it sends the whole tensor each step.
**Data parallel** = replicate the model, split the batch, all-reduce gradients. Sum of per-rank gradients (each divided by the global batch) equals the full-batch gradient, so the result equals single-process training.
**Tensor parallel** splits matmuls inside a layer (needs fast links, so keep it within a node: 2 all-reduces per layer forward, 2 backward). **Pipeline parallel** splits layers into stages; the bubble costs (pp-1)/m of compute with m microbatches. **ZeRO-1** shards optimizer state across data-parallel ranks.
**Memory model:** 4 bytes/param (fp16 weights + grads) + 12 bytes/param Adam state (fp32 master, m, v), divided by tp*pp, optimizer part also by dp with ZeRO-1; activations approximated per layer (~34 s b h bytes, or checkpoints only with recompute) times pipeline microbatches in flight.

Questions you will get:
- *Why is ring all-reduce bandwidth-optimal?* Every link is busy every step and total data per rank is 2(N-1)/N * S, which approaches 2S.
- *Why did star beat ring on your box?* Loopback has no real link bottleneck and the machine has 2 cores; ring's extra steps and thread handoffs cost more than its bandwidth advantage. On a real network the ranking flips for large messages.
- *How do you know the planner is any good?* Calibrated one scalar on three configs and compared to published throughput on seven others: 12.8% MAPE. It is a screening tool, not a replacement for profiling.
- *What does the planner ignore?* Overlap of communication with compute, interleaved pipeline schedules, sequence parallelism, network topology beyond two tiers, stragglers.
- *Why do tp, dp, pp trade off?* tp: big comms but within NVLink; pp: bubble; dp: gradient all-reduce over the slower fabric and memory replication.
