# opus_a2a_gemm_lsa

Correctness-first prototype for a single-launch 8-rank A2A + GEMM fused kernel.

Target logical shape per rank:

- local input shard: `[2048, 1024]` bf16
- replicated weight: `[8192, 8192]` bf16, stored as `[N, K]`
- final output: `[2048, 8192]` bf16
- split-K partitions: 8, one per rank input shard

Kernel layout:

- workgroups `0..15`: LSA rotate put of local input shard into peer receive windows
- remaining workgroups: quad-subtile GEMM tasks over `(m_tile, n_tile, k_part)`

Build inside the MORI/ROCm container:

```bash
cd /shared/amdgpu/home/jiahao_zhou_qle/blyu/opus_a2a_gemm_lsa
make
```

Run:

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MORI_SOCKET_IFNAME=enp193s0f0np0
export LD_LIBRARY_PATH=/shared/amdgpu/home/jiahao_zhou_qle/blyu/mori/python/mori:${LD_LIBRARY_PATH:-}
mpirun --allow-run-as-root -n 8 ./build/a2a_gemm_lsa.exe
```

Useful debug flags:

```bash
./build/a2a_gemm_lsa.exe -m 256 -n 256 --k-shard 128
./build/a2a_gemm_lsa.exe --mode 2   # comm-only
```

Persistent compute scheduling is available for fused mode 0:

```bash
# Static one-WG-per-tile baseline (default).
mpirun --allow-run-as-root -n 8 ./build/a2a_gemm_lsa.exe \
  --persistent 0 --warmup 3 --iters 20

# Persistent compute workers. With --compute-wgs 0 (the default), the host uses
# min(tile_count, CU_count - comm_wgs). Override it to sweep worker counts.
mpirun --allow-run-as-root -n 8 ./build/a2a_gemm_lsa.exe \
  --persistent 1 --compute-wgs 252 --warmup 3 --iters 20
```

For the default `M=2048, N=8192, K=8192` shape there are exactly 256 output
tiles. On a 256-CU gfx950, repeated A/B tests showed no persistent scheduling
speedup: the 252-worker persistent median was about 0.839 ms versus 0.812 ms for
the static baseline. The default therefore remains `--persistent 0`.

A larger `M=6144, N=8192, K=8192` case creates 768 compute tiles, i.e. three
tiles per CU on a 256-CU gfx950:

```bash
mpirun --allow-run-as-root -n 8 ./build/a2a_gemm_lsa.exe \
  -m 6144 -n 8192 --persistent 1 --compute-wgs 248 --warmup 3 --iters 20
```

With the final no-spill scheduler, five alternating A/B runs measured a
2.215 ms median for the 768-CTA static baseline and 2.204 ms for the 248-worker
persistent variant, a small improvement of about 0.5%. Persistent scheduling is
therefore kept as an experiment rather than the default.

Increasing the tile count did not make persistent scheduling more favorable.
Three repeated runs per configuration (`--warmup 3 --iters 20`) gave these
median times for the static baseline versus 252 persistent workers:

- `M=8192`, 1024 tiles, 4 CTA/CU: 2.8839 ms vs 2.8937 ms (persistent 0.34% slower).
- `M=12288`, 1536 tiles, 6 CTA/CU: 4.2415 ms vs 4.2583 ms (persistent 0.40% slower).
- `M=16384`, 2048 tiles, 8 CTA/CU: 5.6097 ms vs 5.6379 ms (persistent 0.50% slower).
- `M=24576`, 3072 tiles, 12 CTA/CU: 8.3997 ms vs 8.4561 ms (persistent 0.67% slower).

The hardware workgroup scheduler already balances these uniform GEMM tiles
well. The persistent counter atomic and workgroup barrier repeat after every
tile, so their accumulated overhead grows slightly with larger tile counts.
