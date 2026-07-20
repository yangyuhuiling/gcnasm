"""Unfused baseline for two comm+compute kernel patterns on gfx950: GEMM+A2A
and A2A+GEMM (a matmul fused with an all_to_all). Reference for a fused kernel
to beat.
Run:      torchrun --standalone --nproc_per_node=4 bench_proj_baseline_amd.py --side all
Requires: torch==2.11.0+rocm7.2 (ROCm 7.2.x, gfx950). Numbers are stack-specific
          (bundled rocBLAS/Tensile + RCCL 2.27.7); compare on the same stack.
Reference (bf16, eager, p50 max across ranks, W=4; M = per-rank rows):
    GEMM+A2A  M=2048: GEMM 470 + copy 33 + a2a 230 = 0.76 ms  |  M=4096 = 1.51 ms
    A2A+GEMM  M=2048: a2a 188 + copy 27 + GEMM 179 = 0.44 ms  |  M=4096 = 0.82 ms
"""
from __future__ import annotations
import argparse
import os
import statistics
import sys
from typing import Callable
import torch
import torch.distributed as dist
# Kernel dims (W = world size). GEMM+A2A: matmul [M,GK] x [GN,GK]^T, then scatter
# the leading AN of GN output cols via one all_to_all. A2A+GEMM: all_to_all
# gather [M*W, OK/W] -> [M, OK], then matmul [M,OK] x [ON,OK]^T -> [M, ON].
GK, GN, AN = 8192, 18432, 10240
OK, ON = 8192, 8192
def _parse_csv_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]
def _p50_ms(fn: Callable, *, warmup: int, iters: int) -> float:
    """Eager p50 latency (ms), reduced as the max across ranks."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    p50 = statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends))
    t = torch.tensor([p50], device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())
def _bench_gemm_a2a(args, rank: int, world: int) -> None:
    n_shard = AN // world
    torch.manual_seed(0xA2A + rank)
    w = torch.randn(GN, GK, dtype=torch.bfloat16, device="cuda")
    if rank == 0:
        print(f"\n=== GEMM+A2A: K={GK} N={GN} a2a_N={AN} world={world} ===")
        print(f"{'M':>7}  {'p50 ms':>8}", flush=True)
    for m in _parse_csv_ints(args.shapes):
        x = torch.randn(m, GK, dtype=torch.bfloat16, device="cuda")
        out = torch.empty(m, GN, dtype=torch.bfloat16, device="cuda")
        src = torch.empty(world * m, n_shard, dtype=torch.bfloat16, device="cuda")
        dst = torch.empty_like(src)
        def baseline():
            torch.matmul(x, w.t(), out=out)
            cols = out[:, :AN].view(m, world, n_shard).transpose(0, 1).contiguous()
            src.copy_(cols.view(world * m, n_shard))
            dist.all_to_all_single(dst, src)
        p50 = _p50_ms(baseline, warmup=args.warmup, iters=args.iters)
        if rank == 0:
            print(f"{m:>7}  {p50:>8.3f}", flush=True)
def _bench_a2a_gemm(args, rank: int, world: int) -> None:
    k_local = OK // world
    torch.manual_seed(0xA2B + rank)
    w = torch.randn(ON, OK, dtype=torch.bfloat16, device="cuda")
    if rank == 0:
        print(f"\n=== A2A+GEMM: K={OK} N={ON} world={world} ===")
        print(f"{'M':>7}  {'p50 ms':>8}", flush=True)
    for m in _parse_csv_ints(args.shapes):
        x = torch.randn(m * world, k_local, dtype=torch.bfloat16, device="cuda")
        a2a_in = torch.empty_like(x)
        a2a_out = torch.empty_like(x)
        gathered = torch.empty(m, OK, dtype=torch.bfloat16, device="cuda")
        out = torch.empty(m, ON, dtype=torch.bfloat16, device="cuda")
        def baseline():
            a2a_in.copy_(x)
            dist.all_to_all_single(a2a_out, a2a_in)
            g = a2a_out.view(world, m, k_local).transpose(0, 1).contiguous()
            gathered.copy_(g.view(m, OK))
            torch.matmul(gathered, w.t(), out=out)
        p50 = _p50_ms(baseline, warmup=args.warmup, iters=args.iters)
        if rank == 0:
            print(f"{m:>7}  {p50:>8.3f}", flush=True)
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--side", choices=["gemm_a2a", "a2a_gemm", "all"], default="all")
    p.add_argument("--shapes", default="2048,4096")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args()
    if "LOCAL_RANK" not in os.environ:
        print("Must be launched with torchrun.", file=sys.stderr)
        return 2
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if args.side in ("gemm_a2a", "all"):
        _bench_gemm_a2a(args, rank, world)
    if args.side in ("a2a_gemm", "all"):
        _bench_a2a_gemm(args, rank, world)
    dist.barrier()
    if rank == 0:
        print("\n[done]", flush=True)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())