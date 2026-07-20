from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import defaultdict
from typing import Callable

import torch
import torch.distributed as dist

DEFAULT_GK, DEFAULT_GN, DEFAULT_AN = 8192, 18432, 10240
OK, ON = 8192, 8192


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _rank_max_ms(ms: float) -> float:
    t = torch.tensor([ms], device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _time_segment(fn: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def _bench_segments(segments: list[tuple[str, Callable[[], None]]], *, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        for _, fn in segments:
            fn()
    torch.cuda.synchronize()

    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(iters):
        for name, fn in segments:
            samples[name].append(_rank_max_ms(_time_segment(fn)))
    return {name: statistics.median(vals) for name, vals in samples.items()}


def _bench_gemm_a2a(args, rank: int, world: int) -> None:
    n_shard = args.an // world
    torch.manual_seed(0xA2A + rank)
    w = torch.randn(args.gn, args.gk, dtype=torch.bfloat16, device="cuda")
    if rank == 0:
        print(f"\n=== GEMM+A2A split: K={args.gk} N={args.gn} a2a_N={args.an} world={world} ===")
    for m in _parse_csv_ints(args.shapes):
        x = torch.randn(m, args.gk, dtype=torch.bfloat16, device="cuda")
        out = torch.empty(m, args.gn, dtype=torch.bfloat16, device="cuda")
        src = torch.empty(world * m, n_shard, dtype=torch.bfloat16, device="cuda")
        dst = torch.empty_like(src)
        cols_holder: list[torch.Tensor] = [torch.empty(world, m, n_shard, dtype=torch.bfloat16, device="cuda")]

        def matmul() -> None:
            torch.matmul(x, w.t(), out=out)

        def pack_contiguous() -> None:
            cols_holder[0] = out[:, :args.an].view(m, world, n_shard).transpose(0, 1).contiguous()

        def copy_src() -> None:
            src.copy_(cols_holder[0].view(world * m, n_shard))

        def a2a() -> None:
            dist.all_to_all_single(dst, src)

        vals = _bench_segments(
            [("matmul", matmul), ("pack_contiguous", pack_contiguous), ("copy_src", copy_src), ("all_to_all", a2a)],
            warmup=args.warmup,
            iters=args.iters,
        )
        if rank == 0:
            total = sum(vals.values())
            print(
                f"M={m}: matmul={vals['matmul']:.3f} ms, pack={vals['pack_contiguous']:.3f} ms, "
                f"copy={vals['copy_src']:.3f} ms, a2a={vals['all_to_all']:.3f} ms, sum={total:.3f} ms",
                flush=True,
            )


def _bench_a2a_gemm(args, rank: int, world: int) -> None:
    k_local = OK // world
    torch.manual_seed(0xA2B + rank)
    w = torch.randn(ON, OK, dtype=torch.bfloat16, device="cuda")
    if rank == 0:
        print(f"\n=== A2A+GEMM split: K={OK} N={ON} world={world} ===")
    for m in _parse_csv_ints(args.shapes):
        x = torch.randn(m * world, k_local, dtype=torch.bfloat16, device="cuda")
        a2a_in = torch.empty_like(x)
        a2a_out = torch.empty_like(x)
        gathered = torch.empty(m, OK, dtype=torch.bfloat16, device="cuda")
        g_holder: list[torch.Tensor] = [torch.empty(m, OK, dtype=torch.bfloat16, device="cuda")]

        def copy_in() -> None:
            a2a_in.copy_(x)

        def a2a() -> None:
            dist.all_to_all_single(a2a_out, a2a_in)

        def pack_contiguous() -> None:
            g_holder[0] = a2a_out.view(world, m, k_local).transpose(0, 1).contiguous()

        def copy_gathered() -> None:
            gathered.copy_(g_holder[0].view(m, OK))

        def matmul() -> None:
            torch.matmul(gathered, w.t(), out=torch.empty(m, ON, dtype=torch.bfloat16, device="cuda"))

        vals = _bench_segments(
            [("copy_in", copy_in), ("all_to_all", a2a), ("pack_contiguous", pack_contiguous), ("copy_gathered", copy_gathered), ("matmul", matmul)],
            warmup=args.warmup,
            iters=args.iters,
        )
        if rank == 0:
            total = sum(vals.values())
            print(
                f"M={m}: copy_in={vals['copy_in']:.3f} ms, a2a={vals['all_to_all']:.3f} ms, "
                f"pack={vals['pack_contiguous']:.3f} ms, copy_gathered={vals['copy_gathered']:.3f} ms, "
                f"matmul={vals['matmul']:.3f} ms, sum={total:.3f} ms",
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=["gemm_a2a", "a2a_gemm", "all"], default="all")
    parser.add_argument("--shapes", default="2048")
    parser.add_argument("--gk", type=int, default=DEFAULT_GK)
    parser.add_argument("--gn", type=int, default=DEFAULT_GN)
    parser.add_argument("--an", type=int, default=DEFAULT_AN)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        print("Must be launched with torchrun.", file=sys.stderr)
        return 2
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if args.side in ("gemm_a2a", "all"):
        _bench_gemm_a2a(args, rank, world)
    if args.side in ("a2a_gemm", "all"):
        _bench_a2a_gemm(args, rank, world)
    dist.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
