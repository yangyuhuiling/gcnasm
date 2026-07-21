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
