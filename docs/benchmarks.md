# Flash Attention Benchmark Results

## Environment

| Property         | Value                                      |
| ---------------- | ------------------------------------------ |
| GPU              | NVIDIA GeForce RTX 4070 Laptop GPU         |
| Input dtype      | FP32                                       |
| Attention mode   | Non-causal self-attention                  |
| Input layout     | `[batch, heads, sequence, head_dimension]` |
| Batch size       | 1                                          |
| Number of heads  | 8                                          |
| Custom kernel    | Serial fused online-softmax baseline       |
| Reference kernel | PyTorch scaled dot-product attention       |

The custom baseline computes attention without materializing the quadratic score or probability matrices. One CUDA thread owns one query row and serially processes all key positions and output dimensions.

## Baseline Results

| Sequence length | Head dimension | Naive online median | PyTorch SDPA median | Relative latency |
| --------------: | -------------: | ------------------: | ------------------: | ---------------: |
|             128 |             64 |           0.1760 ms |           0.0431 ms |            4.08× |
|             512 |             64 |           0.5661 ms |           0.1143 ms |            4.95× |
|            1024 |             64 |           1.1426 ms |           0.4315 ms |            2.65× |
|             512 |            128 |           1.8264 ms |           0.1943 ms |            9.40× |

## Timing Distribution

| Sequence length | Head dimension | Kernel       |       P10 |    Median |       P90 |
| --------------: | -------------: | ------------ | --------: | --------: | --------: |
|             128 |             64 | Naive online | 0.1536 ms | 0.1760 ms | 0.2016 ms |
|             128 |             64 | PyTorch SDPA | 0.0263 ms | 0.0431 ms | 0.0577 ms |
|             512 |             64 | Naive online | 0.5644 ms | 0.5661 ms | 0.6444 ms |
|             512 |             64 | PyTorch SDPA | 0.1131 ms | 0.1143 ms | 0.1227 ms |
|            1024 |             64 | Naive online | 1.1407 ms | 1.1426 ms | 1.1795 ms |
|            1024 |             64 | PyTorch SDPA | 0.4268 ms | 0.4315 ms | 0.4363 ms |
|             512 |            128 | Naive online | 1.8241 ms | 1.8264 ms | 1.8561 ms |
|             512 |            128 | PyTorch SDPA | 0.1938 ms | 0.1943 ms | 0.1957 ms |

## Initial Observations

The baseline is consistently slower than PyTorch SDPA, as expected. Its purpose is to establish a correct fused online-softmax implementation against which subsequent CUDA kernels can be measured.

The head-dimension-128 workload shows the largest performance deficit. Each thread owns 128 FP32 output accumulators in addition to dot-product and softmax state, creating substantially greater register demand than the head-dimension-64 kernel. Compiler resource reports and Nsight Compute profiling are required to determine whether this results in register spilling or reduced occupancy.

The relative gap decreases at sequence length 1024 with head dimension 64. One likely explanation is that the larger workload exposes more parallel query rows and amortizes fixed launch overhead. This remains a hypothesis until profiler data is collected.

The first optimized kernel should replace the current ownership model:

```text
One thread owns one query row and all output dimensions
```

with:

```text
One thread block owns one query row
Threads cooperate on dot products and divide output dimensions
```

This should:

* expose substantially more parallelism within each query row;
* reduce the number of output accumulators held by each thread;
* enable warp-level reduction of query-key dot products;
* prepare the implementation for shared-memory key/value tiling.

## Reproduction Commands

```bash
python -m benchmarks.benchmark_forward \
    --batch 1 --heads 8 --query-length 128 --head-dim 64

python -m benchmarks.benchmark_forward \
    --batch 1 --heads 8 --query-length 512 --head-dim 64

python -m benchmarks.benchmark_forward \
    --batch 1 --heads 8 --query-length 1024 --head-dim 64

python -m benchmarks.benchmark_forward \
    --batch 1 --heads 8 --query-length 512 --head-dim 128
```

## Benchmarking Requirements

Future benchmark entries should record:

* Git commit hash;
* PyTorch and CUDA versions;
* GPU power mode;
* benchmark configuration;
* compiler register and spill report;
* kernel version;
* causal or non-causal mode;
* input dtype;
* warmup, sample, and iteration counts.




## Experiment 1: Block-Per-Query-Row Decomposition

### Hypothesis

The serial online-softmax baseline assigns one CUDA thread to each query row. That thread computes all query-key dot products and retains the entire output vector in thread-local state.

The first optimization changed ownership to:

```text
One CUDA block owns one query row
One thread owns one output dimension
Threads cooperate on each query-key dot product
```

The expected benefits were:

* lower per-thread accumulator demand;
* more parallelism across the head dimension;
* reduced register pressure for head dimensions 64 and 128;
* warp-shuffle reductions for query-key dot products.

### Results

| Sequence length | Head dimension | Serial online | Block online | Block vs. serial | Block vs. SDPA |
| --------------: | -------------: | ------------: | -----------: | ---------------: | -------------: |
|             128 |             64 |     0.1543 ms |    0.1662 ms |     1.08× slower |          6.39× |
|             512 |             64 |     0.5662 ms |    1.9062 ms |     3.37× slower |         16.72× |
|            1024 |             64 |     1.1454 ms |    7.2672 ms |     6.35× slower |         17.16× |
|             512 |            128 |     1.8302 ms |    3.4058 ms |     1.86× slower |         17.42× |

### Interpretation

The block-per-query-row decomposition exposed more threads but did not improve the underlying memory-reuse pattern. Every query row continued to load the complete key and value matrices independently.

For every key position, the kernel performed:

1. A cooperative dot-product reduction.
2. A block-wide synchronization after partial reductions.
3. A scalar online-softmax update by thread zero.
4. A block-wide synchronization before output accumulation.
5. A per-thread output update.
6. A block-wide synchronization before processing the next key.

This introduced approximately three block-wide barriers per key position. The synchronization cost increased linearly with sequence length:

| Sequence length | Approximate barriers per query row |
| --------------: | ---------------------------------: |
|             128 |                                384 |
|             512 |                              1,536 |
|            1024 |                              3,072 |

The kernel introduced synchronization overhead without creating reuse of K or V across multiple query rows. This explains why the regression became substantially larger as sequence length increased.

### Decision

The block-per-query-row design is rejected as an optimization path. It remains in the benchmark suite as a documented negative result.

The next implementation uses a tiled decomposition:

```text
One block owns multiple query rows
K and V tiles are loaded once into shared memory
All query rows in the block reuse each K/V tile
Dot-product reductions remain warp-local
Online-softmax state remains register-resident
```

Initial specialization:

| Parameter                |   Value |
| ------------------------ | ------: |
| Query tile, `Br`         | 16 rows |
| Key/value tile, `Bc`     | 32 rows |
| Head dimension           |      64 |
| Threads per block        |     128 |
| Warps per block          |       4 |
| Query rows per warp      |       4 |
| Shared K storage         |    8 KB |
| Shared V storage         |    8 KB |
| Total K/V shared storage |   16 KB |

This decomposition is the first implementation expected to obtain the defining IO-reuse advantage of FlashAttention.


## Experiment 3: Tilewise Online Softmax and Warp-Local Synchronization

### Objective

Replace the key-at-a-time online-softmax recurrence with a tilewise recurrence matching the structure of the FlashAttention-2 forward algorithm.

For each `Br=16`, `Bc=32` tile, the new kernel:

1. Loads a `32 × 64` K tile and V tile into shared memory.
2. Computes a complete `16 × 32` score tile.
3. Computes the row maximum and exponential sum across the complete key tile.
4. Computes the local `P_tile @ V_tile` contribution.
5. Merges the complete tile into the running online-softmax state once.
6. Assigns independent query and output rows to each warp.

The initial implementation used a block-wide barrier between producing and consuming each warp's probability rows:

```cpp
__syncthreads();
```

However, each warp writes and reads only its own probability slice:

```cpp
probability_shared[warp_id][row][key_column]
```

There is no cross-warp dependency at this point. The block-wide barrier was therefore replaced with:

```cpp
__syncwarp(kFullWarpMask);
```

A block-wide barrier remains at the end of each K/V tile because all warps must finish reading shared K and V before the next tile overwrites them.

### Benchmark Configuration

* GPU: NVIDIA GeForce RTX 4070 Laptop GPU
* Batch size: 1
* Attention heads: 8
* Head dimension: 64
* Data type: FP32
* Causal masking: disabled
* Query and key sequence lengths: equal
* Reported value: median CUDA-event latency

### Results

| Sequence length | Naive online | Block online | Tiled SIMT | Tilewise SIMT | PyTorch SDPA |
| --------------: | -----------: | -----------: | ---------: | ------------: | -----------: |
|             128 |    0.1924 ms |    0.1658 ms |  0.0793 ms | **0.0352 ms** |    0.0373 ms |
|             512 |    0.5654 ms |    1.9059 ms |  0.6546 ms | **0.3981 ms** |    0.1137 ms |
|            1024 |    1.1414 ms |    7.3766 ms |  2.1156 ms | **1.4550 ms** |    0.4318 ms |

### Relative Performance

| Sequence length | Tilewise vs. tiled | Tilewise vs. naive | Tilewise latency relative to SDPA |
| --------------: | -----------------: | -----------------: | --------------------------------: |
|             128 |   **2.25× faster** |   **5.47× faster** |                         **0.94×** |
|             512 |   **1.64× faster** |   **1.42× faster** |                             3.50× |
|            1024 |   **1.45× faster** |       1.27× slower |                             3.37× |

### Analysis

The tilewise kernel initially performed nearly identically to the earlier tiled kernel. Inspection of the synchronization structure revealed that the kernel performed an unnecessary block-wide synchronization after every probability tile.

Replacing that barrier with warp-level synchronization reduced tilewise latency by approximately:

* 2.26× at sequence length 128
* 1.64× at sequence length 512
* 1.46× at sequence length 1024

This demonstrates the importance of FlashAttention-2-style warp ownership. Because each warp owns independent query and output rows, probability consumption and output accumulation can proceed without cross-warp communication.

At sequence length 128, the tilewise FP32 kernel matched or slightly outperformed PyTorch SDPA for this specific workload. This result is limited to the tested configuration and should not be interpreted as a general performance advantage over SDPA.

At sequence lengths 512 and 1024, the remaining gap to SDPA is stable at approximately 3.4–3.5×. The tilewise kernel now scales approximately quadratically with sequence length without pathological synchronization overhead:

```text
Tilewise SIMT:
0.3981 ms → 1.4550 ms = 3.65× increase

PyTorch SDPA:
0.1137 ms → 0.4318 ms = 3.80× increase
```

Doubling sequence length produces approximately four times as much attention work, so this scaling is expected.

The remaining bottleneck is the arithmetic implementation. Both matrix products are still implemented using scalar FP32 SIMT instructions:

```text
QKᵀ:
- scalar FP32 fused multiply-add operations
- repeated warp-shuffle broadcasts

PV:
- scalar FP32 fused multiply-add operations
- shared-memory probability and value reads
```

The tilewise organization reduces synchronization and non-matmul overhead, but it does not accelerate the dominant matrix-multiplication work.

### Conclusion

The experiment validates the following design decisions:

* Parallelizing over query-row tiles provides sequence-level thread-block parallelism.
* Splitting query rows across warps avoids cross-warp output reductions.
* K and V tiles can be reused across multiple query rows through shared memory.
* Online softmax should merge complete K/V tiles rather than individual keys.
* Warp-local data dependencies should use warp synchronization rather than block-wide barriers.

The FP32 tilewise kernel now provides the algorithmic and work-partitioning foundation required for Tensor Core acceleration.

### Next Optimization

Implement an FP16 Tensor Core forward path:

```text
FP16 Q, K, V
    ↓
Tensor Core QKᵀ
    ↓
FP32 tilewise softmax
    ↓
Tensor Core PV
    ↓
FP32 accumulation
    ↓
FP16 output
```

The existing tilewise kernel remains as the FP32 SIMT baseline.

## Experiment 4: Fused FP16 Tensor Core Attention Baseline

### Objective

Replace the scalar FP32 matrix arithmetic in the tilewise attention kernel with FP16 Tensor Core operations while preserving FP32 softmax statistics and FP32 output accumulation.

The goal of this experiment was to establish a correct fused Tensor Core baseline before performing Ada-specific memory, synchronization, and pipeline optimizations.

The kernel computes the complete attention forward pass without materializing the global score or probability matrices:

```text
FP16 Q, K, V
    ↓
Tensor Core QKᵀ
    ↓
FP32 tilewise online softmax
    ↓
Tensor Core PV
    ↓
FP32 running output accumulation
    ↓
FP16 output
```

### Kernel Configuration

* Query tile size, `Br`: 64 rows
* Key/value tile size, `Bc`: 16 rows
* Head dimension: 64
* Warps per block: 4
* Threads per block: 128
* Query rows per warp: 16
* Input data type: FP16
* Tensor Core accumulator type: FP32
* Softmax data type: FP32
* Output data type: FP16
* Causal masking: disabled
* Sequence length requirement: divisible by 16

Each warp owns an independent `16 × 64` query and output slice:

```text
Warp 0 → query rows  0–15
Warp 1 → query rows 16–31
Warp 2 → query rows 32–47
Warp 3 → query rows 48–63
```

All four warps reuse the same shared-memory K and V tile.

### Tensor Core Computation

For each `Bc=16` K/V tile, every warp computes:

```text
QKᵀ:
[16 × 64] × [64 × 16]
    →
[16 × 16] FP32 score tile
```

The head dimension is processed through four `16 × 16 × 16` WMMA accumulation steps.

After tilewise FP32 softmax, the warp computes:

```text
PV:
[16 × 16] × [16 × 64]
    →
[16 × 64] FP32 output contribution
```

The 64 output dimensions are divided into four `16 × 16` WMMA output tiles.

### Correctness-First Data Flow

The initial implementation deliberately uses shared memory between major stages:

```text
WMMA score accumulator
    ↓
FP32 score_shared
    ↓
FP32 softmax
    ↓
FP16 probability_shared
    ↓
WMMA PV accumulator
    ↓
FP32 pv_shared
    ↓
persistent FP32 output registers
```

This structure avoids depending on WMMA’s opaque fragment-element ownership and provides a clear correctness baseline. It also introduces shared-memory traffic that later optimizations should remove.

### Tensor Core Verification

The standalone WMMA experiments established that:

```text
FP16 [16 × 64]
×
FP16 [64 × 16]
→
FP32 [16 × 16]
```

produced correct results through four WMMA reduction steps.

Inspection of the generated SASS confirmed that NVCC emitted native Tensor Core instructions:

```text
HMMA.16816.F32
```

The fused attention kernel uses the same WMMA execution path for both `QKᵀ` and `PV`.

### Benchmark Configuration

* GPU: NVIDIA GeForce RTX 4070 Laptop GPU
* Batch size: 1
* Attention heads: 8
* Head dimension: 64
* Data type: FP16
* Causal masking: disabled
* Query and key sequence lengths: equal
* Reported value: median CUDA-event latency
* Baseline: PyTorch scaled-dot-product attention

### Results

| Sequence length | Tensor Core attention | PyTorch SDPA | Relative latency |
| --------------: | --------------------: | -----------: | ---------------: |
|             128 |             0.0238 ms |    0.0231 ms |            1.03× |
|             512 |             0.1156 ms |    0.0280 ms |            4.13× |
|            1024 |             0.3479 ms |    0.0807 ms |            4.31× |

### Latency Distribution

| Sequence length | Kernel                |       p10 |    Median |       p90 |
| --------------: | --------------------- | --------: | --------: | --------: |
|             128 | Tensor Core attention | 0.0232 ms | 0.0238 ms | 0.0293 ms |
|             128 | PyTorch SDPA          | 0.0184 ms | 0.0231 ms | 0.0343 ms |
|             512 | Tensor Core attention | 0.0912 ms | 0.1156 ms | 0.1186 ms |
|             512 | PyTorch SDPA          | 0.0266 ms | 0.0280 ms | 0.0346 ms |
|            1024 | Tensor Core attention | 0.3471 ms | 0.3479 ms | 0.3536 ms |
|            1024 | PyTorch SDPA          | 0.0803 ms | 0.0807 ms | 0.0831 ms |

### Compiler Resource Usage

The Ada `sm_89` compiler report for the fused kernel was:

```text
0-byte stack frame
0-byte spill stores
0-byte spill loads
168 registers per thread
1 hardware barrier resource
35,328 bytes static shared memory per block
```

With 128 threads per block, the register allocation is approximately:

```text
168 registers/thread × 128 threads
= 21,504 registers/block
```

The kernel retains its persistent output values and WMMA state in registers without spilling into local memory.

The static shared-memory allocation consists approximately of:

| Allocation                               |             Size |
| ---------------------------------------- | ---------------: |
| Q tile, `64 × 64 × FP16`                 |      8,192 bytes |
| K tile, `16 × 64 × FP16`                 |      2,048 bytes |
| V tile, `16 × 64 × FP16`                 |      2,048 bytes |
| Score tiles, `4 × 16 × 16 × FP32`        |      4,096 bytes |
| Probability tiles, `4 × 16 × 16 × FP16`  |      2,048 bytes |
| Temporary PV tiles, `4 × 16 × 64 × FP32` |     16,384 bytes |
| Row metadata                             |        512 bytes |
| **Total**                                | **35,328 bytes** |

### Analysis

At sequence length 128, the custom kernel was within approximately 3% of PyTorch SDPA. The latency distributions overlap, so this should be interpreted as matching SDPA for the tested small workload rather than establishing a general performance advantage.

At sequence lengths 512 and 1024, the kernel remained approximately 4.1–4.3× slower than SDPA.

The relatively stable long-sequence gap indicates that the kernel does not suffer from pathological scaling. Instead, it performs excessive overhead during every `Bc=16` K/V iteration.

For sequence length 1024, the kernel performs:

```text
1024 / 16 = 64 K/V tile iterations
```

Every iteration includes:

1. Cooperative K/V staging.
2. A block-wide synchronization.
3. Tensor Core `QKᵀ`.
4. A score-fragment store to shared memory.
5. FP32 softmax reads and FP16 probability writes.
6. Tensor Core `PV`.
7. A PV-fragment store to shared memory.
8. A PV reload into persistent FP32 output registers.
9. An online-softmax output-rescaling pass.
10. A final block-wide synchronization.

The Tensor Core arithmetic is therefore surrounded by substantial shared-memory traffic, synchronization, conversion, and output-rescaling work.

### Key Findings

The experiment establishes that:

* The fused FP16 Tensor Core attention algorithm is numerically correct.
* Both attention GEMM phases execute through native Tensor Core instructions.
* The kernel maintains FP32 softmax and output state without register spilling.
* The current implementation matches SDPA for the tested small workload.
* The long-sequence performance gap is caused by per-tile overhead rather than incorrect asymptotic scaling.
* Shared-memory traffic and synchronization around the Tensor Core operations are now the primary optimization targets.

### Conclusion

The fused WMMA kernel provides the first complete Tensor Core FlashAttention-style baseline for the project.

Its current structure prioritizes correctness and debuggability over minimal data movement. The kernel successfully maps both `QKᵀ` and `PV` onto Tensor Cores, but performs too many shared-memory round trips and online-softmax merges because the key/value tile contains only 16 rows.

The next optimization should reduce the frequency of these operations without changing the amount of useful matrix arithmetic.

### Next Optimization: Increase `Bc` from 16 to 32

The next kernel specialization will use:

```text
Br = 64
Bc = 32
4 warps per block
16 query rows per warp
```

Each warp will compute two score fragments:

```text
Q [16 × 64] × K₀ᵀ [64 × 16]
    →
S₀ [16 × 16]

Q [16 × 64] × K₁ᵀ [64 × 16]
    →
S₁ [16 × 16]
```

Softmax will operate across the combined 32 scores.

The `PV` computation will accumulate across two reduction fragments:

```text
P[:, 0:16] × V[0:16, :]
+
P[:, 16:32] × V[16:32, :]
```

At sequence length 1024, this reduces the number of K/V iterations from:

```text
Bc=16 → 64 iterations
Bc=32 → 32 iterations
```

The change should approximately halve:

* online-softmax merges;
* output-rescaling passes;
* end-of-tile block synchronizations;
* loop-control overhead.

The estimated shared-memory requirement is approximately 45,568 bytes per block, remaining below the static shared-memory limit and preserving the current expected block residency.



## Experiment: Shared-Memory Bottleneck Diagnosis — Tensor Core FlashAttention (Bc=16/32, D=64)

**Date:** 2026-08-03
**Kernel:** `tensorcore_attention_forward_kernel_d64`, sm_89 (RTX 4070)
**Config:** Br=64, D=64, WMMA 16×16×16, 4 warps/block, Bc∈{16,32}

### Hypothesis going in
Suspected the kernel would be compute-bound (tensor core MMA-limited) or possibly DRAM-bound given the O(N²) attention memory access pattern.

### Method
Profiled with `ncu` on the Bc=32 variant, full SOL + memory workload + warp state + occupancy sections.

### Key results

| Metric | Value | Interpretation |
|---|---|---|
| DRAM Throughput | 3.07% | Global bandwidth is a non-issue |
| L2 Hit Rate | 78.26% | Working set fits in cache |
| L1/TEX Cache Throughput | 83.39% | **Saturated** — shared mem + global loads share this pipe on sm_89 |
| Compute (SM) Throughput | 21.53% | Tensor cores are idle most of the time |
| Short-scoreboard stalls | 43% of 11.58 cycles/instr | Dominant stall reason — `ncu` attributes this directly to shared memory ops |
| Block Limit Shared Mem | 2 blocks/SM | Binding occupancy constraint (vs. 4 for registers, 12 for warps) |
| Theoretical Occupancy | 16.67% | |
| Achieved Occupancy | 16.45% | Tracks theoretical almost exactly — no scheduling/tail loss on top |

### Conclusion
**The kernel is shared-memory/L1-pipe bound, not compute-bound and not DRAM-bound.** Two independent lines of evidence converge on the same root cause:

1. **Occupancy** — static shared memory usage (~44KB for Bc=32: query 8KB + key 4KB + value 4KB + score 8KB + probability 4KB + pv 16KB) caps concurrent blocks at 2/SM, well below what registers or warp slots would allow.
2. **Stall cycles** — the WMMA API forces a store→shared→load round trip at every fragment boundary (`score_shared → probability_shared → pv_shared → output_accumulator`), since fragment-to-thread mapping is opaque under `wmma::`. Combined with the serial, half-warp-utilized online-softmax reduction (lanes 0–15 only, columns looped one at a time), this generates enough LDS/STS traffic to saturate the L1 pipe independent of any actual data movement need.

These aren't two separate problems — they're the same buffers (`pv_shared` being the largest single offender at 16KB) driving both the occupancy ceiling and the stall-cycle count. Shrinking shared memory footprint should yield a compounding win rather than an additive one.

### Next experiment
Rewrite PV accumulation and softmax reduction using raw `mma.sync.aligned.m16n8k16` PTX to get explicit thread-to-register fragment mapping, eliminating the `pv_shared` round trip entirely and replacing the serial softmax loop with warp-shuffle reduction across all 32 lanes. Target: raise `Block Limit Shared Mem` above 2, reduce short-scoreboard stall fraction below current 43%, re-profile and compare Compute/Memory SOL balance.


## Experiment 6: Eliminate `pv_shared` with Explicit `ldmatrix` and `mma.sync`

### Objective

Remove the intermediate shared-memory materialization between the Tensor Core `PV` computation and the persistent FP32 output accumulator.

The previous `Bc=32` kernel used WMMA for both matrix multiplications:

```text
QKᵀ:
WMMA accumulator
→ score_shared
→ online softmax

PV:
WMMA accumulator
→ pv_shared
→ persistent FP32 output registers
```

Nsight Compute identified the kernel as limited by the L1/shared-memory execution path rather than DRAM bandwidth or Tensor Core throughput. The kernel also used enough shared memory to restrict residency to two blocks per SM.

The objective of this experiment was to replace only the `PV` stage with explicit warp-level Tensor Core instructions while preserving the existing:

* WMMA `QKᵀ` implementation
* FP32 tilewise online softmax
* `Bc=32` tile shape
* Four-warps-per-block organization
* FP32 persistent output accumulation

This isolates the effect of eliminating `pv_shared`.

---

### Baseline

The baseline was the fused `Bc=32` WMMA kernel:

```text
Br = 64
Bc = 32
D  = 64
4 warps per block
128 threads per block
```

Each warp owns 16 query rows and computes:

```text
P [16 × 32] × V [32 × 64]
    →
PV [16 × 64]
```

The WMMA implementation divided the output into four `16 × 16` fragments. Each fragment was accumulated in FP32, stored to shared memory, and then reloaded into ordinary FP32 registers:

```text
WMMA PV accumulator
→ FP32 pv_shared
→ warp synchronization
→ FP32 output_accumulator
```

The intermediate buffer was:

```cpp
float pv_shared[4][16][64];
```

Its size was:

```text
4 warps × 16 rows × 64 dimensions × 4 bytes
= 16,384 bytes
```

---

### Profiling Motivation

Nsight Compute reported the following for the `Bc=32` WMMA baseline at `N=1024`:

| Metric                       |  Value |
| ---------------------------- | -----: |
| DRAM throughput              |  3.07% |
| L1/TEX throughput            | 83.39% |
| Compute throughput           | 21.53% |
| Short-scoreboard stall share |  43.0% |
| Theoretical occupancy        | 16.67% |
| Achieved occupancy           | 16.45% |
| Active warps per SM          |   7.90 |
| Shared-memory block limit    |      2 |

The low DRAM utilization showed that external memory bandwidth was not the limiting resource.

The high L1/TEX utilization and short-scoreboard stall share indicated substantial pressure from shared-memory and MIO dependencies. Shared memory was also the binding occupancy constraint:

```text
2 blocks/SM × 4 warps/block
= 8 resident warps/SM
```

The `pv_shared` buffer was the largest removable shared-memory allocation in the kernel.

---

### Implementation

The new specialization retains WMMA for `QKᵀ`, but replaces the `PV` stage with explicit PTX-level Tensor Core operations:

```text
WMMA QKᵀ
→ score_shared
→ FP32 online softmax
→ probability_shared
→ ldmatrix
→ mma.sync
→ direct FP32 output-register merge
```

The new exported specialization is:

```text
tensorcore_attention_forward_bc32_raw_pv
```

#### Raw MMA Shape

The raw instruction uses:

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

Each instruction computes:

```text
A [16 × 16] × B [16 × 8]
    →
D [16 × 8]
```

The complete `PV` operation requires:

```text
64 output columns / 8 columns per MMA = 8 output subtiles

32 reduction elements / 16 per MMA = 2 reduction steps

8 output subtiles × 2 reduction steps
= 16 mma.sync instructions per warp per K/V tile
```

For each output subtile:

```text
P[:,  0:16] × V[ 0:16, output columns]
+
P[:, 16:32] × V[16:32, output columns]
```

is accumulated into four FP32 registers per lane.

#### Operand Loading

The probability operand is loaded from shared memory with:

```text
ldmatrix.sync.aligned.m8n8.x4.shared.b16
```

This collectively loads the `16 × 16` row-major probability fragment.

The value operand is loaded with:

```text
ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16
```

This constructs the register representation required by the column-major B operand of the `row.col` MMA instruction from the physically row-major V tile.

#### Persistent Register Ownership

Explicit `mma.sync` exposes the logical output coordinates owned by every accumulator register.

For each lane:

```text
group_id        = lane_id / 4
thread_in_group = lane_id % 4
```

the four FP32 accumulator registers correspond to two elements from row `group_id` and two elements from row `group_id + 8`.

The persistent output layout was changed to follow this same ownership:

```text
8 output subtiles × 4 FP32 registers per lane
= 32 persistent output values per lane
```

This allows the Tensor Core result to be merged directly into the long-lived online-softmax numerator:

```text
mma.sync accumulator registers
→ apply previous online-softmax scale
→ persistent output_accumulator registers
```

No shared-memory store, synchronization, reload, or lane redistribution is required.

---

### Correctness

The raw-PV specialization was compared against PyTorch scaled-dot-product attention for:

| Batch | Heads | Sequence length | Head dimension |
| ----: | ----: | --------------: | -------------: |
|     1 |     1 |              32 |             64 |
|     1 |     2 |              64 |             64 |
|     1 |     4 |              96 |             64 |
|     1 |     8 |             128 |             64 |
|     1 |     8 |             512 |             64 |

All tests passed with:

```text
rtol = 3e-2
atol = 3e-2
```

The passing tests validate:

* The `ldmatrix.x4` probability layout
* The `ldmatrix.x2.trans` value layout
* The raw MMA operand ordering
* The two-step `K=32` reduction
* Accumulator-register coordinate ownership
* Direct integration with the online-softmax recurrence
* Final normalization and FP16 output storage

---

### Compiler Resource Usage

#### `Bc=32` WMMA PV baseline

```text
128 registers per thread
45,568 bytes shared memory per block
0-byte stack frame
0 spill stores
0 spill loads
```

#### `Bc=32` raw MMA PV

```text
79 registers per thread
29,184 bytes shared memory per block
0-byte stack frame
0 spill stores
0 spill loads
```

#### Resource Reduction

| Resource                |  WMMA PV | Raw MMA PV |     Change |
| ----------------------- | -------: | ---------: | ---------: |
| Registers per thread    |      128 |         79 | **−38.3%** |
| Shared memory per block | 45,568 B |   29,184 B | **−36.0%** |
| Spill stores            |        0 |          0 |  Unchanged |
| Spill loads             |        0 |          0 |  Unchanged |

The shared-memory reduction was exactly:

```text
45,568 − 29,184
= 16,384 bytes
```

This matches the removed `pv_shared` allocation.

The register reduction was an additional benefit. Aligning Tensor Core accumulator ownership with persistent output ownership shortened live ranges and removed the WMMA store/reload path.

---

### Benchmark Results

Benchmark configuration:

```text
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
Batch: 1
Heads: 8
Head dimension: 64
Input/output dtype: FP16
Softmax/output accumulation: FP32
Causal masking: disabled
```

#### Median Latency

| Sequence length | `Bc=32` WMMA PV | Raw MMA PV | Raw-PV improvement |
| --------------: | --------------: | ---------: | -----------------: |
|             128 |       0.0246 ms |  0.0252 ms |              −2.4% |
|             512 |       0.0887 ms |  0.0841 ms |           **5.2%** |
|            1024 |       0.3375 ms |  0.3156 ms |           **6.5%** |

The difference at `N=128` is within the small-workload timing variation and should not be interpreted as a meaningful regression.

At `N=512` and `N=1024`, the raw-PV implementation produced consistent latency reductions.

#### Raw-PV Latency Distribution

| Sequence length |       p10 |    Median |       p90 |
| --------------: | --------: | --------: | --------: |
|             128 | 0.0210 ms | 0.0252 ms | 0.0327 ms |
|             512 | 0.0834 ms | 0.0841 ms | 0.0882 ms |
|            1024 | 0.3148 ms | 0.3156 ms | 0.3271 ms |

---

### Nsight Compute Comparison

The raw-PV specialization was profiled under the same `N=1024` workload as the WMMA baseline.

| Metric                       |   WMMA PV | Raw MMA PV |             Change |
| ---------------------------- | --------: | ---------: | -----------------: |
| Profiled duration            | 546.43 µs |  506.40 µs |          **−7.3%** |
| Theoretical occupancy        |    16.67% |     25.00% |           +8.33 pp |
| Achieved occupancy           |    16.45% |     21.05% |           +4.60 pp |
| Active warps per SM          |      7.90 |      10.10 |         **+27.8%** |
| Shared-memory block limit    |         2 |          3 |           +1 block |
| Register block limit         |         4 |          6 |          +2 blocks |
| Short-scoreboard stall share |     43.0% |     37.64% |        Lower share |
| L1/TEX throughput            |    83.39% |     88.60% | Higher utilization |
| DRAM throughput              |     3.07% |      5.02% |          Still low |
| Compute throughput           |    21.53% |     16.09% |              Lower |

The profiler duration is instrumentation-dependent and should not replace normal benchmark latency. Its relative improvement, however, closely matches the ordinary benchmark result.

---

### Analysis

#### Occupancy Improvement

Removing `pv_shared` changed the shared-memory residency limit from two blocks to three blocks per SM:

```text
Before:
2 blocks × 4 warps = 8 theoretical active warps/SM
8 / 48 = 16.67% occupancy

After:
3 blocks × 4 warps = 12 theoretical active warps/SM
12 / 48 = 25.00% occupancy
```

Achieved occupancy rose from 16.45% to 21.05%, while active warps per SM increased by approximately 28%.

The raw kernel did not reach the full 25% theoretical occupancy throughout execution, but it gained meaningful additional scheduling capacity.

#### Remaining Memory-Pipeline Pressure

Although one large shared-memory round trip was eliminated, L1/TEX throughput increased from 83.39% to 88.60%.

This is consistent with the higher number of resident warps issuing against the remaining shared-memory structures:

```text
query_shared
key_shared
value_shared
score_shared
probability_shared
```

The optimization shortened execution and increased concurrency, but the remaining shared-memory path became more heavily saturated.

#### Short-Scoreboard Stalls

The fraction of issue latency attributed to short-scoreboard dependencies decreased:

```text
43.0% → 37.64%
```

However, cycles between issued instructions increased, and the reported short-scoreboard component rose from approximately 5.0 to 6.7 cycles.

Therefore, the result should not be described as eliminating the shared-memory bottleneck. Instead:

> The `pv_shared` dependency was removed successfully, but the increased occupancy exposed the remaining score and probability shared-memory operations as the dominant bottleneck.

#### Compute Throughput

Compute utilization decreased despite the lower total latency.

This does not indicate that the Tensor Core conversion failed. The kernel remains constrained by shared-memory and MIO dependencies, so the additional active warps do not translate into proportionally higher Tensor Core utilization.

---

### Key Findings

This experiment demonstrated that explicit warp-level MMA ownership can remove an otherwise necessary WMMA shared-memory round trip.

The raw-PV implementation:

* Eliminated the 16 KiB `pv_shared` buffer
* Removed FP32 PV stores and reloads
* Reduced shared memory by 36%
* Reduced registers per thread by 38%
* Retained zero register spilling
* Raised the shared-memory residency limit from two to three blocks per SM
* Increased achieved active warps per SM by approximately 28%
* Improved `N=512` latency by 5.2%
* Improved `N=1024` latency by 6.5%

The experiment also confirmed that the kernel remains limited by the shared-memory/L1 execution path rather than DRAM bandwidth or Tensor Core arithmetic throughput.

---

### Conclusion

Replacing WMMA PV materialization with explicit:

```text
ldmatrix
+
mma.sync.m16n8k16
+
direct accumulator ownership
```

produced both a structural and measurable performance improvement.

The primary benefit was not merely replacing one Tensor Core API with another. The explicit register mapping made it possible to preserve the MMA accumulator ownership through the online-softmax output update, avoiding a 16 KiB shared-memory buffer entirely.

The improvement was largest for longer sequences, where the eliminated store/reload path was repeated across many K/V tiles.

The remaining bottleneck is now the score path:

```text
WMMA QKᵀ accumulator
→ score_shared
→ scalar shared-memory softmax reads
```

---

### Next Experiment

The next specialization will convert `QKᵀ` from WMMA to explicit `mma.sync` and perform the softmax max and sum reductions directly from known accumulator-register ownership.

Target flow:

```text
raw QKᵀ accumulator registers
→ warp-level row maximum
→ exponentiation
→ warp-level row sum
→ FP16 probability_shared
→ raw MMA PV
```

The immediate target is to eliminate:

```cpp
float score_shared[4][16][32];
```

This buffer currently consumes:

```text
4 × 16 × 32 × 4 bytes
= 8,192 bytes
```

Removing it would reduce shared memory from approximately:

```text
29,184 bytes
```

to:

```text
20,992 bytes
```

and should reduce shared-memory instruction volume and short-scoreboard dependencies. It may also raise the shared-memory residency limit from three to four blocks per SM, subject to the register count of the raw-QK specialization.



# Experiment 7A: Raw QKᵀ + Register/Shuffle Softmax

## Goal

Remove the shared-memory score materialization path:

```text
WMMA QKᵀ
→ score_shared
→ scalar shared-memory softmax
```

and replace it with:

```text
raw ldmatrix + mma.sync QKᵀ
→ FP32 score accumulators
→ shuffle-based row max/sum
→ probability_shared
→ existing raw PV
```

The experiment intentionally retained `probability_shared` so that the already-correct raw-PV implementation remained unchanged.

---

## Implementation

For each warp:

```text
Q [16 × 64]
×
Kᵀ [64 × 32]
→
S [16 × 32]
```

using:

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

The score tile is divided into four `16×8` output subtiles and four `K=16` reduction steps:

```text
4 score subtiles
×
4 reduction steps
=
16 mma.sync instructions
per warp per K/V tile
```

Q is loaded once per `K=16` slice and reused across all four score subtiles.

Physical row-major K tiles are converted into the required logical `Kᵀ` operand using `ldmatrix.x2` with two adjacent `8×8` panels.

After QK accumulation, scores remain in FP32 registers.

Each four-lane subgroup owns two complete logical rows:

```text
upper row = group_id
lower row = group_id + 8
```

Row maxima and probability sums are reduced using width-four warp shuffles:

```cpp
__shfl_xor_sync(..., 1, 4);
__shfl_xor_sync(..., 2, 4);
```

The resulting statistics are exchanged with the canonical online-softmax state held in lanes 0–15.

Probabilities are computed in registers and written directly to:

```cpp
probability_shared[4][16][32]
```

for consumption by the existing raw-PV path.

The raw-QK specialization therefore eliminates `score_shared` entirely.

---

## Compiler Resources

Previous raw-PV specialization:

```text
79 registers/thread
29,184 bytes shared memory
0 spills
```

Raw-QK + raw-PV specialization:

```text
70 registers/thread
20,992 bytes shared memory
0 spills
```

Comparison:

| Resource            |   Raw PV | Raw QK + Raw PV |    Change |
| ------------------- | -------: | --------------: | --------: |
| Registers/thread    |       79 |              70 |    −11.4% |
| Shared memory/block | 29,184 B |        20,992 B |    −28.1% |
| Spill stores        |        0 |               0 | unchanged |
| Spill loads         |        0 |               0 | unchanged |

The shared-memory reduction is exactly:

```text
29,184 − 20,992
= 8,192 bytes
```

matching the removed score buffer:

```text
4 × 16 × 32 × 4
= 8,192 bytes
```

---

## Benchmark Results

Configuration:

```text
GPU: RTX 4070 Laptop
B = 1
H = 8
D = 64
FP16
noncausal
```

|    N |    Raw PV | Raw QK + Raw PV |      Improvement |
| ---: | --------: | --------------: | ---------------: |
|  512 | 0.0839 ms |       0.0697 ms | **16.9% faster** |
| 1024 | 0.3164 ms |       0.2322 ms | **26.6% faster** |

Relative to PyTorch SDPA:

```text
N=512:
1.66× → 1.38× SDPA latency

N=1024:
3.90× → 2.86× SDPA latency
```

---

## Nsight Compute Results

At `N=1024`:

| Metric                |    Raw PV | Raw QK + Raw PV |
| --------------------- | --------: | --------------: |
| Profile duration      | 506.50 µs |   **358.02 µs** |
| Theoretical occupancy |     25.0% |       **33.3%** |
| Achieved occupancy    |    21.08% |      **28.99%** |
| Active warps/SM       |     10.12 |       **13.91** |
| Registers/thread      |        79 |          **70** |
| Shared memory/block   | 29.18 KiB |   **20.99 KiB** |
| Compute throughput    |    16.09% |      **23.56%** |
| L1/TEX throughput     |    88.66% |      **92.02%** |
| DRAM throughput       |     3.96% |       **5.46%** |

Removing `score_shared` increased the shared-memory residency limit:

```text
3 blocks/SM
→
4 blocks/SM
```

giving:

```text
12 theoretical active warps/SM
→
16 theoretical active warps/SM
```

and approximately 37% more achieved active warps.

---

## New Bottleneck

Experiment 7A successfully removed score materialization and scalar shared-memory softmax, but the remaining kernel is still strongly limited by the shared-memory/MIO path.

Nsight reports:

```text
L1/TEX throughput:       92.02%
DRAM throughput:          5.46%
Compute throughput:      23.56%

Shared-load conflicts:
approximately 16.5-way

Short-scoreboard stall:
10.7 / 19.9 cycles
≈ 53.9%
```

The increased short-scoreboard share does not indicate that Experiment 7A regressed dependency latency. Total kernel duration fell substantially; shared-memory dependencies simply became a larger fraction of the remaining execution time.

The dominant remaining shared-memory traffic now includes:

```text
probability_shared
K/V staging
ldmatrix operand loads
```

---

## Conclusion

Experiment 7A was successful:

```text
raw QKᵀ
+
shuffle softmax
+
score_shared elimination
```

reduced shared memory by 8 KiB, lowered register usage, increased residency from three to four blocks per SM, raised achieved occupancy from approximately 21% to 29%, and improved `N=1024` normal benchmark latency by approximately 27%.

The profile now points toward reducing repeated shared-memory operand traffic in the PV stage.

The immediate next experiment should therefore hoist and reuse the two packed probability `ldmatrix` fragments across all eight PV output subtiles before attempting full removal of `probability_shared`.


## Experiment 7B — Eliminate `probability_shared`

### Motivation

After fixing the Q/K/V shared-memory layouts in Experiment 7A.6, the remaining conflicted `ldmatrix` traffic came from the probability operand used by the raw PV Tensor Core path.

The raw-QK path already produced the softmax probabilities in registers, but the kernel still performed an unnecessary round trip:

```text
QK accumulator registers
    ↓
register softmax
    ↓
store FP16 probabilities to probability_shared
    ↓
ldmatrix probability_shared
    ↓
PV mma.sync
```

`probability_shared` occupied:

```text
4 warps × 16 rows × 32 columns × 2 bytes
= 4096 bytes
```

The goal of Experiment 7B was to keep the probability tile entirely in registers and feed it directly into the PV `mma.sync` operation.

### Register-layout observation

The raw-QK accumulator ownership already matched the layout required by the `m16n8k16` multiplicand-A fragment closely enough that no cross-lane shuffle was required.

For each lane, adjacent FP32 probability values could be converted to FP16 and packed directly into the four 32-bit registers of `MmaOperandA`.

The mapping was:

```text
score_accumulator[0] + score_accumulator[1]
    → probability columns 0:16

score_accumulator[2] + score_accumulator[3]
    → probability columns 16:32
```

Each pair of adjacent FP32 values was converted with `__floats2half2_rn` and packed into one 32-bit MMA operand register.

The new path became:

```text
QK accumulator registers
    ↓
register softmax
    ↓
FP32 → packed FP16x2
    ↓
MmaOperandA registers
    ↓
PV mma.sync
```

### Implementation strategy

The change was introduced in two stages.

First, the register-built probability fragments were used by raw PV while the existing `probability_shared` stores were left in place. This isolated the register-layout transformation from the shared-memory removal. The full correctness suite passed, confirming that the packed register ownership matched the Tensor Core operand layout.

After validating the mapping, the raw-QK probability stores were removed and `probability_shared` was reduced to a compile-time placeholder for the `<32, true, true>` specialization. Legacy kernel specializations that still require the shared probability tile were left unchanged.

### Resource impact

The final raw-QK/raw-PV specialization used:

```text
Registers per thread:          75
Static shared memory:       18.94 KiB
Local-memory spills:            0
Shared-memory block limit:      5
Register block limit:           6
Theoretical occupancy:      41.67%
Achieved occupancy:         28.20%
```

Removing the 4 KiB probability tile moved the shared-memory residency limit from four blocks per SM to five.

The higher theoretical residency did not fully translate into higher achieved occupancy for the N=1024 benchmark because the launch contained only 128 blocks across 36 SMs, corresponding to approximately 0.71 full waves.

Therefore, the measured latency improvement is attributed primarily to eliminating the probability shared-memory store/load round trip rather than increased realized occupancy.

### Performance

For the primary N=1024 workload:

```text
After Q/K/V padding:       0.1081 ms
After register P path:     0.1010 ms

Improvement:                 ~6.6%
Relative to PyTorch SDPA:    1.24× latency
```

At N=512 the result was approximately neutral to slightly slower:

```text
Before: 0.0394 ms
After:  0.0407 ms
```

The larger workload therefore remains the more representative target for this optimization.

### Profiling result

After Experiment 7B, the previously problematic probability `ldmatrix` load disappeared entirely.

Combined with the padded Q/K/V layouts from Experiment 7A.6, the major Tensor Core shared-memory operand loads now showed zero excessive shared-memory wavefronts.

Scheduler behavior also improved substantially relative to the earlier shared-memory-bound kernel:

```text
One-or-more eligible schedulers:  38.82%
Warp cycles / issued instruction:  8.65
L1/TEX throughput:                 52.93%
Compute throughput:                48.60%
```

The kernel had therefore transitioned away from being dominated by pathological shared-memory accesses toward a more balanced execution profile.

### Conclusion

Experiment 7B removed the final unnecessary probability-tile materialization from the raw-QK/raw-PV path.

The optimization produced a ~6.6% latency reduction at N=1024, reduced static shared-memory usage by approximately 4 KiB, raised theoretical block residency, and eliminated the remaining probability `ldmatrix` traffic.

The optimization sequence up to this point demonstrates a consistent strategy:

```text
remove intermediate materialization
    ↓
reuse operands in registers
    ↓
fix pathological shared-memory layouts
    ↓
eliminate the remaining shared-memory round trip
```

With shared-memory operand conflicts largely resolved, the next bottleneck shifted to the kernel epilogue, where the final FP16 output stores showed poor global-memory coalescing.


## Experiment 7C — Vectorize Final Output Stores

### Motivation

After eliminating the major shared-memory bottlenecks, Nsight Compute identified the final FP16 output epilogue as a remaining memory-layout inefficiency.

The raw MMA accumulator ownership distributed adjacent output values across register pairs, but the kernel stored each FP16 value individually. Each scalar store instruction therefore exposed a sparse warp-wide address pattern to the global-memory coalescer.

Before the change, Nsight reported:

```text
Useful bytes per 32-byte sector:     8
Excessive global sectors:       98,304
```

### Change

Adjacent MMA accumulator registers belonging to the same output row were paired:

```text
registers (0,1)
registers (2,3)
```

Each pair was normalized together, converted from FP32 to FP16 with `__floats2half2_rn`, and written using one 32-bit `half2` store.

This changed the epilogue from:

```text
4 × scalar FP16 stores per MMA subtile
```

to:

```text
2 × packed FP16x2 stores per MMA subtile
```

No cross-lane shuffles or register transpose were required.

### Profiling result

After vectorization:

```text
Useful bytes per 32-byte sector:    16
Excessive global sectors:       32,768
```

The excessive sector count fell by approximately **66.7%**, while useful bytes per transaction doubled.

Profiler kernel duration also improved:

```text
Before: 144.80 us
After:  139.94 us

Improvement: ~3.4%
```

Resource usage remained unchanged:

```text
Registers/thread:      75
Static shared memory:  18.94 KiB
Spills:                 0
```

### Benchmark result

```text
N=512:
0.0407 ms → 0.0353 ms
~13.3% faster

N=1024:
0.1010 ms → 0.0998 ms
~1.2% faster
```

The remaining uncoalesced accesses represented only a small fraction of total memory sectors, so a more complex warp-level epilogue transpose was not pursued.

### Conclusion

Experiment 7C improved output-store coalescing by exploiting the natural adjacency already present in MMA accumulator register pairs.

The cheap `half2` transformation captured most of the available epilogue improvement without additional shared memory, shuffles, or register pressure.

With the major layout and materialization inefficiencies removed, optimization now shifts from reducing memory traffic to **overlapping unavoidable K/V data movement with Tensor Core computation**.
