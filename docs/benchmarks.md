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

