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
