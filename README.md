# FlashAttention CUDA Kernel

A from-scratch FlashAttention-style forward kernel written in CUDA/C++, built to study production GPU performance engineering at the Tensor Core, shared-memory, and warp-scheduling level.

The final kernel uses raw `mma.sync`, `ldmatrix`, `cp.async`, online softmax, XOR-swizzled shared memory, persistent FP32 accumulation, and a shared-memory output epilogue. It supports both noncausal and causal FP16 attention for the optimized `D=64` specialization.

## Performance

Test system:

* NVIDIA GeForce RTX 4070 Laptop GPU
* FP16
* `B=1`
* `H=8`
* `D=64`
* CUDA-event timing
* 30 samples
* 200 iterations per sample
* 30 warmup iterations

### N = 1024

| Mode      |     Custom Kernel |  PyTorch SDPA |
| --------- | ----------------: | ------------: |
| Noncausal | **~65.5–65.7 µs** | ~66.9–80.5 µs |
| Causal    | **~53.1–55.4 µs** | ~56.7–67.4 µs |

Across three isolated runs, the custom noncausal kernel reproduced at:

```text
65.7 µs
65.4 µs
65.7 µs
```

The causal specialization reproduced at:

```text
53.1 µs
55.4 µs
53.4 µs
```

The causal path is approximately **19% lower latency than the custom noncausal kernel** at this workload because future KV tiles are pruned instead of loaded and computed.

PyTorch SDPA showed substantially more run-to-run variance on the laptop GPU, so results are reported as local measurements rather than a claim of general superiority.

### Sequence-Length Sweep

|    N | Noncausal SDPA | Noncausal Custom | Causal SDPA | Causal Custom |
| ---: | -------------: | ---------------: | ----------: | ------------: |
|  128 |        37.5 µs |      **18.8 µs** |     26.7 µs |   **21.8 µs** |
|  256 |        20.6 µs |      **16.2 µs** |     35.2 µs |   **19.6 µs** |
|  512 |        29.3 µs |      **26.0 µs** | **26.1 µs** |       26.3 µs |
| 1024 |    **65.5 µs** |          65.7 µs |     63.9 µs |   **53.3 µs** |

Smaller-workload SDPA timings were noticeably noisier, so the N=1024 repeated measurements are the primary performance reference.

---

## Final Kernel Architecture

The optimized specialization uses:

```text
Tile:       128 × 128
Head dim:   64
Warps:      4
Input:      FP16
Accumulator FP32
```

Each warp owns two independent 16-row query slices:

```text
warp 0 → rows  0–15 and 64–79
warp 1 → rows 16–31 and 80–95
warp 2 → rows 32–47 and 96–111
warp 3 → rows 48–63 and 112–127
```

The main dataflow is:

```text
Q / K / V
   ↓
cp.async global → shared
   ↓
XOR-swizzled shared-memory layout
   ↓
ldmatrix
   ↓
QKᵀ using raw Tensor Core mma.sync
   ↓
online softmax in registers
   ↓
FP32 probability → packed FP16
   ↓
P × V using raw Tensor Core mma.sync
   ↓
persistent FP32 output accumulators
   ↓
shared-memory output ownership transform
   ↓
fully coalesced FP16 global stores
```

The production kernel uses approximately:

```text
255 registers / thread
48 KiB static shared memory
4 warps / block
128 threads / block
```

---

## Tensor Core Path

Both QK and PV use raw:

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

rather than WMMA abstractions.

Operand fragments are loaded using custom `ldmatrix` mappings designed around the MMA lane/register ownership model.

This made the register and shared-memory layout explicit and allowed later optimizations to reason directly about:

* fragment ownership
* register lifetime
* shared-memory bank layout
* output-store ownership
* Tensor Core scheduling

---

## Shared-Memory Layout

Q, K, and V are staged into shared memory using `cp.async`.

Each uses an XOR-swizzled layout to avoid shared-memory bank conflicts while remaining compatible with `ldmatrix`.

For the final 128×128 tile:

```text
Q = 128 × 64 × 2 B = 16 KiB
K = 128 × 64 × 2 B = 16 KiB
V = 128 × 64 × 2 B = 16 KiB

Total = 48 KiB
```

No additional shared-memory allocation is required for the output epilogue: once the final PV iteration completes, the dead V buffer is reused as output scratch space.

---

## Online Softmax

The kernel never materializes the full attention matrix in global memory.

Each query row maintains running softmax state:

```text
running maximum
running normalization sum
```

For every KV tile:

```text
new scores
   ↓
update running max
   ↓
renormalize previous output
   ↓
normalize current probabilities
   ↓
accumulate P × V
```

The two 16-row slices owned by each warp maintain independent softmax state.

---

## Persistent Output Accumulation

An early 128×128 design accumulated PV into temporary MMA fragments before merging them into the persistent output.

That produced severe register pressure and runtime spilling.

The final design accumulates PV directly into the persistent output fragments:

```text
P × V + O → O
```

instead of:

```text
P × V
  ↓
temporary PV fragment
  ↓
merge into O
```

This removed the large transient accumulator lifetime and substantially reduced spill traffic.

---

## Register-Lifetime Scheduling

Moving to a full 128×128 tile pushed the kernel to the 255-register architectural limit.

The first implementation generated approximately:

```text
81,152 runtime local-memory spill requests
```

The compute phases were reorganized into:

```text
QK
↓
normalize score fragments
↓
rescale persistent O
↓
convert normalized scores to packed FP16 P
↓
PV directly into O
```

This prevented score, probability, and output temporaries from all reaching peak liveness simultaneously.

Runtime spill requests fell by approximately **96%** while maintaining the full 128×128 tile.

An important result from this work was that minimizing the compiler's reported spill count was not itself the objective. Execution time improved even when a microscopic static spill remained.

---

## Coalesced Output Epilogue

Tensor Core accumulator ownership is optimized for MMA execution, not row-major global memory stores.

The initial direct FP16 epilogue produced:

```text
16 useful bytes / 32-byte global sector
65,536 global-store sectors
32,768 excessive sectors
```

The final epilogue reuses the dead V shared-memory tile to transform ownership:

```text
MMA-owned output registers
   ↓
normalize + FP16 conversion
   ↓
XOR-swizzled shared-memory scatter
   ↓
CTA synchronization
   ↓
warp-level row ownership
   ↓
32 lanes × half2
   ↓
contiguous 128-byte row store
```

Nsight Compute confirmed:

```text
global-store utilization:
16 B / sector → 32 B / sector

global-store sectors:
65,536 → 32,768

excessive sectors:
32,768 → 0
```

This reduced N=1024 kernel latency by roughly 4%.

---

## Causal Attention

Causal support is implemented as a compile-time specialization rather than a runtime branch inside the noncausal hot path.

The kernel exploits the 128×128 tile geometry at two levels.

### Whole-Tile Pruning

For a query tile beginning at `q0`:

```text
K tile < q0  → completely valid
K tile = q0  → diagonal tile
K tile > q0  → completely masked
```

Future KV tiles are never loaded or computed.

At N=1024:

```text
Noncausal:
8 query tiles × 8 KV tiles
= 64 tile interactions

Causal:
1 + 2 + 3 + ... + 8
= 36 tile interactions
```

### Diagonal Masking

Only the diagonal tile needs element-level masking.

For each MMA-owned score element:

```text
key_column > query_row
        ↓
      -infinity
```

The mask is applied after QK and before the online-softmax update.

This preserves the normal fast path for all fully valid tiles.

---

## Profiling-Driven Optimization

The kernel was optimized experimentally using Nsight Compute rather than by assuming that every low-level optimization would help.

The major progression was:

```text
SIMT attention
↓
Tensor Core PV
↓
Tensor Core QK
↓
register-resident softmax
↓
remove score/probability shared intermediates
↓
XOR-swizzled shared memory
↓
cp.async staging
↓
64×32 production prototype
↓
128×32
↓
128×64
↓
128×128
↓
persistent direct-O accumulation
↓
register-lifetime scheduling
↓
shared-memory output epilogue
↓
causal tile pruning
```

Several plausible optimizations were rejected when measurements showed they were not profitable.

---

## Rejected Software Pipeline

The final noncausal experiment attempted to overlap:

```text
V(t) copy  ↔ QK(t)
K(t+1) copy ↔ softmax(t) + PV(t)
```

using asynchronous copies.

The overlap did reduce some latency-related stalls:

```text
Long Scoreboard:
0.73 → 0.51

Math Pipe Throttle:
5.46 → 4.96
```

but the synchronization required for safe shared-buffer reuse increased:

```text
Wait:
1.19 → 1.46

Barrier:
0.14 → 0.28
```

and kernel latency regressed from approximately:

```text
65.4 µs → 68.4 µs
```

The optimization was therefore rejected.

This was the stopping point for noncausal optimization.

---

## Final Bottleneck

Source-correlated Nsight Compute profiling showed that approximately **99% of math-pipe-throttle samples** originated from Tensor Core `HMMA.16816.F32` instructions.

The stalls were split approximately:

```text
QK: 52%
PV: 48%
```

rather than being concentrated in softmax, shared-memory traffic, conversions, or output stores.

By the final implementation:

* global output stores were fully coalesced
* `cp.async` transactions were efficient
* shared-memory traffic showed no meaningful excess
* spill traffic was negligible
* memory throughput was not saturated

The remaining pressure was primarily the Tensor Core execution stream itself.

---

## Correctness

The final production specialization is tested against:

```python
torch.nn.functional.scaled_dot_product_attention
```

for both:

```text
causal=False
causal=True
```

and sequence lengths:

```text
128
256
512
1024
```

using FP16 inputs and `D=64`.

The project intentionally focuses on one optimized specialization rather than attempting full FlashAttention feature parity.

---

## Benchmarking

To benchmark only the final kernel and PyTorch SDPA:

```bash
python -m benchmarks.benchmark_forward \
  --batch 1 \
  --heads 8 \
  --query-length 1024 \
  --head-dim 64 \
  --dtype float16 \
  --kernels pytorch_sdpa tensorcore_production_128x128 \
  --samples 30 \
  --iterations 200 \
  --warmup 30
```

For causal attention:

```bash
python -m benchmarks.benchmark_forward \
  --batch 1 \
  --heads 8 \
  --query-length 1024 \
  --head-dim 64 \
  --dtype float16 \
  --causal \
  --kernels pytorch_sdpa tensorcore_production_128x128 \
  --samples 30 \
  --iterations 200 \
  --warmup 30
```

---

## Scope

The final optimized path intentionally specializes for:

```text
FP16
D = 64
self-attention
sequence length divisible by 128
causal or noncausal
NVIDIA Tensor Cores
```

This project is an implementation and performance-engineering study, not a replacement for production FlashAttention libraries.

The goal was to understand and reproduce the core techniques behind high-performance attention kernels from first principles:

```text
Tensor Core programming
warp/register ownership
shared-memory layout
asynchronous staging
online softmax
register lifetime management
coalesced output ownership
causal tile pruning
profiler-guided optimization
```

## Key Takeaway

The largest gains did not come from a single CUDA trick.

They came from treating tile geometry, register lifetime, Tensor Core ownership, shared-memory layout, synchronization, and global-memory layout as one coupled system—and using profiling to reject optimizations when their synchronization or scheduling cost exceeded their theoretical benefit.
