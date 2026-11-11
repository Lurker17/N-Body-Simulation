# v2: faster brute force

Same simulation and renderer as [v1](../v1/), with four changes:

1. **FMA chain.** The softening term is folded into the distance FMA chain:
   `fmaf(rx,rx, fmaf(ry,ry, fmaf(rz,rz, eps2)))`. Each interaction drops from 13 to
   12 FP32 instructions (6 FFMA, 3 FMUL, 3 FADD, plus one MUFU.RSQ).
2. **Q split** (GPU Gems 3, 31.4.2). Each body's sum over N is split across Q threads.
   A block is 256 threads arranged as (256/Q bodies) × (Q threads per body), so the
   grid has Q times more blocks. The Q partial sums are added through shared memory
   at the end.
3. **Prefetching.** The shared-memory tile is double buffered with `cp.async`: tile
   t+1 is copied while tile t is computed.
4. **Frame pipeline.** Frames are copied to pinned host buffers asynchronously and
   a writer thread feeds them to ffmpeg, so the GPU keeps simulating while frames are
   encoded.

Changes 1–3 are template parameters of `calculateForces`. At startup, the program
times all 16 combinations on the actual N and uses the fastest.

## Build and run

```
make
./nbody                          # 64K bodies -> nbody.mp4, autotuned kernel
./nbody --bench --n 131072       # time and error of all 16 kernel variants
./nbody --config 8,0,1           # force Q=8, no prefetch, FMA chain
./nbody --slots 1                # turn off copy/encode overlap
```

Flags are the same as v1, plus `--config q,prefetch,fma` (default `auto`) and
`--slots` (frames in flight, default 3).

## Results

A100 80GB PCIe, MIG 7g.80gb (98 SMs, 1.41 GHz), CUDA 13.0.

### Force kernel, speedup over v1

| N | v1 (ms) | + FMA chain | + prefetch only | best | best config |
|---|---|---|---|---|---|
| 16,384 | 0.704 | 1.13x | 1.05x | 1.50x | Q=8, FMA |
| 65,536 | 7.97 | 1.09x | 1.01x | 1.25x | Q=8, FMA |
| 131,072 | 31.5 | 1.09x | 1.01x | 1.25x | Q=8, FMA |
| 262,144 | 114.3 | 1.09x | 1.01x | 1.14x | Q=8, FMA |

- **FMA chain:** +9%, matching 13 → 12 instructions.
- **Q split:** the largest gain at small N. At 16,384 bodies v1 launches 64
  blocks for 98 SMs, and Q=8 launches 512. It still gives 5–14% at large N. A
  likely cause is the last wave: at 262,144 bodies v1 has 1,024 blocks for about 588
  resident slots, so the second wave runs partly empty. This is not confirmed,
  because performance counters aren't accessible on this machine.
- **Prefetch:** about 1%, and slightly negative when combined with Q=8. The kernel
  is compute bound, and 48 resident warps per SM already hide the global load
  latency, so there is little stall left to remove.

At 262,144 bodies the best variant reaches 685 billion interactions/s. With 12 FP32
instructions per interaction, the issue limit is 98 × 64 × 1.41 GHz / 12 =
737 billion/s, so the kernel runs at 93% of it.

### Accuracy

Error per body is measured against an FP64 direct sum (`--bench`). At 131,072 bodies:

| | median rel. error | max rel. error |
|---|---|---|
| v1 (Q=1) | 2.9e-5 | 3.0e-3 |
| Q=8 | 2.0e-6 | 2.3e-5 |

The Q split is also more accurate. Each thread adds N/Q terms instead of N, so float
rounding builds up less. The FMA chain does not change the error.

### End to end

65,536 bodies, 300 frames at 1080p:

| | wall time | fps |
|---|---|---|
| v1 kernel, no overlap (`--config 1,0,0 --slots 1`) | 12.4 s | 24.2 |
| v1 kernel, overlap (`--config 1,0,0`) | 10.1 s | 29.8 |
| v2 (autotuned + overlap) | 8.0 s | 37.3 |

With overlap, wall time is within 0.5 s of physics time.

## Files

| file | contents |
|---|---|
| `physics.cuh` | templated force kernel, integrator, autotuner |
| `frame_pipeline.h` | pinned frame slots and the writer thread |
| `render.cuh` | v1 renderer; `renderFrame` now only enqueues kernels |
| `scene.h`, `video.h` | unchanged from v1 |
| `main.cu` | loop, `--bench` with FP64 reference |
