# Brute-force N-body (GPU Gems 3, Ch. 31) + a CUDA-only renderer

```
make
./nbody                          # 64K bodies, colliding galaxies -> nbody.mp4 (~40 s)
./nbody --n 131072               # 128K bodies (~2.5 min)
./nbody --scene sphere           # cold collapsing sphere
./nbody --bench --n 262144       # physics only, prints GFLOP/s (chapter's 20 FLOP/interaction)
```

Other flags: `--frames --steps --dt --width --height --brightness --exposure --bloom --trail --block --out`

## Files

| file | what |
|---|---|
| `physics.cuh` | The chapter's code: `bodyBodyInteraction` → `tileCalculation` → `calculateForces` (shared-memory tiling), plus a leapfrog `integrateBodies`. |
| `render.cuh`  | The graphics pipeline written as CUDA kernels (explained below). |
| `scene.h`     | Initial conditions on the CPU (disk galaxies, sphere). |
| `video.h`     | Pipes raw RGB frames into `ffmpeg`. |
| `main.cu`     | Loop: `simulationStep` × k → `renderFrame` → `writeFrame`. |

## Why no OpenGL?

The A100 has no display output, and in MIG mode it doesn't support OpenGL/EGL at all.
So the renderer is plain CUDA kernels, and each frame is encoded into an mp4.
This works on any CUDA GPU and has the side benefit of exposing every stage.

## Graphics in CUDA terms

| graphics word | what it actually is here |
|---|---|
| **framebuffer** | a `float4[width*height]` array in global memory |
| **camera / view transform** | change of basis: `dot(p - eye, right/up/forward)` → camera-space (x, y, z) |
| **perspective projection** | `x/z`, `y/z` (far things shrink), scaled by focal length = `1/tan(fov/2)` |
| **NDC → viewport** | map [-1,1] to [0,width] pixels; flip y because row 0 is the top |
| **vertex shader** | a kernel with one thread per body that computes its pixel position (`projectToScreen`) |
| **rasterization** | deciding which pixels a primitive covers. For a point, that's the 4 nearest pixels with bilinear weights (anti-aliasing) |
| **additive blending** | `atomicAdd` of light into the framebuffer, so dense regions get brighter |
| **HDR** | the framebuffer holds unbounded floats (an amount of light), not 0–255 |
| **bloom** | a separable Gaussian blur of the image added back on top, so bright regions glow |
| **tonemapping** | `1 - exp(-exposure * light)` squashes [0,∞) into [0,1), then gamma `x^(1/2.2)` and convert to 8-bit |
| **fragment / pixel shader** | a kernel with one thread per pixel (`gaussianBlur`, `tonemap`) |

A frame is 6 kernel launches (`renderFrame` in `render.cuh`): fade → splat → blurH → blurV → tonemap → memcpy.

## Numbers on this A100 (MIG 7g.80gb)

| N | ms/step | GFLOP/s |
|---|---|---|
| 16K  | 0.68 | 7,900 |
| 64K  | 7.8  | 11,000 |
| 128K | 31   | 11,100 |
| 256K | 113  | 12,100 |

FP32 peak is 19.5 TFLOP/s. Rendering a 1080p frame costs well under a millisecond, so physics dominates.

## Physics notes

- The kernel requires `N % blockSize == 0`, so `padToMultiple` adds massless bodies (`w = 0`) far away. They exert no force.
- There is no `i == j` check, as in the chapter: `r = 0`, so with softening the self-term is exactly 0.
- Cold, thin, self-gravitating disks fragment into clumps (a Toomre instability). `addDiskGalaxy` keeps 60% of the mass in the central body and adds 10% velocity dispersion to stabilize them; those are the `centralFraction` / `dispersion` knobs.
