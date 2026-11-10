# v1: brute force

Direct-sum gravity following Nyland, Harris and Prins,
[Fast N-Body Simulation with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda)
(GPU Gems 3, ch. 31). Every body interacts with every other body on every step.

## Build and run

```
make
./nbody                          # 64K bodies, colliding galaxies -> nbody.mp4
./nbody --n 131072               # 128K bodies
./nbody --scene sphere           # collapsing sphere
./nbody --bench --n 262144       # physics only, prints GFLOP/s
```

| flag | default | |
|---|---|---|
| `--n` | 65536 | number of bodies (rounded up to a multiple of `--block`) |
| `--scene` | `galaxies` | `galaxies` or `sphere` |
| `--frames` | 900 | video frames, at 30 fps |
| `--steps` | 4 | simulation steps per frame |
| `--dt` | 0.005 | timestep |
| `--block` | 256 | threads per block, which is also the tile size |
| `--width`, `--height` | 1920, 1080 | output resolution |
| `--brightness` | 0.03 | light added per body |
| `--exposure` | 1.0 | tonemap exposure |
| `--bloom` | 0.6 | glow strength, 0 disables it |
| `--trail` | 0.0 | fraction of the previous frame kept, for motion trails |
| `--out` | `nbody.mp4` | output file |
| `--bench` | | run 20 physics steps, print timing, exit |

## Files

| file | contents |
|---|---|
| `physics.cuh` | force kernel and integrator |
| `render.cuh` | camera, projection and the rendering kernels |
| `scene.h` | initial conditions, built on the host |
| `video.h` | pipes raw RGB frames to ffmpeg |
| `main.cu` | argument parsing and the simulate/render/encode loop |

## Physics

Each body is a `float4` of `(x, y, z, mass)`. Acceleration on body i:

```
a_i = sum_j  m_j * r_ij / (|r_ij|^2 + eps^2)^(3/2)        (G = 1)
```

`calculateForces` assigns one thread per body. Each block loads a tile of `blockDim.x`
positions into shared memory, every thread accumulates against the tile, and the
block moves on to the next tile. This cuts global memory reads by a factor of the
block size. The structure follows the chapter's listings 31-1 to 31-3
(`bodyBodyInteraction`, `tileCalculation`, `calculateForces`).

- The integrator is kick-drift leapfrog: `v += a*dt; x += v*dt`.
- Softening `eps = 0.04` keeps close encounters finite. It also makes the self-term
  (i = j, r = 0) exactly zero, so the kernel has no branch for it.
- The kernel requires `N % blockSize == 0`. `padToMultiple` fills the remainder with
  massless bodies placed far away.
- FLOP counts use the chapter's convention of 20 per interaction.

### Initial conditions

`galaxies` builds two disks on a bound, off-center encounter. Each disk has a heavy
central body plus stars on circular orbits, with speed set from the enclosed mass.
A cold, thin, self-gravitating disk fragments into clumps within a few orbits. To
keep the disks stable until the encounter, 60% of each galaxy's mass is in the central
body and stars get a 10% random velocity dispersion (`centralFraction` and
`dispersion` in `addDiskGalaxy`).

`sphere` is a uniform ball with slow rotation that collapses under its own gravity.

## Rendering

Rendering is done in CUDA kernels. Each frame runs the following stages
(`renderFrame` in `render.cuh`):

1. `fadeFramebuffer` clears the float RGB accumulation buffer, or scales it by
   `--trail` to leave motion trails.
2. `splatBodies` runs one thread per body. It transforms the position into camera
   space (dot products with the camera's right, up and forward axes), divides by
   depth for perspective, and maps the result to pixel coordinates. It then
   `atomicAdd`s the body's color into the four nearest pixels with bilinear weights,
   so dense regions accumulate more light.
3. `gaussianBlur` runs a horizontal pass and then a vertical pass, producing the
   bloom buffer.
4. `tonemap` maps `hdr + bloom * blurred` through `1 - exp(-exposure * x)`, applies
   gamma 2.2 and writes 8-bit RGB.
5. The image is copied to the host and written to the ffmpeg pipe.

The camera orbits the origin and pulls back over the run as debris spreads.

## Performance

A100 80GB PCIe (MIG 7g.80gb), CUDA 13.0, `-O3 -use_fast_math`, block size 256:

| N | ms/step | interactions/s | GFLOP/s |
|---|---|---|---|
| 16,384 | 0.68 | 394 G | 7,881 |
| 65,536 | 7.84 | 548 G | 10,961 |
| 131,072 | 31.0 | 554 G | 11,084 |
| 262,144 | 113.2 | 607 G | 12,142 |

FP32 peak on the A100 is 19.5 TFLOP/s. The force kernel dominates runtime. In a
720-frame run at 64K bodies, physics took 22.6 s of the 24.9 s total; rendering and
encoding took the rest.
