# N-Body Simulation

Gravitational N-body simulation in CUDA, rendered on the GPU and encoded to video.

Each version lives in its own folder and builds on its own.

| version | method | notes |
|---|---|---|
| [v1](v1/) | brute force, O(N²) | GPU Gems 3, ch. 31. Shared-memory tiling, point-splat renderer. |

## Requirements

- CUDA toolkit. The Makefiles target `sm_80` (A100); change `-arch` for other GPUs.
- `ffmpeg` with `libx264`.
