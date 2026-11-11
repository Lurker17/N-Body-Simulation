// render.cuh — a tiny graphics pipeline written as plain CUDA kernels.
//
// A real GPU graphics API (OpenGL/Vulkan) runs these same stages in fixed-function
// hardware + shaders. Here every stage is a kernel you can read:
//
//   bodies (world space, float4)
//        |  1. splatBodies     one thread per body:
//        |                     "vertex shader"  -> world -> camera -> screen pixel
//        |                     "rasterizer"     -> which pixels does the point touch?
//        |                     "blending"       -> atomicAdd light into those pixels
//        v
//   HDR framebuffer (float RGB per pixel, unbounded "amount of light")
//        |  2. blurH / blurV   one thread per pixel: Gaussian blur = "bloom" glow
//        v
//   HDR + bloom
//        |  3. tonemap         one thread per pixel: unbounded light -> 0..255 RGB
//        v
//   8-bit RGB image (rt.ldr, on the GPU). frame_pipeline.h copies it to the host
//   asynchronously and a writer thread feeds it to ffmpeg.
#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

// ============================================================================
// Camera — where we look from, and how 3D maps to 2D
// ============================================================================
// A camera is just an origin (eye) plus three orthonormal axes:
//   right, up       -> span the image plane
//   forward         -> direction we look in (depth axis)
// Going from world to camera space is a change of basis: three dot products.
// (OpenGL packs this into a 4x4 "view matrix"; it's the same math.)
struct Camera {
    float3 eye, right, up, forward;
    float focal;   // = 1 / tan(fov/2). Bigger -> zoomed in.
    float aspect;  // width / height
};

__host__ __device__ inline float3 f3sub(float3 a, float3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__host__ __device__ inline float f3dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ inline float3 f3cross(float3 a, float3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__host__ __device__ inline float3 f3norm(float3 a) {
    float s = 1.f / sqrtf(f3dot(a, a));
    return {a.x * s, a.y * s, a.z * s};
}

// Build a camera at `eye` looking at `target`, with `worldUp` roughly "up".
inline Camera lookAt(float3 eye, float3 target, float3 worldUp, float fovDeg, int w, int h) {
    Camera c;
    c.eye = eye;
    c.forward = f3norm(f3sub(target, eye));
    c.right = f3norm(f3cross(c.forward, worldUp));
    c.up = f3cross(c.right, c.forward);
    c.focal = 1.f / tanf(0.5f * fovDeg * 3.14159265f / 180.f);
    c.aspect = (float)w / (float)h;
    return c;
}

// Camera that circles the origin: the usual "orbit" camera.
inline Camera orbitCamera(float angleRad, float elevationRad, float distance, float fovDeg, int w,
                          int h) {
    float3 eye = {distance * cosf(elevationRad) * cosf(angleRad), distance * sinf(elevationRad),
                  distance * cosf(elevationRad) * sinf(angleRad)};
    return lookAt(eye, {0, 0, 0}, {0, 1, 0}, fovDeg, w, h);
}

// ============================================================================
// Projection — the core of the "vertex shader"
// ============================================================================
// world point -> camera space -> perspective divide -> pixel coordinates.
// Returns false if the point is behind the camera.
__device__ inline bool projectToScreen(const Camera& cam, float3 world, int width, int height,
                                       float& px, float& py, float& depth) {
    // (a) change of basis into camera space
    float3 d = f3sub(world, cam.eye);
    float xc = f3dot(d, cam.right);
    float yc = f3dot(d, cam.up);
    float zc = f3dot(d, cam.forward);  // distance along view direction
    if (zc < 0.01f) return false;      // behind us: clip it

    // (b) perspective divide: things twice as far look half as big.
    //     ndc ("normalized device coords") is in [-1, 1] across the screen.
    float ndcX = (cam.focal / cam.aspect) * xc / zc;
    float ndcY = cam.focal * yc / zc;

    // (c) viewport transform: [-1,1] -> [0,width] x [0,height] (y flipped: row 0 = top)
    px = (ndcX * 0.5f + 0.5f) * width;
    py = (1.f - (ndcY * 0.5f + 0.5f)) * height;
    depth = zc;
    return true;
}

// ============================================================================
// Render settings + GPU buffers
// ============================================================================
struct RenderSettings {
    int width = 1920, height = 1080;
    float brightness = 0.03f;  // light each body adds
    float exposure = 1.0f;     // tonemap: how quickly light saturates to white
    float bloom = 0.6f;        // glow strength (0 = off)
    float trail = 0.0f;        // 0 = clear each frame; 0.8 = keep 80% of last frame (motion trails)
};

constexpr int BLOOM_RADIUS = 12;
__constant__ float cBloomWeights[BLOOM_RADIUS + 1];  // Gaussian weights, half kernel

struct RenderTarget {
    int width, height;
    float4* hdr;      // accumulated light (rgb, w unused)
    float4* tmp;      // scratch for separable blur
    float4* bloomBuf; // blurred copy of hdr
    uchar3* ldr;      // final 8-bit image on GPU
};

inline RenderTarget createRenderTarget(int w, int h) {
    RenderTarget rt{w, h};
    size_t n = (size_t)w * h;
    cudaMalloc(&rt.hdr, n * sizeof(float4));
    cudaMalloc(&rt.tmp, n * sizeof(float4));
    cudaMalloc(&rt.bloomBuf, n * sizeof(float4));
    cudaMalloc(&rt.ldr, n * sizeof(uchar3));
    cudaMemset(rt.hdr, 0, n * sizeof(float4));

    // Precompute Gaussian weights once, normalized so the blur conserves light.
    float wts[BLOOM_RADIUS + 1], sum = 0.f, sigma = BLOOM_RADIUS / 2.5f;
    for (int i = 0; i <= BLOOM_RADIUS; i++) {
        wts[i] = expf(-(i * i) / (2.f * sigma * sigma));
        sum += (i == 0 ? 1.f : 2.f) * wts[i];
    }
    for (int i = 0; i <= BLOOM_RADIUS; i++) wts[i] /= sum;
    cudaMemcpyToSymbol(cBloomWeights, wts, sizeof(wts));
    return rt;
}

inline void destroyRenderTarget(RenderTarget& rt) {
    cudaFree(rt.hdr); cudaFree(rt.tmp); cudaFree(rt.bloomBuf); cudaFree(rt.ldr);
}

// ============================================================================
// Stage 0: clear (or fade, for motion trails)
// ============================================================================
__global__ void fadeFramebuffer(float4* hdr, int nPix, float keep) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nPix) return;
    float4 c = hdr[i];
    hdr[i] = make_float4(c.x * keep, c.y * keep, c.z * keep, 0.f);
}

// ============================================================================
// Stage 1: splat bodies  (vertex shader + rasterizer + additive blending)
// ============================================================================
// Each body is a point of light. It almost never lands exactly on a pixel center,
// so we split its light among the 4 nearest pixels by distance ("bilinear splat").
// That's anti-aliasing: points move smoothly instead of jumping pixel to pixel.
// Many bodies hit the same pixel -> atomicAdd. Adding light (instead of overwriting)
// is "additive blending"; dense regions naturally get brighter.
__device__ inline void addLight(float4* hdr, int w, int h, int x, int y, float3 c, float amt) {
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    float4* p = &hdr[y * w + x];
    atomicAdd(&p->x, c.x * amt);
    atomicAdd(&p->y, c.y * amt);
    atomicAdd(&p->z, c.z * amt);
}

__global__ void splatBodies(const float4* __restrict__ pos, const float4* __restrict__ color,
                            int N, Camera cam, float4* hdr, int w, int h, float brightness) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 p = pos[i];
    float px, py, depth;
    if (!projectToScreen(cam, make_float3(p.x, p.y, p.z), w, h, px, py, depth)) return;

    // pixel (x,y) has its center at (x+0.5, y+0.5)
    float fx = px - 0.5f, fy = py - 0.5f;
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    float tx = fx - x0, ty = fy - y0;  // fractional position inside the 2x2 cell

    float4 c4 = color[i];
    float3 c = make_float3(c4.x, c4.y, c4.z);
    addLight(hdr, w, h, x0,     y0,     c, brightness * (1 - tx) * (1 - ty));
    addLight(hdr, w, h, x0 + 1, y0,     c, brightness * tx       * (1 - ty));
    addLight(hdr, w, h, x0,     y0 + 1, c, brightness * (1 - tx) * ty);
    addLight(hdr, w, h, x0 + 1, y0 + 1, c, brightness * tx       * ty);
}

// ============================================================================
// Stage 2: bloom (post-processing)
// ============================================================================
// Real cameras/eyes bleed very bright light into neighbors. We fake it with a
// Gaussian blur of the image, added back on top. A 2D Gaussian is separable:
// blur rows, then blur columns => 2*(2R+1) taps per pixel instead of (2R+1)^2.
template <bool HORIZONTAL>
__global__ void gaussianBlur(const float4* __restrict__ src, float4* __restrict__ dst, int w,
                             int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float3 acc = make_float3(0, 0, 0);
    for (int k = -BLOOM_RADIUS; k <= BLOOM_RADIUS; k++) {
        int sx = HORIZONTAL ? min(max(x + k, 0), w - 1) : x;
        int sy = HORIZONTAL ? y : min(max(y + k, 0), h - 1);
        float4 s = src[sy * w + sx];
        float wt = cBloomWeights[abs(k)];
        acc.x += s.x * wt; acc.y += s.y * wt; acc.z += s.z * wt;
    }
    dst[y * w + x] = make_float4(acc.x, acc.y, acc.z, 0.f);
}

// ============================================================================
// Stage 3: tonemap  (HDR light -> displayable 8-bit color)
// ============================================================================
// The HDR buffer holds "how much light", from 0 to arbitrarily large. A screen
// can only show 0..255. Tonemapping squashes [0, inf) -> [0, 1) smoothly:
//     mapped = 1 - exp(-exposure * light)
// (dim stays proportional, bright saturates to white instead of clipping harshly)
// then gamma-encodes it (screens expect ~x^(1/2.2) for perceived brightness).
__global__ void tonemap(const float4* __restrict__ hdr, const float4* __restrict__ bloom,
                        uchar3* __restrict__ out, int nPix, float exposure, float bloomStrength) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nPix) return;
    float4 a = hdr[i], b = bloom[i];
    float r = a.x + bloomStrength * b.x;
    float g = a.y + bloomStrength * b.y;
    float bl = a.z + bloomStrength * b.z;
    auto map = [&](float v) {
        v = 1.f - __expf(-exposure * v);
        v = __powf(v, 1.f / 2.2f);
        return (unsigned char)(fminf(v, 1.f) * 255.f + 0.5f);
    };
    out[i] = make_uchar3(map(r), map(g), map(bl));
}

// ============================================================================
// The whole pipeline for one frame. Only enqueues kernels; returns immediately.
// Leaves the 8-bit image in rt.ldr.
// ============================================================================
inline void renderFrame(RenderTarget& rt, const Camera& cam, const float4* dPos,
                        const float4* dColor, int N, const RenderSettings& s) {
    int nPix = rt.width * rt.height;
    const int T = 256;

    // 0. clear (keep=0) or fade (keep>0 leaves motion trails)
    fadeFramebuffer<<<(nPix + T - 1) / T, T>>>(rt.hdr, nPix, s.trail);

    // 1. draw every body as a point of light
    splatBodies<<<(N + T - 1) / T, T>>>(dPos, dColor, N, cam, rt.hdr, rt.width, rt.height,
                                        s.brightness);

    // 2. bloom = blur(hdr), done as horizontal pass then vertical pass
    dim3 b2(16, 16), g2((rt.width + 15) / 16, (rt.height + 15) / 16);
    gaussianBlur<true><<<g2, b2>>>(rt.hdr, rt.tmp, rt.width, rt.height);
    gaussianBlur<false><<<g2, b2>>>(rt.tmp, rt.bloomBuf, rt.width, rt.height);

    // 3. HDR + bloom -> 8-bit RGB
    tonemap<<<(nPix + T - 1) / T, T>>>(rt.hdr, rt.bloomBuf, rt.ldr, nPix, s.exposure, s.bloom);
}
