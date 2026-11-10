// physics.cuh — brute-force O(N^2) gravity, straight from GPU Gems 3, Ch. 31
// (Nyland, Harris, Prins: "Fast N-Body Simulation with CUDA").
//
// Layout (same as the chapter):
//   pos[i] = (x, y, z, mass)      float4 so one 16-byte load gets everything
//   vel[i] = (vx, vy, vz, unused)
//
// The idea of the chapter: every thread owns one body i and must sum the pull
// of all N bodies j. Instead of every thread reading all N positions from
// global memory, the block cooperatively loads a "tile" of blockDim.x bodies
// into shared memory, everyone consumes that tile, then the next tile is loaded.
// Global traffic drops by a factor of blockDim.x.
#pragma once
#include <cuda_runtime.h>

struct PhysicsParams {
    float softening2;  // EPS^2 in the chapter; keeps the force finite when r -> 0
    float dt;          // timestep
    float damping;     // 1.0 = none; slightly <1 bleeds energy for stability
};

// --- Chapter Listing 31-1: one body-body interaction (~20 FLOPs) ------------
// a_i += m_j * r_ij / (|r_ij|^2 + eps^2)^(3/2)       (G = 1)
__device__ __forceinline__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai,
                                                      float softening2) {
    float3 r;
    r.x = bj.x - bi.x;                                  // 3 FLOPs
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + softening2;  // 6 FLOPs
    float distSixth = distSqr * distSqr * distSqr;      // 2 FLOPs
    float invDistCube = rsqrtf(distSixth);              // 2 FLOPs (counted as in paper)
    float s = bj.w * invDistCube;                       // 1 FLOP
    ai.x += r.x * s;                                    // 6 FLOPs
    ai.y += r.y * s;
    ai.z += r.z * s;
    return ai;
}
constexpr double FLOPS_PER_INTERACTION = 20.0;  // the chapter's convention

// --- Chapter Listing 31-2: consume one tile that is sitting in shared memory --
__device__ __forceinline__ float3 tileCalculation(float4 myPosition, float3 accel,
                                                  const float4* shPosition, float softening2) {
#pragma unroll 32
    for (int j = 0; j < blockDim.x; j++)
        accel = bodyBodyInteraction(myPosition, shPosition[j], accel, softening2);
    return accel;
}

// --- Chapter Listing 31-3: the full force kernel ------------------------------
// grid = N / p blocks, block = p threads, shared mem = p * sizeof(float4).
// Requires N % p == 0 (main.cu rounds N up to guarantee it).
__global__ void calculateForces(const float4* __restrict__ pos, float4* __restrict__ accelOut,
                                int N, float softening2) {
    extern __shared__ float4 shPosition[];
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    float4 myPosition = pos[gtid];
    float3 acc = make_float3(0.f, 0.f, 0.f);

    for (int tile = 0; tile < N / blockDim.x; tile++) {
        // 1) everyone loads one body of this tile into shared memory
        shPosition[threadIdx.x] = pos[tile * blockDim.x + threadIdx.x];
        __syncthreads();
        // 2) everyone computes against all p bodies of the tile
        acc = tileCalculation(myPosition, acc, shPosition, softening2);
        __syncthreads();  // don't overwrite the tile while others still read it
    }
    accelOut[gtid] = make_float4(acc.x, acc.y, acc.z, 0.f);
}

// --- Integration: leapfrog "kick-drift" (what the chapter's demo uses) -------
//   v += a * dt ;  x += v * dt
__global__ void integrateBodies(float4* __restrict__ pos, float4* __restrict__ vel,
                                const float4* __restrict__ accel, int N, PhysicsParams p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float4 x = pos[i], v = vel[i], a = accel[i];
    v.x = (v.x + a.x * p.dt) * p.damping;
    v.y = (v.y + a.y * p.dt) * p.damping;
    v.z = (v.z + a.z * p.dt) * p.damping;
    x.x += v.x * p.dt;
    x.y += v.y * p.dt;
    x.z += v.z * p.dt;
    pos[i] = x;  // x.w (mass) untouched
    vel[i] = v;
}

// Host helper: one full simulation step = forces + integrate.
inline void simulationStep(float4* dPos, float4* dVel, float4* dAccel, int N, int blockSize,
                           const PhysicsParams& p) {
    int blocks = N / blockSize;
    calculateForces<<<blocks, blockSize, blockSize * sizeof(float4)>>>(dPos, dAccel, N,
                                                                       p.softening2);
    integrateBodies<<<blocks, blockSize>>>(dPos, dVel, dAccel, N, p);
}
