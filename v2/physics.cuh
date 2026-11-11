// physics.cuh (v2): the v1 force kernel plus three independent changes, each a
// template switch so they can be benchmarked one at a time:
//
//   FMA_CHAIN  fold eps^2 into the distance FMA chain: 13 -> 12 FP32 instructions
//   Q          split each body's j-range across Q threads (GPU Gems 3 ch. 31, 31.4.2),
//              so small N still launches enough blocks to fill every SM
//   PREFETCH   double-buffer the shared-memory tile with cp.async: load tile t+1
//              while computing on tile t
//
// Q = 1, PREFETCH = false, FMA_CHAIN = false is exactly the v1 kernel.
#pragma once
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <cstdio>

struct PhysicsParams {
    float softening2;  // eps^2
    float dt;
    float damping;     // 1.0 = none
};

constexpr int FORCE_BLOCK = 256;                 // threads per block, P * Q
constexpr double FLOPS_PER_INTERACTION = 20.0;   // the chapter's convention

// --- one interaction ---------------------------------------------------------
template <bool FMA_CHAIN>
__device__ __forceinline__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai,
                                                      float softening2) {
    float3 r = make_float3(bj.x - bi.x, bj.y - bi.y, bj.z - bi.z);  // 3 FADD
    float distSqr;
    if (FMA_CHAIN)
        // eps^2 is the starting value of the chain: 3 FFMA
        distSqr = fmaf(r.x, r.x, fmaf(r.y, r.y, fmaf(r.z, r.z, softening2)));
    else
        // v1: compiles to FMUL + 2 FFMA + FADD
        distSqr = r.x * r.x + r.y * r.y + r.z * r.z + softening2;
    float distSixth = distSqr * distSqr * distSqr;  // 2 FMUL
    float invDistCube = rsqrtf(distSixth);          // MUFU.RSQ
    float s = bj.w * invDistCube;                   // 1 FMUL
    ai.x = fmaf(r.x, s, ai.x);                      // 3 FFMA
    ai.y = fmaf(r.y, s, ai.y);
    ai.z = fmaf(r.z, s, ai.z);
    return ai;
}

// --- consume one tile of P bodies from shared memory ------------------------
template <int P, bool FMA_CHAIN>
__device__ __forceinline__ float3 tileCalculation(float4 me, float3 acc, const float4* tile,
                                                  float softening2) {
#pragma unroll 32
    for (int j = 0; j < P; j++) acc = bodyBodyInteraction<FMA_CHAIN>(me, tile[j], acc, softening2);
    return acc;
}

// --- force kernel --------------------------------------------------------------
// Block = (P, Q) threads, P * Q = 256. The block owns P bodies (threadIdx.x).
// Thread row ty handles the slice of bodies [ty * N/Q, (ty+1) * N/Q).
// Each row has its own P-wide tile in shared memory; after the loop the Q partial
// accelerations of a body are summed through shared memory.
//
// Shared memory: [stage][Q][P] float4, stage = 2 with PREFETCH, else 1.
// Requires N % 256 == 0.
template <int Q, bool PREFETCH, bool FMA_CHAIN>
__global__ void __launch_bounds__(FORCE_BLOCK)
calculateForces(const float4* __restrict__ pos, float4* __restrict__ accelOut, int N,
                float softening2) {
    constexpr int P = FORCE_BLOCK / Q;
    extern __shared__ float4 sh[];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int i = blockIdx.x * P + tx;

    const float4 me = pos[i];
    float3 acc = make_float3(0.f, 0.f, 0.f);

    const int sliceStart = ty * (N / Q);
    const int numTiles = (N / Q) / P;
    auto tileBuf = [&](int stage) { return sh + (stage * Q + ty) * P; };

    if (!PREFETCH) {
        // v1 structure: load, sync, compute, sync
        for (int t = 0; t < numTiles; t++) {
            tileBuf(0)[tx] = pos[sliceStart + t * P + tx];
            __syncthreads();  // tile fully written before anyone reads it
            acc = tileCalculation<P, FMA_CHAIN>(me, acc, tileBuf(0), softening2);
            __syncthreads();  // everyone done reading before it's overwritten
        }
    } else {
        // Double buffering with cp.async (global -> shared without going through
        // registers, and without blocking the issuing thread).
        // Stage t&1 holds tile t. Before computing on tile t we start the copy of
        // tile t+1 into the other stage.
        __pipeline_memcpy_async(&tileBuf(0)[tx], &pos[sliceStart + tx], sizeof(float4));
        __pipeline_commit();
        for (int t = 0; t < numTiles; t++) {
            if (t + 1 < numTiles)
                __pipeline_memcpy_async(&tileBuf((t + 1) & 1)[tx],
                                        &pos[sliceStart + (t + 1) * P + tx], sizeof(float4));
            __pipeline_commit();       // (empty group on the last tile keeps the count uniform)
            __pipeline_wait_prior(1);  // my copy of tile t has landed; tile t+1 may be in flight
            __syncthreads();           // everyone's copy of tile t has landed
            acc = tileCalculation<P, FMA_CHAIN>(me, acc, tileBuf(t & 1), softening2);
            __syncthreads();           // everyone done with stage t&1 before tile t+2 refills it
        }
    }

    if (Q == 1) {
        accelOut[i] = make_float4(acc.x, acc.y, acc.z, 0.f);
        return;
    }
    // Sum the Q partial accelerations of body i. The last loop iteration ended
    // with __syncthreads, so the shared buffer is free to reuse.
    sh[ty * P + tx] = make_float4(acc.x, acc.y, acc.z, 0.f);
    __syncthreads();
    if (ty == 0) {
        for (int q = 1; q < Q; q++) {
            float4 p = sh[q * P + tx];
            acc.x += p.x; acc.y += p.y; acc.z += p.z;
        }
        accelOut[i] = make_float4(acc.x, acc.y, acc.z, 0.f);
    }
}

// --- integration (unchanged from v1) --------------------------------------------
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
    pos[i] = x;
    vel[i] = v;
}

// --- picking a variant at runtime ------------------------------------------------
struct ForceConfig {
    int q = 1;
    bool prefetch = false;
    bool fmaChain = false;
};

inline void printConfig(const ForceConfig& c, FILE* f = stdout) {
    fprintf(f, "Q=%d prefetch=%d fma_chain=%d", c.q, c.prefetch, c.fmaChain);
}

template <int Q, bool PF, bool FMA>
inline void launchForcesT(const float4* dPos, float4* dAccel, int N, float eps2, cudaStream_t s) {
    constexpr int P = FORCE_BLOCK / Q;
    size_t smem = (PF ? 2 : 1) * FORCE_BLOCK * sizeof(float4);
    calculateForces<Q, PF, FMA><<<N / P, dim3(P, Q), smem, s>>>(dPos, dAccel, N, eps2);
}

template <int Q>
inline void launchForcesQ(const ForceConfig& c, const float4* dPos, float4* dAccel, int N,
                          float eps2, cudaStream_t s) {
    if (c.prefetch) {
        if (c.fmaChain) launchForcesT<Q, true, true>(dPos, dAccel, N, eps2, s);
        else            launchForcesT<Q, true, false>(dPos, dAccel, N, eps2, s);
    } else {
        if (c.fmaChain) launchForcesT<Q, false, true>(dPos, dAccel, N, eps2, s);
        else            launchForcesT<Q, false, false>(dPos, dAccel, N, eps2, s);
    }
}

inline void launchForces(const ForceConfig& c, const float4* dPos, float4* dAccel, int N,
                         float eps2, cudaStream_t s = 0) {
    switch (c.q) {
        case 1: launchForcesQ<1>(c, dPos, dAccel, N, eps2, s); break;
        case 2: launchForcesQ<2>(c, dPos, dAccel, N, eps2, s); break;
        case 4: launchForcesQ<4>(c, dPos, dAccel, N, eps2, s); break;
        case 8: launchForcesQ<8>(c, dPos, dAccel, N, eps2, s); break;
        default: fprintf(stderr, "Q must be 1, 2, 4 or 8\n"); exit(1);
    }
}

// One simulation step = forces + integrate.
inline void simulationStep(const ForceConfig& c, float4* dPos, float4* dVel, float4* dAccel, int N,
                           const PhysicsParams& p, cudaStream_t s = 0) {
    launchForces(c, dPos, dAccel, N, p.softening2, s);
    integrateBodies<<<(N + 255) / 256, 256, 0, s>>>(dPos, dVel, dAccel, N, p);
}

// Time the force kernel alone (positions are only read, so this has no side effects).
inline float timeForces(const ForceConfig& c, const float4* dPos, float4* dAccel, int N,
                        float eps2, int reps) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    launchForces(c, dPos, dAccel, N, eps2);  // warm-up
    cudaEventRecord(a);
    for (int r = 0; r < reps; r++) launchForces(c, dPos, dAccel, N, eps2);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / reps;
}

// Try every variant on the real N and keep the fastest.
inline ForceConfig autotuneForces(const float4* dPos, float4* dAccel, int N, float eps2) {
    ForceConfig best;
    float bestMs = 1e30f;
    for (int q : {1, 2, 4, 8})
        for (bool pf : {false, true})
            for (bool fma : {false, true}) {
                ForceConfig c{q, pf, fma};
                float ms = timeForces(c, dPos, dAccel, N, eps2, N >= 131072 ? 1 : 5);
                if (ms < bestMs) { bestMs = ms; best = c; }
            }
    return best;
}
