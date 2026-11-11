// main.cu (v2) — same loop as v1, with an autotuned force kernel and frames that
// are copied/encoded while the GPU keeps simulating.
//
//   ./nbody                               # colliding galaxies, 64K bodies -> nbody.mp4
//   ./nbody --n 131072 --frames 900
//   ./nbody --bench --n 65536             # every kernel variant: time + error vs v1
//   ./nbody --config 4,1,1                # force Q=4, prefetch on, fma chain on
//   ./nbody --slots 1                     # disable copy/encode overlap (for comparison)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "frame_pipeline.h"
#include "physics.cuh"
#include "render.cuh"
#include "scene.h"
#include "video.h"

#define CUDA_CHECK(x)                                                                    \
    do {                                                                                 \
        cudaError_t e = (x);                                                             \
        if (e != cudaSuccess) {                                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, \
                    __LINE__);                                                           \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

struct Options {
    int n = 65536;
    int frames = 900;       // 30 s at 30 fps
    int stepsPerFrame = 4;
    int fps = 30;
    int slots = 3;          // frames in flight between GPU and encoder
    std::string scene = "galaxies";
    std::string out = "nbody.mp4";
    std::string config = "auto";  // or "q,prefetch,fma", e.g. "4,1,1"
    bool bench = false;
    PhysicsParams phys{0.04f * 0.04f, 0.005f, 1.0f};  // softening^2, dt, damping
    RenderSettings render;
};

static Options parseArgs(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; i++) {
        auto is = [&](const char* k) { return strcmp(argv[i], k) == 0 && i + 1 < argc; };
        if (is("--n")) o.n = atoi(argv[++i]);
        else if (is("--frames")) o.frames = atoi(argv[++i]);
        else if (is("--steps")) o.stepsPerFrame = atoi(argv[++i]);
        else if (is("--slots")) o.slots = atoi(argv[++i]);
        else if (is("--dt")) o.phys.dt = atof(argv[++i]);
        else if (is("--scene")) o.scene = argv[++i];
        else if (is("--out")) o.out = argv[++i];
        else if (is("--config")) o.config = argv[++i];
        else if (is("--width")) o.render.width = atoi(argv[++i]);
        else if (is("--height")) o.render.height = atoi(argv[++i]);
        else if (is("--brightness")) o.render.brightness = atof(argv[++i]);
        else if (is("--exposure")) o.render.exposure = atof(argv[++i]);
        else if (is("--bloom")) o.render.bloom = atof(argv[++i]);
        else if (is("--trail")) o.render.trail = atof(argv[++i]);
        else if (strcmp(argv[i], "--bench") == 0) o.bench = true;
        else { fprintf(stderr, "unknown/incomplete arg: %s\n", argv[i]); exit(1); }
    }
    return o;
}

static double gflops(int N, double ms) {
    return (double)N * N * FLOPS_PER_INTERACTION / (ms * 1e-3) / 1e9;
}

// FP64 reference: one thread per body, direct sum in double precision.
// Used only by --bench to measure the float kernels' error.
__global__ void referenceForcesFP64(const float4* __restrict__ pos, double3* __restrict__ out,
                                    int N, double eps2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float4 bi = pos[i];
    double ax = 0, ay = 0, az = 0;
    for (int j = 0; j < N; j++) {
        float4 bj = pos[j];
        double rx = (double)bj.x - bi.x, ry = (double)bj.y - bi.y, rz = (double)bj.z - bi.z;
        double d2 = rx * rx + ry * ry + rz * rz + eps2;
        double s = bj.w / (d2 * sqrt(d2));
        ax += rx * s; ay += ry * s; az += rz * s;
    }
    out[i] = make_double3(ax, ay, az);
}

// Relative error |a - ref| / |ref| per body; returns {median, max}.
static std::pair<double, double> relError(const std::vector<float4>& a,
                                          const std::vector<double3>& ref) {
    std::vector<double> e;
    e.reserve(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        double dx = a[i].x - ref[i].x, dy = a[i].y - ref[i].y, dz = a[i].z - ref[i].z;
        double r = std::sqrt(ref[i].x * ref[i].x + ref[i].y * ref[i].y + ref[i].z * ref[i].z);
        if (r > 0) e.push_back(std::sqrt(dx * dx + dy * dy + dz * dz) / r);
    }
    std::sort(e.begin(), e.end());
    return {e[e.size() / 2], e.back()};
}

static void runBench(const float4* dPos, float4* dAccel, int N, float eps2) {
    auto download = [&] {
        std::vector<float4> h(N);
        CUDA_CHECK(cudaMemcpy(h.data(), dAccel, N * sizeof(float4), cudaMemcpyDeviceToHost));
        return h;
    };
    const int reps = N >= 131072 ? 5 : 20;

    double3* dRef;
    CUDA_CHECK(cudaMalloc(&dRef, N * sizeof(double3)));
    referenceForcesFP64<<<(N + 127) / 128, 128>>>(dPos, dRef, N, eps2);
    std::vector<double3> ref(N);
    CUDA_CHECK(cudaMemcpy(ref.data(), dRef, N * sizeof(double3), cudaMemcpyDeviceToHost));
    cudaFree(dRef);

    ForceConfig v1{1, false, false};
    float v1ms = timeForces(v1, dPos, dAccel, N, eps2, reps);

    printf("N=%d   error is relative to an FP64 direct sum\n", N);
    printf("%-4s %-9s %-10s %10s %12s %10s %8s %11s %11s\n", "Q", "prefetch", "fma_chain", "ms",
           "G inter/s", "GFLOP/s", "vs v1", "median err", "max err");
    for (int q : {1, 2, 4, 8})
        for (bool pf : {false, true})
            for (bool fma : {false, true}) {
                ForceConfig c{q, pf, fma};
                float ms = timeForces(c, dPos, dAccel, N, eps2, reps);
                launchForces(c, dPos, dAccel, N, eps2);
                auto [med, worst] = relError(download(), ref);
                printf("%-4d %-9d %-10d %10.3f %12.1f %10.0f %7.3fx %11.1e %11.1e\n", q, pf, fma,
                       ms, (double)N * N / (ms * 1e-3) / 1e9, gflops(N, ms), v1ms / ms, med, worst);
            }
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    Options opt = parseArgs(argc, argv);

    // ---- 1. initial conditions on the CPU ------------------------------------
    float softening = sqrtf(opt.phys.softening2);
    Scene scene = opt.scene == "sphere" ? makeCollapsingSphere(opt.n)
                                        : makeCollidingGalaxies(opt.n, softening);
    padToMultiple(scene, FORCE_BLOCK);
    int N = (int)scene.pos.size();

    // ---- 2. copy to the GPU --------------------------------------------------
    float4 *dPos, *dVel, *dAccel, *dColor;
    CUDA_CHECK(cudaMalloc(&dPos, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&dVel, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&dAccel, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&dColor, N * sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(dPos, scene.pos.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVel, scene.vel.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dColor, scene.color.data(), N * sizeof(float4), cudaMemcpyHostToDevice));

    if (opt.bench) {
        runBench(dPos, dAccel, N, opt.phys.softening2);
        CUDA_CHECK(cudaGetLastError());
        return 0;
    }

    // ---- 3. pick the force kernel variant ---------------------------------------
    ForceConfig cfg;
    if (opt.config == "auto") {
        cfg = autotuneForces(dPos, dAccel, N, opt.phys.softening2);
    } else {
        int q, pf, fma;
        if (sscanf(opt.config.c_str(), "%d,%d,%d", &q, &pf, &fma) != 3) {
            fprintf(stderr, "--config expects q,prefetch,fma e.g. 4,1,1\n");
            return 1;
        }
        cfg = ForceConfig{q, pf != 0, fma != 0};
    }
    CUDA_CHECK(cudaGetLastError());

    // ---- 4. main loop: simulate -> render -> (async) copy + encode -----------------
    RenderTarget rt = createRenderTarget(opt.render.width, opt.render.height);
    VideoWriter video;
    if (!openVideo(video, opt.out, rt.width, rt.height, opt.fps)) {
        fprintf(stderr, "could not start ffmpeg\n");
        return 1;
    }
    printf("N=%d bodies, %d frames x %d steps, %dx%d -> %s\nforce kernel: ", N, opt.frames,
           opt.stepsPerFrame, rt.width, rt.height, opt.out.c_str());
    printConfig(cfg);
    printf(", %d frame slots\n", opt.slots);

    double physicsMs = 0;
    auto collectTiming = [&](FrameSlot& s) {
        if (!s.timed) return;
        float ms;
        cudaEventElapsedTime(&ms, s.physStart, s.physEnd);
        physicsMs += ms;
        s.timed = false;
    };

    auto wall0 = std::chrono::steady_clock::now();
    {
        FramePipeline pipeline(video, opt.slots);
        for (int frame = 0; frame < opt.frames; frame++) {
            int s = pipeline.acquire();  // blocks only if every slot is still in flight
            FrameSlot& slot = pipeline.slot(s);
            collectTiming(slot);         // this slot's previous frame is done by now

            // physics
            cudaEventRecord(slot.physStart);
            for (int k = 0; k < opt.stepsPerFrame; k++)
                simulationStep(cfg, dPos, dVel, dAccel, N, opt.phys);
            cudaEventRecord(slot.physEnd);
            slot.timed = true;

            // graphics: a camera that slowly orbits and pulls back (debris spreads out)
            float progress = (float)frame / opt.frames;
            Camera cam = orbitCamera(0.4f + 0.0025f * frame, 0.45f,
                                     scene.cameraDistance * (1.f + 0.6f * progress), 45.f,
                                     rt.width, rt.height);
            renderFrame(rt, cam, dPos, dColor, N, opt.render);

            // async copy to the slot; the writer thread encodes it when it lands
            pipeline.submit(s, rt.ldr);

            if (frame % 60 == 0) printf("frame %4d/%d\n", frame + 1, opt.frames);
        }
        pipeline.finish();
        CUDA_CHECK(cudaGetLastError());
        for (auto& s : pipeline.slots()) collectTiming(s);
    }
    closeVideo(video);
    double wallS = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();

    int totalSteps = opt.frames * opt.stepsPerFrame;
    printf("done in %.1f s (%.1f fps). physics: %.1f s total, %.2f ms/step, %.0f GFLOP/s. wrote %s\n",
           wallS, opt.frames / wallS, physicsMs / 1e3, physicsMs / totalSteps,
           gflops(N, physicsMs / totalSteps), opt.out.c_str());

    destroyRenderTarget(rt);
    cudaFree(dPos); cudaFree(dVel); cudaFree(dAccel); cudaFree(dColor);
    return 0;
}
