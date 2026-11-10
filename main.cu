// main.cu — glue: build a scene, then loop { simulate a few steps; render; encode }.
//
//   ./nbody                               # colliding galaxies, 64K bodies -> nbody.mp4
//   ./nbody --n 131072 --frames 900       # bigger / longer
//   ./nbody --scene sphere                # collapsing sphere
//   ./nbody --bench --n 262144            # physics only: GFLOP/s like the chapter's tables
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "physics.cuh"
#include "render.cuh"
#include "scene.h"
#include "video.h"

#define CUDA_CHECK(x)                                                                  \
    do {                                                                               \
        cudaError_t e = (x);                                                           \
        if (e != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, \
                    __LINE__);                                                         \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

struct Options {
    int n = 65536;
    int frames = 900;          // 30 s at 30 fps
    int stepsPerFrame = 4;     // simulation steps between rendered frames
    int blockSize = 256;       // "p" in the chapter: tile width = threads per block
    int fps = 30;
    std::string scene = "galaxies";
    std::string out = "nbody.mp4";
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
        else if (is("--block")) o.blockSize = atoi(argv[++i]);
        else if (is("--dt")) o.phys.dt = atof(argv[++i]);
        else if (is("--scene")) o.scene = argv[++i];
        else if (is("--out")) o.out = argv[++i];
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

static double gflops(int N, int steps, float ms) {
    double interactions = (double)N * N * steps;
    return interactions * FLOPS_PER_INTERACTION / (ms * 1e-3) / 1e9;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);  // show progress live even when piped
    Options opt = parseArgs(argc, argv);

    // ---- 1. initial conditions on the CPU ------------------------------------
    float softening = sqrtf(opt.phys.softening2);
    Scene scene = opt.scene == "sphere" ? makeCollapsingSphere(opt.n)
                                        : makeCollidingGalaxies(opt.n, softening);
    padToMultiple(scene, opt.blockSize);
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

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // ---- bench mode: physics only ---------------------------------------------
    if (opt.bench) {
        simulationStep(dPos, dVel, dAccel, N, opt.blockSize, opt.phys);  // warm-up
        int steps = 20;
        cudaEventRecord(t0);
        for (int s = 0; s < steps; s++) simulationStep(dPos, dVel, dAccel, N, opt.blockSize, opt.phys);
        cudaEventRecord(t1);
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        printf("N=%d  block=%d  %.3f ms/step  %.2f G interactions/s  %.1f GFLOP/s\n", N,
               opt.blockSize, ms / steps, (double)N * N * steps / (ms * 1e-3) / 1e9,
               gflops(N, steps, ms));
        return 0;
    }

    // ---- 3. main loop: simulate -> render -> encode ----------------------------
    RenderTarget rt = createRenderTarget(opt.render.width, opt.render.height);
    VideoWriter video;
    if (!openVideo(video, opt.out, rt.width, rt.height, opt.fps)) {
        fprintf(stderr, "could not start ffmpeg\n");
        return 1;
    }
    printf("N=%d bodies, %d frames x %d steps, %dx%d -> %s\n", N, opt.frames, opt.stepsPerFrame,
           rt.width, rt.height, opt.out.c_str());

    float physicsMs = 0.f;
    for (int frame = 0; frame < opt.frames; frame++) {
        // physics
        cudaEventRecord(t0);
        for (int s = 0; s < opt.stepsPerFrame; s++)
            simulationStep(dPos, dVel, dAccel, N, opt.blockSize, opt.phys);
        cudaEventRecord(t1);

        // graphics: a camera that slowly orbits and pulls back (debris spreads out)
        float progress = (float)frame / opt.frames;
        Camera cam = orbitCamera(0.4f + 0.0025f * frame, 0.45f,
                                 scene.cameraDistance * (1.f + 0.6f * progress), 45.f, rt.width,
                                 rt.height);
        renderFrame(rt, cam, dPos, dColor, N, opt.render);  // ends with a sync'ing memcpy
        CUDA_CHECK(cudaGetLastError());

        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        physicsMs += ms;

        writeFrame(video, rt.host);

        if (frame % 30 == 0 || frame == opt.frames - 1)
            printf("frame %4d/%d   physics %.2f ms/step   %.0f GFLOP/s\n", frame + 1, opt.frames,
                   ms / opt.stepsPerFrame, gflops(N, opt.stepsPerFrame, ms));
    }

    closeVideo(video);
    int totalSteps = opt.frames * opt.stepsPerFrame;
    printf("done. avg physics: %.2f ms/step, %.0f GFLOP/s.  wrote %s\n", physicsMs / totalSteps,
           gflops(N, totalSteps, physicsMs), opt.out.c_str());

    destroyRenderTarget(rt);
    cudaFree(dPos); cudaFree(dVel); cudaFree(dAccel); cudaFree(dColor);
    return 0;
}
