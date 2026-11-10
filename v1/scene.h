// scene.h — initial conditions, built on the CPU (plain C++, nothing CUDA-specific).
//
// Every scene fills three arrays of length N:
//   pos[i]   = (x, y, z, mass)
//   vel[i]   = (vx, vy, vz, 0)
//   color[i] = (r, g, b, 0)     only used by the renderer
#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <string>
#include <vector>

struct Scene {
    std::vector<float4> pos, vel, color;
    float cameraDistance = 8.f;  // how far the orbit camera sits
};

namespace detail {
inline float3 add(float3 a, float3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline float3 mul(float3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float3 cross(float3 a, float3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float3 normalize(float3 a) {
    float l = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    return mul(a, 1.f / l);
}
inline float3 lerp(float3 a, float3 b, float t) { return add(mul(a, 1 - t), mul(b, t)); }
}  // namespace detail

// A disk galaxy: one heavy central body ("black hole") + `count-1` stars on
// roughly circular orbits around it.
//   v_circular(r) = sqrt(G * M_enclosed(r) / r), adjusted for softening.
inline void addDiskGalaxy(Scene& s, int count, float3 center, float3 bulkVel, float3 normal,
                          float totalMass, float radius, float3 coreColor, float3 edgeColor,
                          float softening, std::mt19937& rng) {
    // Knobs: a cold, thin, heavy disk is gravitationally unstable and breaks into
    // clumps. Real galaxies are stabilized by a dark-matter halo; we approximate
    // that by putting most mass in the center and adding random "thermal" motion.
    const float centralFraction = 0.6f;  // share of mass in the central body
    const float dispersion = 0.10f;      // random velocity, as a fraction of v_circular

    using namespace detail;
    std::uniform_real_distribution<float> U(0.f, 1.f);
    std::normal_distribution<float> Ngauss(0.f, 1.f);

    const float centralMass = centralFraction * totalMass;
    const float diskMass = totalMass - centralMass;
    const float starMass = diskMass / (count - 1);
    const float rMin = 0.08f * radius;

    // Two axes spanning the disk plane (perpendicular to `normal`)
    normal = normalize(normal);
    float3 helper = std::fabs(normal.y) < 0.9f ? float3{0, 1, 0} : float3{1, 0, 0};
    float3 e1 = normalize(cross(normal, helper));
    float3 e2 = cross(normal, e1);

    // central body
    s.pos.push_back({center.x, center.y, center.z, centralMass});
    s.vel.push_back({bulkVel.x, bulkVel.y, bulkVel.z, 0});
    s.color.push_back({1.f, 0.95f, 0.85f, 0});

    for (int k = 1; k < count; k++) {
        // r uniform in [rMin, R] -> denser toward the center, like a real disk
        float u = U(rng);
        float r = rMin + (radius - rMin) * u;
        float theta = 6.2831853f * U(rng);
        float h = 0.03f * radius * Ngauss(rng);  // disk thickness

        float3 radial = add(mul(e1, std::cos(theta)), mul(e2, std::sin(theta)));
        float3 tangent = cross(normal, radial);
        float3 p = add(add(center, mul(radial, r)), mul(normal, h));

        // mass inside r (uniform-in-r disk => enclosed disk mass grows linearly)
        float menc = centralMass + diskMass * u;
        float r2 = r * r + softening * softening;
        float vCirc = std::sqrt(menc * r * r / (r2 * std::sqrt(r2)));
        float3 jitter = {Ngauss(rng), Ngauss(rng), Ngauss(rng)};
        float3 v = add(add(bulkVel, mul(tangent, vCirc)), mul(jitter, dispersion * vCirc));

        float3 c = lerp(coreColor, edgeColor, std::pow(u, 0.6f));
        s.pos.push_back({p.x, p.y, p.z, starMass});
        s.vel.push_back({v.x, v.y, v.z, 0});
        s.color.push_back({c.x, c.y, c.z, 0});
    }
}

// Two disk galaxies on a close, bound encounter -> tidal tails and a merger.
inline Scene makeCollidingGalaxies(int N, float softening, unsigned seed = 42) {
    Scene s;
    std::mt19937 rng(seed);
    int nA = N / 2, nB = N - nA;
    addDiskGalaxy(s, nA, {-2.5f, 0.0f, 1.2f}, {0.30f, 0.0f, -0.04f}, {0.1f, 1.0f, 0.15f}, 1.0f,
                  1.0f, {1.0f, 0.9f, 0.75f}, {0.35f, 0.55f, 1.0f}, softening, rng);
    addDiskGalaxy(s, nB, {2.5f, 0.3f, -1.2f}, {-0.30f, 0.0f, 0.04f}, {0.6f, 0.7f, -0.4f}, 1.0f,
                  1.0f, {1.0f, 0.85f, 0.6f}, {1.0f, 0.45f, 0.25f}, softening, rng);
    s.cameraDistance = 9.f;
    return s;
}

// A cold, slowly rotating ball of stars that collapses under its own gravity
// (closer to the demo shown in the GPU Gems chapter).
inline Scene makeCollapsingSphere(int N, unsigned seed = 42) {
    using namespace detail;
    Scene s;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    const float mass = 1.f / N;
    while ((int)s.pos.size() < N) {
        float3 p = {U(rng), U(rng), U(rng)};
        float r2 = p.x * p.x + p.y * p.y + p.z * p.z;
        if (r2 > 1.f) continue;  // rejection-sample a unit ball
        p = mul(p, 2.f);
        float3 v = mul(cross({0, 1, 0}, p), 0.25f);  // slow spin around y
        float t = std::sqrt(r2);
        float3 c = lerp({1.f, 0.9f, 0.7f}, {0.4f, 0.6f, 1.f}, t);
        s.pos.push_back({p.x, p.y, p.z, mass});
        s.vel.push_back({v.x, v.y, v.z, 0});
        s.color.push_back({c.x, c.y, c.z, 0});
    }
    s.cameraDistance = 7.f;
    return s;
}

// Pad to a multiple of the CUDA block size with massless, invisible bodies:
// the chapter's kernel assumes N % blockSize == 0. Massless bodies exert no
// force (bj.w = 0), so physics is unchanged.
inline void padToMultiple(Scene& s, int multiple) {
    while (s.pos.size() % multiple) {
        s.pos.push_back({1e4f, 1e4f, 1e4f, 0.f});
        s.vel.push_back({0, 0, 0, 0});
        s.color.push_back({0, 0, 0, 0});
    }
}
