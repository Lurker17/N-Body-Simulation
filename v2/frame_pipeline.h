// frame_pipeline.h — overlap GPU work with copying and encoding frames.
//
// v1 did, per frame:  [physics][render][copy to host][fwrite to ffmpeg]  — all serial,
// so the GPU sat idle during the copy and the fwrite.
//
// v2 keeps a ring of SLOTS pinned host buffers. The main thread only enqueues work:
//
//   main thread:    acquire slot -> launch physics + render -> async copy into slot
//                   -> record "ready" event -> hand slot to writer -> next frame
//   writer thread:  wait for slot's "ready" event -> fwrite to ffmpeg -> release slot
//
// The main thread blocks only when all slots are in flight, i.e. when the encoder
// is the bottleneck. Physics for frame k+1 runs on the GPU while frame k is being
// copied and encoded.
//
// All GPU work stays on one stream, so the order physics(k) -> render(k) -> copy(k)
// -> physics(k+1) is guaranteed without extra synchronization.
#pragma once
#include <condition_variable>
#include <cstdint>
#include <cuda_runtime.h>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "video.h"

struct FrameSlot {
    uint8_t* host = nullptr;        // pinned, width*height*3 bytes
    cudaEvent_t physStart, physEnd; // timing of this frame's physics steps
    cudaEvent_t ready;              // copy into `host` has finished
    bool timed = false;             // physStart/physEnd hold a measurement not yet read
};

class FramePipeline {
public:
    FramePipeline(VideoWriter& video, int slots) : video_(video), slots_(slots) {
        size_t bytes = (size_t)video.width * video.height * 3;
        for (int s = 0; s < slots; s++) {
            cudaMallocHost(&slots_[s].host, bytes);
            cudaEventCreate(&slots_[s].physStart);
            cudaEventCreate(&slots_[s].physEnd);
            cudaEventCreateWithFlags(&slots_[s].ready, cudaEventBlockingSync);
            free_.push(s);
        }
        writer_ = std::thread([this] { writerLoop(); });
    }

    ~FramePipeline() {
        for (auto& s : slots_) {
            cudaFreeHost(s.host);
            cudaEventDestroy(s.physStart);
            cudaEventDestroy(s.physEnd);
            cudaEventDestroy(s.ready);
        }
    }

    // Wait for a free slot. Returns its index.
    int acquire() {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return !free_.empty(); });
        int s = free_.front();
        free_.pop();
        return s;
    }

    FrameSlot& slot(int s) { return slots_[s]; }

    // Enqueue the async copy of the finished image and hand the slot to the writer.
    void submit(int s, const void* dImage) {
        FrameSlot& fs = slots_[s];
        cudaMemcpyAsync(fs.host, dImage, (size_t)video_.width * video_.height * 3,
                        cudaMemcpyDeviceToHost);
        cudaEventRecord(fs.ready);
        {
            std::lock_guard<std::mutex> lk(m_);
            filled_.push(s);
        }
        cv_.notify_all();
    }

    // Wait until every submitted frame is written, then stop the writer thread.
    void finish() {
        {
            std::lock_guard<std::mutex> lk(m_);
            done_ = true;
        }
        cv_.notify_all();
        writer_.join();
    }

    std::vector<FrameSlot>& slots() { return slots_; }

private:
    void writerLoop() {
        for (;;) {
            int s;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return !filled_.empty() || done_; });
                if (filled_.empty()) return;  // done_ and nothing left
                s = filled_.front();
                filled_.pop();
            }
            cudaEventSynchronize(slots_[s].ready);  // GPU copy finished
            writeFrame(video_, slots_[s].host);      // may block if ffmpeg is busy
            {
                std::lock_guard<std::mutex> lk(m_);
                free_.push(s);
            }
            cv_.notify_all();
        }
    }

    VideoWriter& video_;
    std::vector<FrameSlot> slots_;
    std::queue<int> free_, filled_;
    std::mutex m_;
    std::condition_variable cv_;
    std::thread writer_;
    bool done_ = false;
};
