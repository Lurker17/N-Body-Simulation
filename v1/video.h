// video.h — send raw RGB frames to ffmpeg through a pipe; ffmpeg writes the .mp4.
// No graphics here: a frame is just width*height*3 bytes, row by row, top to bottom.
#pragma once
#include <cstdio>
#include <cstdint>
#include <string>

struct VideoWriter {
    FILE* pipe = nullptr;
    int width = 0, height = 0;
};

inline bool openVideo(VideoWriter& v, const std::string& path, int w, int h, int fps) {
    v.width = w;
    v.height = h;
    std::string cmd = "ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgb24 -s " +
                      std::to_string(w) + "x" + std::to_string(h) + " -r " + std::to_string(fps) +
                      " -i - -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p "
                      "-movflags +faststart \"" + path + "\"";
    v.pipe = popen(cmd.c_str(), "w");
    return v.pipe != nullptr;
}

inline void writeFrame(VideoWriter& v, const uint8_t* rgb) {
    fwrite(rgb, 1, (size_t)v.width * v.height * 3, v.pipe);
}

inline void closeVideo(VideoWriter& v) {
    if (v.pipe) pclose(v.pipe);
    v.pipe = nullptr;
}
