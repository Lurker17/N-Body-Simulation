NVCC  ?= nvcc
FLAGS := -O3 -arch=sm_80 -std=c++17 -use_fast_math -Xcompiler -pthread

nbody: main.cu physics.cuh render.cuh scene.h video.h frame_pipeline.h
	$(NVCC) $(FLAGS) -o $@ main.cu

clean:
	rm -f nbody *.mp4

.PHONY: clean
