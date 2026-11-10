NVCC  ?= nvcc
FLAGS := -O3 -arch=sm_80 -std=c++17 -use_fast_math

nbody: main.cu physics.cuh render.cuh scene.h video.h
	$(NVCC) $(FLAGS) -o $@ main.cu

clean:
	rm -f nbody *.mp4

.PHONY: clean
