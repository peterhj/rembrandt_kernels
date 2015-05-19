SRC_PREFIX ?= 
CUDA_HOME := /usr/local/cuda

CC := gcc-4.7
CC_FLAGS := -O3 -fPIC -std=gnu99 -I$(SRC_PREFIX) -I$(CUDA_HOME)/include -Wall -Werror

NVCC := $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS := -arch=sm_50 -O3 -Xcompiler '-fPIC' -I$(SRC_PREFIX) -I$(CUDA_HOME)/include

TARGET_LIB := librembrandt_kernels_cuda.a
OBJECTS := image_kernels.o map_kernels.o map_numerical_kernels.o reduce_kernels.o

.PHONY: all clean

%.o: $(SRC_PREFIX)%.c
	$(CC) $(CC_FLAGS) -c -o $@ $<

%.o: $(SRC_PREFIX)%.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

all: $(OBJECTS)
	ar cr $(TARGET_LIB) $(OBJECTS)

clean:
	rm -f $(TARGET_LIB) $(OBJECTS)
