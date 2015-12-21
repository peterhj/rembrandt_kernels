#ifndef __REMBRANDT_KERNELS_CUDA_COMMON_H__
#define __REMBRANDT_KERNELS_CUDA_COMMON_H__

#include <assert.h>
#include <stdio.h>

//#if defined(__cplusplus)
//#endif

#define CUDA_POST_KERNEL_CHECK { cudaError_t code = cudaPeekAtLastError(); if (code != cudaSuccess) { fprintf(stderr, "FATAL: cuda error: %s\n", cudaGetErrorString(code)); fflush(NULL); assert(0); } }
//#define CUDA_POST_KERNEL_CHECK assert((cudaPeekAtLastError()) == cudaSuccess)

#define max(a, b) \
    ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
       _a > _b ? _a : _b; })

#define min(a, b) \
    ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
       _a < _b ? _a : _b; })

#define CUDA_BLOCK_DIM_1D(n) (min((n+32-1)/32, 1024))
#define CUDA_GRID_DIM_1D(n) ((n+1024-1)/1024)

#endif
