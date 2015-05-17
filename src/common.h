#ifndef __REMBRANDT_KERNELS_CUDA_COMMON_H__
#define __REMBRANDT_KERNELS_CUDA_COMMON_H__

#include <assert.h>

//#if defined(__cplusplus)
//#endif

#define CUDA_POST_KERNEL_CHECK assert((cudaPeekAtLastError()) == cudaSuccess)

#define max(a, b) \
    ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
       _a > _b ? _a : _b; })

#define min(a, b) \
    ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
       _a < _b ? _a : _b; })

#endif
