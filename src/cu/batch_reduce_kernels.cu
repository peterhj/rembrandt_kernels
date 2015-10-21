#include "common.h"
#include <cuda_runtime_api.h>
#include <stdint.h>

__global__ void batch_blockreduce_argmax_kernel(
    const float *xs,
    int len,
    int batch_size,
    int n,
    float *x_max_block,
    int *x_argmax_block)
{
  __shared__ float cache[1024];
  __shared__ int cache_idx[1024];
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  int block = blockIdx.x;
  if (i < n) {
    cache[tid]      = xs[i];
    cache_idx[tid]  = tid;
  }
  __syncthreads();
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0 && (i + s) < n && cache[tid] < cache[tid + s]) {
      cache[tid]      = cache[tid + s];
      cache_idx[tid]  = cache_idx[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    x_max_block[block] = cache[0];
    if (x_argmax_block != NULL) {
      x_argmax_block[block] = cache_idx[0];
    }
  }
}

extern "C" void rembrandt_kernel_batch_blockreduce_argmax(
    const float *xs,
    int len,
    int batch_size,
    float *xs_max,
    int32_t *xs_idx,
    cudaStream_t stream)
{
  // XXX: assert(len <= 1024);
  int n = len * batch_size;
  batch_blockreduce_argmax_kernel<<<batch_size, len, 0, stream>>>(
      xs, len, batch_size, n, xs_max, xs_idx);
  CUDA_POST_KERNEL_CHECK;
}
