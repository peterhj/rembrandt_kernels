#include "common.h"
#include <cuda_runtime_api.h>

__global__ void blockreduce_argmax_float_kernel(
    const int n,
    const float *x,
    float *x_max_block,
    //int *x_argmax_block)
    float *x_argmax_block)
{
  __shared__ float cache[1024];
  __shared__ int cache_idx[1024];
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  if (i < n) {
    cache[tid]      = x[i];
    cache_idx[tid]  = i;
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
    x_max_block[blockIdx.x] = cache[0];
    if (x_argmax_block != NULL) {
      //x_argmax_block[blockIdx.x] = cache_idx[0];
      x_argmax_block[blockIdx.x] = (float)(cache_idx[0]);
    }
  }
}

extern "C" void rembrandt_kernel_blockreduce_argmax_float(
    int n,
    const float *x,
    float *max_block,
    //int *idx_block,
    float *idx_block,
    cudaStream_t stream)
{
  blockreduce_argmax_float_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      n, x, max_block, idx_block);
  CUDA_POST_KERNEL_CHECK;
}

extern "C" void rembrandt_kernel_blockreduce_argmin_float(
    int n,
    const float *x,
    float *min_block,
    int *idx_block,
    cudaStream_t stream)
{
  assert(0 && "unimplemented!");
}
