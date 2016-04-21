#include "common.h"
#include <cuda_runtime_api.h>
#include <stdint.h>

/*__global__ void batch_map_normalize_kernel(
    float *x,
    int num_channels,
    int batch_size,
    const float *norm)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
}

extern "C" void rembrandt_kernel_batch_map_normalize(
    float *x,
    int num_channels,
    int batch_size,
    const float *norm,
    cudaStream_t stream)
{
  //assert(num_channels <= 1024);
  int n = batch_size * 1024;
  batch_map_normalize_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(x, num_channels, batch_size, norm);
  CUDA_POST_KERNEL_CHECK;
}*/

__global__ void batch_blockmap256_flip_kernel(
    const float *src,
    int num_channels,
    int batch_size,
    float *dst)
{
  __shared__ float in_cache[256 + 8];
  __shared__ float out_cache[256 + 8];
  int j = threadIdx.x;
  int batch_idx = blockIdx.x;
  int idx = j + batch_idx * num_channels;
  if (batch_idx < batch_size) {
    if (j < num_channels) {
      in_cache[OFFSET_BANK(j)] = src[idx];
    }
    __syncthreads();
    if (j < num_channels) {
      out_cache[OFFSET_BANK(num_channels - j - 1)] = in_cache[OFFSET_BANK(j)];
    }
    __syncthreads();
    if (j < num_channels) {
      dst[idx] = out_cache[OFFSET_BANK(j)];
    }
  }
}

extern "C" void rembrandt_kernel_batch_blockmap256_flip(
    const float *src,
    int num_channels,
    int batch_size,
    float *dst,
    cudaStream_t stream)
{
  //assert(num_channels <= 256);
  int n = num_channels * batch_size;
  batch_blockmap256_flip_kernel<<<(n+256-1)/256, 256, 0, stream>>>(
      src,
      num_channels,
      batch_size,
      dst);
}
