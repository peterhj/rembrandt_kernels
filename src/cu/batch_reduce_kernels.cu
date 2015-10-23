#include "common.h"
#include <cuda_runtime_api.h>
#include <stdint.h>

#define OFFSET_BANK(idx) ({ __typeof__ (idx) _idx = idx; ((_idx) + ((_idx) / 32)); })

__global__ void batch_blockreduce_argmax_kernel(
    const float *xs,
    int len,
    int batch_size,
    float *x_max_block,
    int *x_argmax_block)
{
  __shared__ float cache[1024 + 32];
  __shared__ int cache_idx[1024 + 32];
  int tid = threadIdx.x;
  int block = blockIdx.x;
  int i = tid + block * len;
  if (tid < len && block < batch_size) {
    cache[OFFSET_BANK(tid)]     = xs[i];
    cache_idx[OFFSET_BANK(tid)] = tid;
    __syncthreads();
    for (int s = 1; s < blockDim.x; s *= 2) {
      if (tid % (2*s) == 0 && (tid + s) < len && cache[OFFSET_BANK(tid)] < cache[OFFSET_BANK(tid + s)]) {
        cache[OFFSET_BANK(tid)]     = cache[OFFSET_BANK(tid + s)];
        cache_idx[OFFSET_BANK(tid)] = cache_idx[OFFSET_BANK(tid + s)];
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
  // FIXME(20151022): could make more efficient use of blocks but w/e.
  int n = batch_size * 1024;
  batch_blockreduce_argmax_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      xs, len, batch_size, xs_max, xs_idx);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void batch_blockscan_prefix_sum_kernel(
    const float *xs,
    int len,
    int batch_size,
    float *xs_prefix_sum)
{
  // XXX: See:
  // <http://www.cs.cmu.edu/~guyb/papers/Ble93.pdf>
  // <http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html>
  __shared__ float cache[1024 + 32];
  int tid = threadIdx.x;
  int block = blockIdx.x;
  int i = tid + block * len;
  if (block < batch_size) {
    if (tid < len) {
      cache[OFFSET_BANK(tid)] = xs[i];
    } else {
      cache[OFFSET_BANK(tid)] = 0.0f;
    }
  }
  __syncthreads();
  if (block < batch_size) {
    for (int s = 1; s < blockDim.x; s *= 2) {
      if ((tid+1) % (2*s) == 0) {
        cache[OFFSET_BANK(tid + 2*s-1)] += cache[OFFSET_BANK(tid + s-1)];
      }
      __syncthreads();
    }
    cache[OFFSET_BANK(blockDim.x-1)] = 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s >= 1; s /= 2) {
      if ((tid+1) % (2*s) == 0) {
        float tmp = cache[OFFSET_BANK(tid + s-1)];
        cache[OFFSET_BANK(tid + s-1)] = cache[OFFSET_BANK(tid + 2*s-1)];
        cache[OFFSET_BANK(tid + 2*s-1)] += tmp;
      }
      __syncthreads();
    }
    if (tid < len) {
      xs_prefix_sum[i] = cache[OFFSET_BANK(tid)];
    }
  }
}

extern "C" void rembrandt_kernel_batch_blockscan_prefix_sum(
    const float *xs,
    int len,
    int batch_size,
    float *xs_prefix_sum,
    cudaStream_t stream)
{
  // XXX: assert(len <= 1024);
  // FIXME(20151022): could make more efficient use of blocks but w/e.
  int n = batch_size * 1024;
  batch_blockscan_prefix_sum_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      xs, len, batch_size, xs_prefix_sum);
  CUDA_POST_KERNEL_CHECK;
}
