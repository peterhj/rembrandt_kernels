#include "common.h"
#include <cuda_runtime_api.h>
#include <stdint.h>

#define OFFSET_BANK(idx) ({ __typeof__ (idx) _idx = idx; ((_idx) + ((_idx) / 32)); })

__global__ void inner_prod_blockreduce_batch_kernel(
    const float *xs,
    int len,
    int batch_size,
    const float *ws,
    float alpha,
    float *sum)
{
  __shared__ float cache[1024 + 32];
  int tid = threadIdx.x;
  int block = blockIdx.x;
  int i = tid + block * len;
  if (tid < len && block < batch_size) {
    cache[OFFSET_BANK(tid)] = ws[i] * xs[i];
  } else {
    cache[OFFSET_BANK(tid)] = 0.0f;
  }
  __syncthreads();
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0 && (tid + s) < len) {
      cache[OFFSET_BANK(tid)] += cache[OFFSET_BANK(tid + s)];
    }
    __syncthreads();
  }
  if (tid == 0) {
    if (alpha != 0.0f) {
      float sum_0 = sum[block];
      sum[block] = alpha * sum_0 + cache[0];
    } else {
      sum[block] = cache[0];
    }
  }
}

extern "C" void rembrandt_kernel_inner_prod_blockreduce_batch(
    const float *xs,
    int len,
    int batch_size,
    const float *ws,
    float alpha,
    float *sum,
    cudaStream_t stream)
{
  // XXX: assert(len <= 1024);
  // FIXME(20151022): could make more efficient use of blocks but w/e.
  int n = batch_size * 1024;
  inner_prod_blockreduce_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      xs, len, batch_size, ws, alpha, sum);
  CUDA_POST_KERNEL_CHECK;
}
