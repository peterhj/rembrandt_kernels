#include "common.h"
#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <stdint.h>

#define OFFSET_BANK(idx) ({ __typeof__ (idx) _idx = idx; ((_idx) + ((_idx) / 32)); })

__global__ void batch_blocksort_bitonic_argrevsort_kernel(
    const float *xs,
    int len,
    int batch_size,
    float *xs_value_block,
    int32_t *xs_index_block)
{
  // XXX: See:
  // <https://graphics.cg.uni-saarland.de/fileadmin/cguds/courses/ws1213/pp_cuda/slides/06_-_Sorting_in_Parallel.pdf>
  __shared__ float cache_val[1024 + 32];
  __shared__ int32_t cache_idx[1024 + 32];
  int tid = threadIdx.x;
  int tid_offset = OFFSET_BANK(tid);
  int i = tid + blockIdx.x * len;
  if (tid < len) {
    cache_val[tid_offset] = xs[i];
    cache_idx[tid_offset] = tid;
  } else {
    cache_val[tid_offset] = -CUDART_INF_F;
    cache_idx[tid_offset] = -1;
  }
  __syncthreads();
  if (tid < len) {
    for (int k = 2; k <= blockDim.x; k *= 2) {
      for (int j = k / 2; j >= 1; j /= 2) {
        int ixj = tid ^ j;
        int ixj_offset = OFFSET_BANK(ixj);
        if (ixj > tid) {
          // XXX: Note that the comparisons are reversed (this is revsort).
          if ((tid & k) == 0) {
            if (cache_val[tid_offset] < cache_val[ixj_offset]) {
              float tmp_val = cache_val[tid_offset];
              int32_t tmp_idx = cache_idx[tid_offset];
              cache_val[tid_offset] = cache_val[ixj_offset];
              cache_idx[tid_offset] = cache_idx[ixj_offset];
              cache_val[ixj_offset] = tmp_val;
              cache_idx[ixj_offset] = tmp_idx;
            }
          } else {
            if (cache_val[tid_offset] > cache_val[ixj_offset]) {
              float tmp_val = cache_val[tid_offset];
              int32_t tmp_idx = cache_idx[tid_offset];
              cache_val[tid_offset] = cache_val[ixj_offset];
              cache_idx[tid_offset] = cache_idx[ixj_offset];
              cache_val[ixj_offset] = tmp_val;
              cache_idx[ixj_offset] = tmp_idx;
            }
          }
        }
        __syncthreads();
      }
    }
    xs_value_block[i] = cache_val[tid_offset];
    xs_index_block[i] = cache_idx[tid_offset];
  }
}

extern "C" void rembrandt_kernel_batch_blocksort_bitonic_argrevsort(
    const float *xs,
    int len,
    int batch_size,
    float *xs_value_block,
    int *xs_index_block,
    cudaStream_t stream)
{
  // XXX: assert(len <= 1024);
  int n = batch_size * 1024;
  batch_blocksort_bitonic_argrevsort_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      xs, len, batch_size, xs_value_block, xs_index_block);
  CUDA_POST_KERNEL_CHECK;
}
