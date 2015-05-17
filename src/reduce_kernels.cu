#include "common.h"
#include <cuda_runtime_api.h>

extern "C" void rembrandt_kernel_blockreduce_argmax_float(
    int n,
    const float *x,
    float *max_block,
    int *idx_block,
    cudaStream_t stream)
{
  assert(0 && "unimplemented!");
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
