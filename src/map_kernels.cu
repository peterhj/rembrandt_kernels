#include "common.h"
#include <cuda_runtime_api.h>

__global__ void map_set_constant_float_kernel(
    int n,
    float *x,
    float c)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    x[i] = c;
  }
}

extern "C" void rembrandt_kernel_map_set_constant_float(
    int n,
    float *x,
    float c,
    cudaStream_t stream)
{
  dim3 block_dim((n+1024-1)/1024);
  dim3 grid_dim(min(n, 1024));
  map_set_constant_float_kernel<<<block_dim, grid_dim, 0, stream>>>(n, x, c);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_set_constant_i32_kernel(
    int n,
    int *x,
    int c)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    x[i] = c;
  }
}

extern "C" void rembrandt_kernel_map_set_constant_i32(
    int n,
    int *x,
    int c,
    cudaStream_t stream)
{
  dim3 block_dim((n+1024-1)/1024);
  dim3 grid_dim(min(n, 1024));
  map_set_constant_i32_kernel<<<block_dim, grid_dim, 0, stream>>>(n, x, c);
  CUDA_POST_KERNEL_CHECK;
}

extern "C" void rembrandt_kernel_map_kahan_sum_update(
    int n,
    const float *x,
    float *y_sum,
    float *y_err,
    cudaStream_t stream)
{
  assert(0 && "unimplemented!");
}

extern "C" void rembrandt_kernel_map_kahan_sum_finish(
    int n,
    const float *y_sum,
    const float *y_err,
    float *s,
    cudaStream_t stream)
{
  assert(0 && "unimplemented!");
}
