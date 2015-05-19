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
  dim3 block_dim(CUDA_BLOCK_DIM_1D(n));
  dim3 grid_dim(CUDA_GRID_DIM_1D(n));
  map_set_constant_float_kernel<<<grid_dim, block_dim, 0, stream>>>(n, x, c);
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
  dim3 block_dim(CUDA_BLOCK_DIM_1D(n));
  dim3 grid_dim(CUDA_GRID_DIM_1D(n));
  map_set_constant_i32_kernel<<<grid_dim, block_dim, 0, stream>>>(n, x, c);
  CUDA_POST_KERNEL_CHECK;
}
