#include "common.h"
#include <cuda_runtime_api.h>

__global__ void map_relu_activation_kernel(
    int n,
    float *x)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float x_i = x[i];
    x[i] = fmaxf(0.0, x_i);
  }
}

extern "C" void rembrandt_kernel_map_relu_activation(
    int n,
    float *x,
    cudaStream_t stream)
{
  /*dim3 block_dim(CUDA_BLOCK_DIM_1D(n));
  dim3 grid_dim(CUDA_GRID_DIM_1D(n));
  map_relu_activation_kernel<<<grid_dim, block_dim, 0, stream>>>(n, x);*/
  map_relu_activation_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(n, x);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_sigmoid_activation_kernel(
    int n,
    float *x)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float x_i = x[i];
    x[i] = 1.0 / (1.0 + expf(-x_i));
  }
}

extern "C" void rembrandt_kernel_map_sigmoid_activation(
    int n,
    float *x,
    cudaStream_t stream)
{
  /*dim3 block_dim(CUDA_BLOCK_DIM_1D(n));
  dim3 grid_dim(CUDA_GRID_DIM_1D(n));
  map_sigmoid_activation_kernel<<<grid_dim, block_dim, 0, stream>>>(n, x);*/
  map_sigmoid_activation_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(n, x);
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
