#include "common.h"
#include <cuda_runtime_api.h>

__global__ void map_noop_kernel() {
  // Do nothing.
}

extern "C" void rembrandt_kernel_map_noop(int n, cudaStream_t stream) {
  map_noop_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>();
  CUDA_POST_KERNEL_CHECK;
}

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
  map_set_constant_float_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(n, x, c);
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
  map_set_constant_i32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(n, x, c);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_add_constant_float_kernel(
    float *x,
    int n,
    float c)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    x[i] += c;
  }
}

extern "C" void rembrandt_kernel_map_add_constant_float(
    float *x,
    int n,
    float c,
    cudaStream_t stream)
{
  map_add_constant_float_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(x, n, c);
  CUDA_POST_KERNEL_CHECK;
}
