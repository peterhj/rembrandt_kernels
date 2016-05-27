#include "common.h"
#include <cuda_runtime_api.h>
#include <stdio.h>

__global__ void div_map_inplace_kernel(
    float *xs,
    int n,
    const float *ys)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float x = xs[i] / ys[i];
    xs[i] = x;
  }
}

extern "C" void rembrandt_kernel_div_map_inplace(
    float *xs,
    int n,
    const float *ys,
    cudaStream_t stream)
{
  div_map_inplace_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(xs, n, ys);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void exp_map_kernel(
    const float *xs,
    int n,
    float *ys)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float x = xs[i];
    ys[i] = expf(x);
  }
}

extern "C" void rembrandt_kernel_exp_map(
    const float *xs,
    int n,
    float *ys,
    cudaStream_t stream)
{
  exp_map_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(xs, n, ys);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void scalar_sub_map_batch_inplace_kernel(
    float *xs,
    int frame_len,
    int batch_size,
    const float *scalars)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int i = idx % frame_len;
  int batch_idx = idx / frame_len;
  if ((i < frame_len) && (batch_idx < batch_size)) {
    float x = xs[idx] - scalars[batch_idx];
    xs[idx] = x;
  }
}

extern "C" void rembrandt_kernel_scalar_sub_map_batch_inplace(
    float *xs,
    int frame_len,
    int batch_size,
    const float *scalars,
    cudaStream_t stream)
{
  int n = frame_len * batch_size;
  scalar_sub_map_batch_inplace_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(xs, frame_len, batch_size, scalars);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void scalar_div_map_batch_kernel(
    const float *xs,
    int frame_len,
    int batch_size,
    const float *scalars,
    float *ys)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int i = idx % frame_len;
  int batch_idx = idx / frame_len;
  if ((i < frame_len) && (batch_idx < batch_size)) {
    float x = xs[idx] / scalars[batch_idx];
    ys[idx] = x;
  }
}

extern "C" void rembrandt_kernel_scalar_div_map_batch(
    const float *xs,
    int frame_len,
    int batch_size,
    const float *scalars,
    float *ys,
    cudaStream_t stream)
{
  int n = frame_len * batch_size;
  scalar_div_map_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(xs, frame_len, batch_size, scalars, ys);
  CUDA_POST_KERNEL_CHECK;
}
