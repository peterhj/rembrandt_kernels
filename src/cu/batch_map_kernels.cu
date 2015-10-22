#include "common.h"
#include <cuda_runtime_api.h>
#include <stdint.h>

__global__ void batch_map_zero_mask_inplace(
    float *z,
    int num_channels,
    int batch_size,
    int n,
    const float *zero_mask)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    if (zero_mask[idx] > 0.0f) {
      z[idx] = 0.0f;
    }
  }
}

extern "C" void rembrandt_kernel_batch_map_zero_mask_inplace(
    float *z,
    int num_channels,
    int batch_size,
    const float *zero_mask,
    cudaStream_t stream)
{
  int n = num_channels * batch_size;
  batch_map_zero_mask_inplace<<<(n+1024-1)/1024, 1024, 0, stream>>>(z, num_channels, batch_size, n, zero_mask);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void batch_map_softmax_cross_entropy_loss_backprop_kernel(
    const float *z,
    int num_channels,
    int batch_size,
    int n,
    const int32_t *labels,
    float *delta,
    float minibatch_size)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int batch_idx = idx / num_channels;
  int j = idx % num_channels;
  if (idx < n) {
    float z_i = z[idx];
    if (j == labels[batch_idx]) {
      delta[idx] = (z_i - 1.0f) / minibatch_size;
    } else {
      delta[idx] = z_i / minibatch_size;
    }
  }
}

extern "C" void rembrandt_kernel_batch_map_softmax_cross_entropy_loss_backprop(
    const float *z,
    int num_channels,
    int batch_size,
    const int32_t *labels,
    float *delta,
    float minibatch_size,
    cudaStream_t stream)
{
  int n = num_channels * batch_size;
  batch_map_softmax_cross_entropy_loss_backprop_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(z, num_channels, batch_size, n, labels, delta, minibatch_size);
  CUDA_POST_KERNEL_CHECK;
}
