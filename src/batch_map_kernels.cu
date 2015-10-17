#include "common.h"
#include <cuda_runtime_api.h>
#include <stdint.h>

__global__ void batch_map_softmax_cross_entropy_loss_backprop_kernel(
    const float *z,
    int num_channels,
    int batch_size,
    const int32_t *labels,
    float *delta,
    float minibatch_size)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int batch_idx = idx / num_channels;
  int i = idx % num_channels;
  if (batch_idx < batch_size) {
    float z_i = z[idx];
    if (i == labels[batch_idx]) {
      delta[idx] = (z_i - 1.0) / minibatch_size;
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
  batch_map_softmax_cross_entropy_loss_backprop_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(z, num_channels, batch_size, labels, delta, minibatch_size);
  CUDA_POST_KERNEL_CHECK;
}
