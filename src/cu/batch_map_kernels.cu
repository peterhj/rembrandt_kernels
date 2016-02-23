#include "common.h"
#include <cuda_runtime_api.h>
#include <stdint.h>

__global__ void batch_blockmap_normalize_kernel(
    float *xs,
    int len,
    int batch_size,
    const float *norm)
{
  int tid = threadIdx.x;
  int block = blockIdx.x;
  int i = tid + block * len;
  if (tid < len && block < batch_size) {
    float x_i = xs[i];
    float norm_i = norm[block];
    x_i = x_i / norm_i;
    xs[i] = x_i;
  }
}

extern "C" void rembrandt_kernel_batch_blockmap_normalize(
    float *xs,
    int num_channels,
    int batch_size,
    const float *norm,
    cudaStream_t stream)
{
  //assert(num_channels <= 1024);
  int n = batch_size * 1024;
  batch_blockmap_normalize_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(xs, num_channels, batch_size, norm);
  CUDA_POST_KERNEL_CHECK;
}

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

__global__ void batch_map_pos_mask_inplace(
    float *z,
    int n,
    const float *pos_mask)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    if (pos_mask[idx] > 0.0f) {
      // Do nothing.
    } else {
      z[idx] = 0.0f;
    }
  }
}

extern "C" void rembrandt_kernel_batch_map_pos_mask_inplace(
    float *z,
    int num_channels,
    int batch_size,
    const float *pos_mask,
    cudaStream_t stream)
{
  int n = num_channels * batch_size;
  batch_map_pos_mask_inplace<<<(n+1024-1)/1024, 1024, 0, stream>>>(z, n, pos_mask);
  CUDA_POST_KERNEL_CHECK;
}

/*__global__ void batch_map_sparsemask_inplace(
    float *z,
    int num_channels,
    int batch_size,
    int n,
    const float *sparsemask)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
}

extern "C" void rembrandt_kernel_batch_map_sparsemask_inplace(
    float *z,
    int num_channels,
    int batch_size,
    const float *sparsemask,
    cudaStream_t stream)
{
  int n = num_channels * batch_size;
  batch_map_sparsemask_inplace<<<(n+1024-1)/1024, 1024, 0, stream>>>(z, num_channels, batch_size, n, sparsemask);
}*/

__global__ void map_rect_inplace_kernel(
    float *z,
    int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float z_i = z[i];
    z[i] = fmaxf(0.0f, z_i);
  }
}

extern "C" void rembrandt_kernel_batch_map_rect_inplace(
    float *z,
    int num_channels,
    int batch_size,
    cudaStream_t stream)
{
  int n = num_channels * batch_size;
  map_rect_inplace_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(z, n);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_rect_backprop_inplace_kernel(
    const float *out_act,
    int n,
    float *out_delta)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float z_i = out_act[i];
    if (z_i > 0.0f) {
      // Do nothing.
    } else {
      out_delta[i] = 0.0f;
    }
  }
}

extern "C" void rembrandt_kernel_batch_map_rect_backprop_inplace(
    const float *out_act,
    int num_channels,
    int batch_size,
    float *out_delta,
    cudaStream_t stream)
{
  int n = num_channels * batch_size;
  map_rect_backprop_inplace_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
        out_act, n, out_delta);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_bounded_rect_inplace_kernel(
    float *z,
    int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float z_i = z[i];
    z[i] = fminf(1.0f, fmaxf(0.0f, z_i));
  }
}

extern "C" void rembrandt_kernel_batch_map_bounded_rect_inplace(
    float *z,
    int num_channels,
    int batch_size,
    cudaStream_t stream)
{
  int n = num_channels * batch_size;
  map_bounded_rect_inplace_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(z, n);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_bounded_rect_backprop_inplace_kernel(
    const float *out_act,
    int n,
    float *out_delta)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float z_i = out_act[i];
    if (z_i > 0.0f && z_i < 1.0f) {
      // Do nothing.
    } else {
      out_delta[i] = 0.0f;
    }
  }
}

extern "C" void rembrandt_kernel_batch_map_bounded_rect_backprop_inplace(
    const float *out_act,
    int num_channels,
    int batch_size,
    float *out_delta,
    cudaStream_t stream)
{
  int n = num_channels * batch_size;
  map_bounded_rect_backprop_inplace_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
        out_act, n, out_delta);
  CUDA_POST_KERNEL_CHECK;
}

/*__global__ void batch_map_boltzmann_q_transform(
    const float *probs,
    int n,
    float inv_beta,
    float *qvalues)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    qvalues[i] = inv_beta * logf(probs[i]);
  }
}

extern "C" void rembrandt_kernel_batch_map_boltzmann_q_transform(
    const float *probs,
    int num_channels,
    int batch_size,
    float beta,
    float *qvalues,
    cudaStream_t stream)
{
  int n = num_channels * batch_size;
  float inv_beta = 1.0 / beta;
  batch_map_boltzmann_q_transform<<<(n+1024-1)/1024, 1024, 0, stream>>>(probs, n, inv_beta, qvalues);
  CUDA_POST_KERNEL_CHECK;
}*/

__global__ void batch_map_softmax_cross_entropy_loss_kernel(
    const float *probs,
    int num_channels,
    int batch_size,
    const int32_t *labels,
    float *loss1,
    float minibatch_size)
{
  int batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (batch_idx < batch_size) {
    int j = labels[batch_idx];
    int idx = j + batch_idx * num_channels;
    loss1[batch_idx] = -logf(probs[idx]) / minibatch_size;
  }
}

extern "C" void rembrandt_kernel_batch_map_softmax_cross_entropy_loss(
    const float *probs,
    int num_channels,
    int batch_size,
    const int32_t *labels,
    float *loss1,
    float minibatch_size,
    cudaStream_t stream)
{
  batch_map_softmax_cross_entropy_loss_kernel<<<(batch_size+1024-1), 1024, 0, stream>>>(
      probs, num_channels, batch_size, labels, loss1, minibatch_size);
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

__global__ void batch_map_softmax_kl_diverence_loss_backward_kernel(
    const float *logits,
    int n,
    int num_channels,
    int batch_size,
    const int32_t *labels,
    float *delta,
    float scale)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int batch_idx = idx / num_channels;
  int j = idx % num_channels;
  if (idx < n) {
    float y_i = logits[idx];
    if (j == labels[batch_idx]) {
      delta[idx] = scale * (y_i - 1.0f);
    } else {
      delta[idx] = scale * y_i;
    }
  }
}

extern "C" void rembrandt_kernel_batch_map_softmax_kl_diverence_loss_backward(
    const float *logits,
    int num_channels,
    int batch_size,
    const int32_t *labels,
    float *delta,
    float scale,
    cudaStream_t stream)
{
  int n = num_channels * batch_size;
  batch_map_softmax_kl_diverence_loss_backward_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(logits, n, num_channels, batch_size, labels, delta, scale);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void batch_map_multi_bin_logistic_kernel(
    const float *in_act,
    int n,
    float *out_act)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    out_act[idx] = 1.0f / (1.0f + expf(-in_act[idx]));
  }
}

extern "C" void rembrandt_kernel_batch_map_multi_bin_logistic(
    const float *in_act,
    int num_channels,
    int batch_size,
    float *out_act,
    cudaStream_t stream)
{
  int n = num_channels * batch_size;
  batch_map_multi_bin_logistic_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(in_act, n, out_act);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void batch_map_multi_bin_logistic_xent_loss_backprop_kernel(
    const float *out_act,
    int num_channels,
    int batch_size,
    const int32_t *cat_labels,
    const int32_t *bin_labels,
    float *in_delta,
    float minibatch_size)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int n = num_channels * batch_size;
  int batch_idx = idx / num_channels;
  int i = idx % num_channels;
  if (idx < n) {
    float y_i = out_act[idx];
    float b_i = (float)(bin_labels[idx]);
    if (i == cat_labels[batch_idx]) {
      in_delta[idx] = (y_i - b_i) / minibatch_size;
    } else {
      in_delta[idx] = 0.0f;
    }
  }
}

extern "C" void rembrandt_kernel_batch_map_multi_bin_logistic_xent_loss_backprop(
    const float *out_act,
    int num_channels,
    int batch_size,
    const int32_t *cat_labels,
    const int32_t *bin_labels,
    float *in_delta,
    float *maybe_loss,
    float minibatch_size,
    cudaStream_t stream)
{
  int n = num_channels * batch_size;
  batch_map_multi_bin_logistic_xent_loss_backprop_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(out_act, num_channels, batch_size, cat_labels, bin_labels, in_delta, minibatch_size);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void batch_map_softmax_kl_loss(
    const float *out_act,
    int num_channels,
    int batch_size,
    const int32_t *labels,
    const float *weights,
    float *loss1)
{
  int batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (batch_idx < batch_size) {
    int j = labels[batch_idx];
    int idx = j + batch_idx * num_channels;
    float z = out_act[idx];
    float w = weights[batch_idx];
    loss1[batch_idx] = -w * logf(z);
  }
}

extern "C" void rembrandt_kernel_batch_map_softmax_kl_loss(
    const float *out_act,
    int num_channels,
    int batch_size,
    const int32_t *labels,
    const float *weights,
    float *loss1,
    cudaStream_t stream)
{
  int n = batch_size;
  batch_map_softmax_kl_loss<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      out_act,
      num_channels,
      batch_size,
      labels,
      weights,
      loss1);
}

__global__ void batch_map_softmax_kl_backward(
    const float *out_act,
    int num_channels,
    int batch_size,
    const int32_t *labels,
    const float *weights,
    float *in_delta)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int n = num_channels * batch_size;
  int batch_idx = idx / num_channels;
  int j = idx % num_channels;
  if (idx < n) {
    int label = labels[batch_idx];
    float z = out_act[idx];
    float w = weights[batch_idx];
    if (j == label) {
      in_delta[idx] = w * (z - 1.0f);
    } else {
      in_delta[idx] = w * z;
    }
  }
}

extern "C" void rembrandt_kernel_batch_map_softmax_kl_backward(
    const float *out_act,
    int num_channels,
    int batch_size,
    const int32_t *labels,
    const float *weights,
    float *in_delta,
    cudaStream_t stream)
{
  int n = num_channels * batch_size;
  batch_map_softmax_kl_backward<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      out_act,
      num_channels,
      batch_size,
      labels,
      weights,
      in_delta);
}
