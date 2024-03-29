#include "common.h"
#include <cuda_runtime_api.h>
#include <stdint.h>

__global__ void batch_map_preproc_pca3_noise(
    const float *src,
    int spatial_size,
    int batch_size,
    const float *alphas,
    const float *evals,
    const float *evecs,
    float *dst)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int n = 3 * spatial_size * batch_size;
  int batch_idx = idx / (3 * spatial_size);
  int j = idx % (3 * spatial_size);
  int rgb = j / spatial_size;
  if (idx < n) {
    float x = src[idx];
    float y = x +
        alphas[3*batch_idx]   * evals[0] * evecs[rgb] +
        alphas[3*batch_idx+1] * evals[1] * evecs[rgb+3] +
        alphas[3*batch_idx+2] * evals[2] * evecs[rgb+6];
    dst[idx] = y;
  }
}

extern "C" void rembrandt_kernel_batch_map_preproc_pca3_noise(
    const float *src,
    int spatial_size,
    int batch_size,
    const float *alphas,
    const float *evals,
    const float *evecs,
    float *dst,
    cudaStream_t stream)
{
  int n = 3 * spatial_size * batch_size;
  batch_map_preproc_pca3_noise<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      src,
      spatial_size,
      batch_size,
      alphas,
      evals,
      evecs,
      dst);
}

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

__global__ void map_sigmoid_inplace_forward_kernel(
    float *y,
    int n,
    float beta)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float y_i = y[i];
    y[i] = beta / (1.0f + expf(-y_i));
  }
}

extern "C" void rembrandt_kernel_batch_map_sigmoid_inplace_forward(
    float *z,
    int num_channels,
    int batch_size,
    float beta,
    cudaStream_t stream)
{
  int n = num_channels * batch_size;
  map_sigmoid_inplace_forward_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(z, n, beta);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_sigmoid_inplace_backward_kernel(
    const float *out_act,
    int n,
    float *out_delta,
    float beta)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float y_i = out_act[i];
    float delta_i = out_delta[i];
    out_delta[i] = delta_i * y_i * (1.0f - y_i / beta);
  }
}

extern "C" void rembrandt_kernel_batch_map_sigmoid_inplace_backward(
    const float *out_act,
    int num_channels,
    int batch_size,
    float *out_delta,
    float beta,
    cudaStream_t stream)
{
  int n = num_channels * batch_size;
  map_sigmoid_inplace_backward_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
        out_act, n, out_delta, beta);
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

__global__ void batch_map_softmax_kl_divergence_loss_backward_kernel(
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

extern "C" void rembrandt_kernel_batch_map_softmax_kl_divergence_loss_backward(
    const float *logits,
    int num_channels,
    int batch_size,
    const int32_t *labels,
    float *delta,
    float scale,
    cudaStream_t stream)
{
  int n = num_channels * batch_size;
  batch_map_softmax_kl_divergence_loss_backward_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(logits, n, num_channels, batch_size, labels, delta, scale);
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

__global__ void batch_map_softmax_kl_loss1(
    const float *out_act,
    int num_channels,
    int batch_size,
    const int32_t *labels,
    const float *weights,
    float loss_factor,
    float *loss1)
{
  int batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (batch_idx < batch_size) {
    int j = labels[batch_idx];
    int idx = j + batch_idx * num_channels;
    float z = out_act[idx];
    float w = weights[batch_idx];
    float src_loss = loss1[batch_idx];
    loss1[batch_idx] = -w * logf(z) + loss_factor * src_loss;
  }
}

extern "C" void rembrandt_kernel_batch_map_softmax_kl_loss1(
    const float *out_act,
    int num_channels,
    int batch_size,
    const int32_t *labels,
    const float *weights,
    float loss_factor,
    float *loss1,
    cudaStream_t stream)
{
  int n = batch_size;
  batch_map_softmax_kl_loss1<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      out_act,
      num_channels,
      batch_size,
      labels,
      weights,
      loss_factor,
      loss1);
}

__global__ void batch_map_softmax_ind_backward(
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
      in_delta[idx] = w * z * (1.0f - z);
    } else {
      in_delta[idx] = w * z * (-z);
    }
  }
}

extern "C" void rembrandt_kernel_batch_map_softmax_ind_backward(
    const float *out_act,
    int num_channels,
    int batch_size,
    const int32_t *labels,
    const float *weights,
    float *in_delta,
    cudaStream_t stream)
{
  int n = num_channels * batch_size;
  batch_map_softmax_ind_backward<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      out_act,
      num_channels,
      batch_size,
      labels,
      weights,
      in_delta);
}

__global__ void batch_map_marginalized_softmax_ind_backward(
    const float *out_act,
    int num_channels,
    int batch_size,
    const float *weights,
    const float *cat_weights,
    float *in_delta)
{
  __shared__ float z_cache[1024 + 32];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int n = num_channels * batch_size;
  int batch_idx = idx / num_channels;
  int j = idx % num_channels;
  if (idx < n) {
    float z_j = out_act[idx];
    float w = weights[batch_idx];
    z_cache[j] = z_j;
    __syncthreads();
    int batch_offset = batch_idx * num_channels;
    float delta = 0.0f;
    for (int k = 0; k < num_channels; k++) {
      float cw_k = cat_weights[batch_offset + k];
      float z_k = z_cache[OFFSET_BANK(k)];
      if (k == j) {
        delta += cw_k * z_k * (1.0f - z_j);
      } else {
        delta += cw_k * z_k * (-z_j);
      }
    }
    in_delta[idx] = w * delta;
  }
}

extern "C" void rembrandt_kernel_batch_map_marginalized_softmax_ind_backward(
    const float *out_act,
    int num_channels,
    int batch_size,
    const float *weights,
    const float *cat_weights,
    float *in_delta,
    cudaStream_t stream)
{
  int n = num_channels * batch_size;
  batch_map_marginalized_softmax_ind_backward<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      out_act,
      num_channels,
      batch_size,
      weights,
      cat_weights,
      in_delta);
}

__global__ void batch_map_logistic_forward(
    const float *in_values,
    int num_channels,
    int batch_size,
    float *out_values)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int n = num_channels * batch_size;
  //int batch_idx = idx / num_channels;
  if (idx < n) {
    float x_i = in_values[idx];
    float y_i = 1.0f / (1.0f + expf(-x_i));
    out_values[idx] = y_i;
  }
}

extern "C" void rembrandt_kernel_batch_map_logistic_forward(
    const float *in_values,
    int num_channels,
    int batch_size,
    float *out_values,
    cudaStream_t stream)
{
  int n = num_channels * batch_size;
  batch_map_logistic_forward<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_values,
      num_channels,
      batch_size,
      out_values);
}

__global__ void batch_map_logistic_ind_backward(
    const float *out_values,
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
    if (j == label) {
      float w = weights[batch_idx];
      float y_i = out_values[idx];
      float delta_i = w * y_i * (1.0f - y_i);
      in_delta[idx] = delta_i;
    } else {
      in_delta[idx] = 0.0f;
    }
  }
}

extern "C" void rembrandt_kernel_batch_map_logistic_ind_backward(
    const float *out_values,
    int num_channels,
    int batch_size,
    const int32_t *labels,
    const float *weights,
    float *in_delta,
    cudaStream_t stream)
{
  int n = num_channels * batch_size;
  batch_map_logistic_ind_backward<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      out_values,
      num_channels,
      batch_size,
      labels,
      weights,
      in_delta);
}

__global__ void batch_map_antilogistic_forward(
    const float *in_values,
    int num_channels,
    int batch_size,
    const float *logit_sums,
    float *out_values)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int n = num_channels * batch_size;
  int batch_idx = idx / num_channels;
  //int j = idx % num_channels;
  if (idx < n) {
    float logit_sum = logit_sums[batch_idx];
    float x_i = in_values[idx];
    float y_i = 1.0f / (1.0f + expf(-2.0f * x_i + logit_sum));
    out_values[idx] = y_i;
  }
}

extern "C" void rembrandt_kernel_batch_map_antilogistic_forward(
    const float *in_values,
    int num_channels,
    int batch_size,
    const float *logit_sums,
    float *out_values,
    cudaStream_t stream)
{
  int n = num_channels * batch_size;
  batch_map_antilogistic_forward<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_values,
      num_channels,
      batch_size,
      logit_sums,
      out_values);
}

__global__ void batch_map_antilogistic_kl_backward(
    const float *out_values,
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
    float direction = -1.0f;
    if (j == label) {
      direction = 1.0f;
    }
    float w = weights[batch_idx];
    float y_i = out_values[idx];
    float y_truth = out_values[label + batch_idx * num_channels];
    float delta_i = -direction * w * y_i * (1.0f - y_i) / y_truth;
    in_delta[idx] = delta_i;
  }
}

extern "C" void rembrandt_kernel_batch_map_antilogistic_kl_backward(
    const float *out_values,
    int num_channels,
    int batch_size,
    const int32_t *labels,
    const float *weights,
    float *in_delta,
    cudaStream_t stream)
{
  int n = num_channels * batch_size;
  batch_map_antilogistic_kl_backward<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      out_values,
      num_channels,
      batch_size,
      labels,
      weights,
      in_delta);
}
