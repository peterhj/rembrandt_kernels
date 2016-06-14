#include "common.h"
#include <cuda_runtime_api.h>
#include <stdint.h>

__global__ void conv_diag_affine_fwd_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *scale,
    const float *bias,
    float *out_act)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int u = idx % spatial_dim;
  int c = (idx / spatial_dim) % num_channels;
  int batch_idx = idx / (spatial_dim * num_channels);
  if (u < spatial_dim && c < num_channels && batch_idx < batch_size) {
    float gamma = scale[c];
    float beta = bias[c];
    float y = gamma * in_act[idx] + beta;
    out_act[idx] = y;
  }
}

extern "C" void rembrandt_kernel_conv_diag_affine_fwd_batch(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *scale,
    const float *bias,
    float *out_act,
    cudaStream_t stream)
{
  int n = spatial_dim * num_channels * batch_size;
  conv_diag_affine_fwd_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, scale, bias, out_act);
}

extern "C" void rembrandt_kernel_conv_diag_affine_fwd_inplace_batch(
    float *out_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *scale,
    const float *bias,
    cudaStream_t stream)
{
  int n = spatial_dim * num_channels * batch_size;
  conv_diag_affine_fwd_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      out_act, spatial_dim, num_channels, batch_size, scale, bias, out_act);
}

__global__ void conv_diag_affine_bwd_batch_kernel(
    const float *in_act,
    const float *out_delta,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *scale,
    float *scale_grad,
    float *bias_grad,
    float *in_delta)
{
  __shared__ float scale_grad_cache[1024+32];
  __shared__ float bias_grad_cache[1024+32];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int bank_idx = OFFSET_BANK(threadIdx.x);
  int block_spatial_dim = (spatial_dim+4*32-1)/(4*32);
  int c = (idx / 32) % num_channels;
  int u0 = ((idx / (32 * num_channels)) % block_spatial_dim) * (4*32);
  int batch_idx = idx / (32 * num_channels * block_spatial_dim);
  scale_grad_cache[bank_idx] = 0.0f;
  bias_grad_cache[bank_idx] = 0.0f;
  if (c < num_channels && u0 < spatial_dim && batch_idx < batch_size) {
    float gamma = scale[c];
    float d_gamma = 0.0f;
    float d_beta = 0.0f;
    int i0 = c * spatial_dim + batch_idx * spatial_dim * num_channels;
    int u_limit = min(spatial_dim, u0 + 4*32);
    for (int u = u0; u < u_limit; u += 32) {
      int i = i0 + u;
      float dy = out_delta[i];
      d_gamma += dy * in_act[i];
      d_beta += dy;
      in_delta[i] += dy * gamma;
    }
    scale_grad_cache[bank_idx] = d_gamma;
    bias_grad_cache[bank_idx] = d_beta;
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (bank_idx % 2 == 0) {
      scale_grad_cache[bank_idx] += scale_grad_cache[bank_idx+1];
      bias_grad_cache[bank_idx] += bias_grad_cache[bank_idx+1];
    }
    __syncthreads();
    if (bank_idx % 4 == 0) {
      scale_grad_cache[bank_idx] += scale_grad_cache[bank_idx+2];
      bias_grad_cache[bank_idx] += bias_grad_cache[bank_idx+2];
    }
    __syncthreads();
    if (bank_idx % 8 == 0) {
      scale_grad_cache[bank_idx] += scale_grad_cache[bank_idx+4];
      bias_grad_cache[bank_idx] += bias_grad_cache[bank_idx+4];
    }
    __syncthreads();
    if (bank_idx % 16 == 0) {
      scale_grad_cache[bank_idx] += scale_grad_cache[bank_idx+8];
      bias_grad_cache[bank_idx] += bias_grad_cache[bank_idx+8];
    }
    __syncthreads();
    if (bank_idx % 32 == 0) {
      float d_gamma = scale_grad_cache[bank_idx] + scale_grad_cache[bank_idx+16];
      atomicAdd(&scale_grad[c], d_gamma);
      float d_beta = bias_grad_cache[bank_idx] + bias_grad_cache[bank_idx+16];
      atomicAdd(&bias_grad[c], d_beta);
    }
  }
}

__global__ void conv_diag_linear_bwd_batch_kernel(
    const float *in_act,
    const float *out_delta,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *scale,
    float *scale_grad,
    float *in_delta)
{
  __shared__ float scale_grad_cache[1024+32];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int bank_idx = OFFSET_BANK(threadIdx.x);
  int block_spatial_dim = (spatial_dim+4*32-1)/(4*32);
  int c = (idx / 32) % num_channels;
  int u0 = ((idx / (32 * num_channels)) % block_spatial_dim) * (4*32);
  int batch_idx = idx / (32 * num_channels * block_spatial_dim);
  scale_grad_cache[bank_idx] = 0.0f;
  if (c < num_channels && u0 < spatial_dim && batch_idx < batch_size) {
    float gamma = scale[c];
    float d_gamma = 0.0f;
    float d_beta = 0.0f;
    int i0 = c * spatial_dim + batch_idx * spatial_dim * num_channels;
    int u_limit = min(spatial_dim, u0 + 4*32);
    for (int u = u0; u < u_limit; u += 32) {
      int i = i0 + u;
      float dy = out_delta[i];
      d_gamma += dy * in_act[i];
      d_beta += dy;
      in_delta[i] += dy * gamma;
    }
    scale_grad_cache[bank_idx] = d_gamma;
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (bank_idx % 2 == 0) {
      scale_grad_cache[bank_idx] += scale_grad_cache[bank_idx+1];
    }
    __syncthreads();
    if (bank_idx % 4 == 0) {
      scale_grad_cache[bank_idx] += scale_grad_cache[bank_idx+2];
    }
    __syncthreads();
    if (bank_idx % 8 == 0) {
      scale_grad_cache[bank_idx] += scale_grad_cache[bank_idx+4];
    }
    __syncthreads();
    if (bank_idx % 16 == 0) {
      scale_grad_cache[bank_idx] += scale_grad_cache[bank_idx+8];
    }
    __syncthreads();
    if (bank_idx % 32 == 0) {
      float d_gamma = scale_grad_cache[bank_idx] + scale_grad_cache[bank_idx+16];
      atomicAdd(&scale_grad[c], d_gamma);
    }
  }
}
