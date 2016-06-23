#include "common.h"
#include <cuda_runtime_api.h>
#include <stdint.h>

__global__ void conv_diag_affine_white_var_fwd_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *__restrict__ mean,
    const float *__restrict__ var,
    float epsilon,
    float *out_act)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int u = idx % spatial_dim;
  int c = (idx / spatial_dim) % num_channels;
  int batch_idx = idx / (spatial_dim * num_channels);
  if (u < spatial_dim && c < num_channels && batch_idx < batch_size) {
    float m = mean[c];
    float v = var[c];
    float y = (in_act[idx] - m) * rsqrtf(v + epsilon);
    out_act[idx] = y;
  }
}

extern "C" void rembrandt_conv_diag_affine_white_var_fwd_batch(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *mean,
    const float *var,
    float epsilon,
    float *out_act,
    cudaStream_t stream)
{
  int n = spatial_dim * num_channels * batch_size;
  conv_diag_affine_white_var_fwd_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, mean, var, epsilon, out_act);
}

__global__ void conv_diag_affine_white_fwd_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *__restrict__ mean,
    const float *__restrict__ istd,
    float *out_act)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int u = idx % spatial_dim;
  int c = (idx / spatial_dim) % num_channels;
  int batch_idx = idx / (spatial_dim * num_channels);
  if (u < spatial_dim && c < num_channels && batch_idx < batch_size) {
    float m = mean[c];
    float is = istd[c];
    float y = is * (in_act[idx] - m);
    out_act[idx] = y;
  }
}

extern "C" void rembrandt_conv_diag_affine_white_fwd_batch(
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
  conv_diag_affine_white_fwd_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, scale, bias, out_act);
}

__global__ void conv_diag_affine_fwd_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *__restrict__ scale,
    const float *__restrict__ bias,
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

extern "C" void rembrandt_conv_diag_affine_fwd_batch(
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

extern "C" void rembrandt_conv_diag_affine_fwd_inplace_batch(
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

__global__ void conv_diag_linear_fwd_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *__restrict__ scale,
    float *out_act)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int u = idx % spatial_dim;
  int c = (idx / spatial_dim) % num_channels;
  int batch_idx = idx / (spatial_dim * num_channels);
  if (u < spatial_dim && c < num_channels && batch_idx < batch_size) {
    float gamma = scale[c];
    float y = gamma * in_act[idx];
    out_act[idx] = y;
  }
}

extern "C" void rembrandt_conv_diag_linear_fwd_batch(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *scale,
    float *out_act,
    cudaStream_t stream)
{
  int n = spatial_dim * num_channels * batch_size;
  conv_diag_linear_fwd_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, scale, out_act);
}

__global__ void conv_diag_affine_bwd_data_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *out_delta,
    const float *__restrict__ scale,
    float *in_delta)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int warp_idx = idx % 32;
  int c = (idx / 32) % num_channels;
  int u0 = warp_idx + ((idx / (32 * num_channels)) % block_spatial_dim) * (16*32);
  int batch_idx = idx / (32 * num_channels * block_spatial_dim);
  if (c < num_channels && u0 < spatial_dim && batch_idx < batch_size) {
    float gamma = scale[c];
    int i0 = c * spatial_dim + batch_idx * spatial_dim * num_channels;
    int u_limit = min(spatial_dim, u0 + 16*32);
    for (int u = u0; u < u_limit; u += 32) {
      int i = i0 + u;
      float dy = out_delta[i];
      in_delta[i] = dy * gamma;
      //in_delta[i] += dy * gamma;
      //atomicAdd(&in_delta[i], dy * gamma);
    }
  }
}

extern "C" void rembrandt_conv_diag_affine_bwd_data_batch(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *out_delta,
    const float *scale,
    float *in_delta,
    cudaStream_t stream)
{
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int n = 32 * num_channels * block_spatial_dim * batch_size;
  conv_diag_affine_bwd_data_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, out_delta, scale, in_delta);
}

__global__ void conv_diag_affine_bwd_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *out_delta,
    const float *__restrict__ scale,
    float *scale_grad,
    float *bias_grad,
    float *in_delta)
{
  __shared__ float scale_grad_cache[1024+32];
  __shared__ float bias_grad_cache[1024+32];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int bank_idx = OFFSET_BANK(threadIdx.x);
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int warp_idx = idx % 32;
  int c = (idx / 32) % num_channels;
  int u0 = warp_idx + ((idx / (32 * num_channels)) % block_spatial_dim) * (16*32);
  int batch_idx = idx / (32 * num_channels * block_spatial_dim);
  if (c < num_channels && u0 < spatial_dim && batch_idx < batch_size) {
    float gamma = scale[c];
    float d_gamma = 0.0f;
    float d_beta = 0.0f;
    int i0 = c * spatial_dim + batch_idx * spatial_dim * num_channels;
    int u_limit = min(spatial_dim, u0 + 16*32);
    for (int u = u0; u < u_limit; u += 32) {
      int i = i0 + u;
      float dy = out_delta[i];
      d_gamma += dy * in_act[i];
      d_beta += dy;
      in_delta[i] = dy * gamma;
      //in_delta[i] += dy * gamma;
      //atomicAdd(&in_delta[i], dy * gamma);
    }
    scale_grad_cache[bank_idx] = d_gamma;
    bias_grad_cache[bank_idx] = d_beta;
  } else {
    scale_grad_cache[bank_idx] = 0.0f;
    bias_grad_cache[bank_idx] = 0.0f;
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 2 == 0) {
      scale_grad_cache[bank_idx] += scale_grad_cache[bank_idx+1];
      bias_grad_cache[bank_idx] += bias_grad_cache[bank_idx+1];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 4 == 0) {
      scale_grad_cache[bank_idx] += scale_grad_cache[bank_idx+2];
      bias_grad_cache[bank_idx] += bias_grad_cache[bank_idx+2];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 8 == 0) {
      scale_grad_cache[bank_idx] += scale_grad_cache[bank_idx+4];
      bias_grad_cache[bank_idx] += bias_grad_cache[bank_idx+4];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 16 == 0) {
      scale_grad_cache[bank_idx] += scale_grad_cache[bank_idx+8];
      bias_grad_cache[bank_idx] += bias_grad_cache[bank_idx+8];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 32 == 0 && u0 < spatial_dim) {
      float d_gamma = scale_grad_cache[bank_idx] + scale_grad_cache[bank_idx+16];
      atomicAdd(&scale_grad[c], d_gamma);
      float d_beta = bias_grad_cache[bank_idx] + bias_grad_cache[bank_idx+16];
      atomicAdd(&bias_grad[c], d_beta);
    }
  }
}

extern "C" void rembrandt_conv_diag_affine_bwd_batch(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *out_delta,
    const float *scale,
    float *scale_grad,
    float *bias_grad,
    float *in_delta,
    cudaStream_t stream)
{
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int n = 32 * num_channels * block_spatial_dim * batch_size;
  conv_diag_affine_bwd_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, out_delta, scale, scale_grad, bias_grad, in_delta);
}

__global__ void conv_diag_linear_bwd_batch_kernel(
    const float *in_act,
    const float *out_delta,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *__restrict__ scale,
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
  if (c < num_channels && u0 < spatial_dim && batch_idx < batch_size) {
    float gamma = scale[c];
    float d_gamma = 0.0f;
    int i0 = c * spatial_dim + batch_idx * spatial_dim * num_channels;
    int u_limit = min(spatial_dim, u0 + 4*32);
    for (int u = u0; u < u_limit; u += 32) {
      int i = i0 + u;
      float dy = out_delta[i];
      d_gamma += dy * in_act[i];
      in_delta[i] += dy * gamma;
    }
    scale_grad_cache[bank_idx] = d_gamma;
  } else {
    scale_grad_cache[bank_idx] = 0.0f;
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 2 == 0) {
      scale_grad_cache[bank_idx] += scale_grad_cache[bank_idx+1];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 4 == 0) {
      scale_grad_cache[bank_idx] += scale_grad_cache[bank_idx+2];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 8 == 0) {
      scale_grad_cache[bank_idx] += scale_grad_cache[bank_idx+4];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 16 == 0) {
      scale_grad_cache[bank_idx] += scale_grad_cache[bank_idx+8];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 32 == 0 && u0 < spatial_dim) {
      float d_gamma = scale_grad_cache[bank_idx] + scale_grad_cache[bank_idx+16];
      atomicAdd(&scale_grad[c], d_gamma);
    }
  }
}

__global__ void conv_bnorm_inferfwd_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *__restrict__ mean,
    const float *__restrict__ var,
    const float *__restrict__ scale,
    const float *__restrict__ bias,
    float epsilon,
    float *out_act)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int u = idx % spatial_dim;
  int c = (idx / spatial_dim) % num_channels;
  int batch_idx = idx / (spatial_dim * num_channels);
  if (u < spatial_dim && c < num_channels && batch_idx < batch_size) {
    float m = mean[c];
    float v = var[c];
    float s = scale[c];
    float b = bias[c];
    float y = (in_act[idx] - m) * rsqrtf(v + epsilon) * s + b;
    out_act[idx] = y;
  }
}

extern "C" void rembrandt_conv_bnorm_inferfwd_batch(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *mean,
    const float *var,
    const float *scale,
    const float *bias,
    float epsilon,
    float *out_act,
    cudaStream_t stream)
{
  int n = spatial_dim * num_channels * batch_size;
  conv_bnorm_inferfwd_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, mean, var, scale, bias, epsilon, out_act);
}

__global__ void conv_bnorm_bwd_var_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *out_delta,
    const float *__restrict__ mean,
    const float *__restrict__ var,
    float epsilon,
    float *var_grad)
{
  __shared__ float d_sigma_cache[1024+32];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int bank_idx = OFFSET_BANK(threadIdx.x);
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int warp_idx = idx % 32;
  int c = (idx / 32) % num_channels;
  int u0 = warp_idx + ((idx / (32 * num_channels)) % block_spatial_dim) * (16*32);
  int batch_idx = idx / (32 * num_channels * block_spatial_dim);
  if (c < num_channels && u0 < spatial_dim && batch_idx < batch_size) {
    float mu = mean[c];
    float sigma = var[c];
    float d_sigma = 0.0f;
    int i0 = c * spatial_dim + batch_idx * spatial_dim * num_channels;
    int u_limit = min(spatial_dim, u0 + 16*32);
    for (int u = u0; u < u_limit; u += 32) {
      int i = i0 + u;
      d_sigma += out_delta[i] * -0.5f * rsqrtf(sigma + epsilon) / (sigma + epsilon) * (in_act[i] - mu);
    }
    d_sigma_cache[bank_idx] = d_sigma;
  } else {
    d_sigma_cache[bank_idx] = 0.0f;
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 2 == 0) {
      d_sigma_cache[bank_idx] += d_sigma_cache[bank_idx+1];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 4 == 0) {
      d_sigma_cache[bank_idx] += d_sigma_cache[bank_idx+2];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 8 == 0) {
      d_sigma_cache[bank_idx] += d_sigma_cache[bank_idx+4];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 16 == 0) {
      d_sigma_cache[bank_idx] += d_sigma_cache[bank_idx+8];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 32 == 0 && u0 < spatial_dim) {
      float d_sigma = d_sigma_cache[bank_idx] + d_sigma_cache[bank_idx+16];
      atomicAdd(&var_grad[c], d_sigma);
    }
  }
}

__global__ void conv_bnorm_bwd_mean_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *out_delta,
    const float *__restrict__ mean,
    const float *__restrict__ var,
    const float *__restrict__ var_grad,
    float epsilon,
    float *mean_grad)
{
  __shared__ float d_mu_cache[1024+32];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int bank_idx = OFFSET_BANK(threadIdx.x);
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int warp_idx = idx % 32;
  int c = (idx / 32) % num_channels;
  int u0 = warp_idx + ((idx / (32 * num_channels)) % block_spatial_dim) * (16*32);
  int batch_idx = idx / (32 * num_channels * block_spatial_dim);
  if (c < num_channels && u0 < spatial_dim && batch_idx < batch_size) {
    float inv_var_norm = 1.0f / ((float)(spatial_dim - 1) * (float)(batch_size - 1));
    float mu = mean[c];
    float sigma = var[c];
    float d_sigma = var_grad[c];
    float d_mu = 0.0f;
    int i0 = c * spatial_dim + batch_idx * spatial_dim * num_channels;
    int u_limit = min(spatial_dim, u0 + 16*32);
    for (int u = u0; u < u_limit; u += 32) {
      int i = i0 + u;
      d_mu += out_delta[i] * -rsqrtf(sigma + epsilon) + d_sigma * -2.0f * inv_var_norm * (in_act[i] - mu);
    }
    d_mu_cache[bank_idx] = d_mu;
  } else {
    d_mu_cache[bank_idx] = 0.0f;
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 2 == 0) {
      d_mu_cache[bank_idx] += d_mu_cache[bank_idx+1];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 4 == 0) {
      d_mu_cache[bank_idx] += d_mu_cache[bank_idx+2];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 8 == 0) {
      d_mu_cache[bank_idx] += d_mu_cache[bank_idx+4];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 16 == 0) {
      d_mu_cache[bank_idx] += d_mu_cache[bank_idx+8];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 32 == 0 && u0 < spatial_dim) {
      float d_mu = d_mu_cache[bank_idx] + d_mu_cache[bank_idx+16];
      atomicAdd(&mean_grad[c], d_mu);
    }
  }
}

__global__ void conv_bnorm_bwd_data_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *out_delta,
    const float *__restrict__ mean,
    const float *__restrict__ mean_grad,
    const float *__restrict__ var,
    const float *__restrict__ var_grad,
    float epsilon,
    float *in_delta)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int bank_idx = OFFSET_BANK(threadIdx.x);
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int warp_idx = idx % 32;
  int c = (idx / 32) % num_channels;
  int u0 = warp_idx + ((idx / (32 * num_channels)) % block_spatial_dim) * (16*32);
  int batch_idx = idx / (32 * num_channels * block_spatial_dim);
  if (c < num_channels && u0 < spatial_dim && batch_idx < batch_size) {
    float inv_mean_norm = 1.0f / ((float)(spatial_dim) * (float)(batch_size));
    float inv_var_norm = 1.0f / ((float)(spatial_dim - 1) * (float)(batch_size - 1));
    float mu = mean[c];
    float d_mu = mean_grad[c];
    float sigma = var[c];
    float d_sigma = var_grad[c];
    int i0 = c * spatial_dim + batch_idx * spatial_dim * num_channels;
    int u_limit = min(spatial_dim, u0 + 16*32);
    for (int u = u0; u < u_limit; u += 32) {
      int i = i0 + u;
      in_delta[i] = out_delta[i] * rsqrtf(sigma + epsilon) + d_mu * inv_mean_norm + d_sigma * 2.0f * inv_var_norm * (in_act[i] - mu);
    }
  }
}

extern "C" void rembrandt_conv_bnorm_bwd_batch(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *out_delta,
    const float *mean,
    const float *var,
    float epsilon,
    float *mean_grad,
    float *var_grad,
    float *in_delta,
    cudaStream_t stream)
{
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int n = 32 * num_channels * block_spatial_dim * batch_size;
  conv_bnorm_bwd_var_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, out_delta, mean, var, epsilon, var_grad);
  conv_bnorm_bwd_mean_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, out_delta, mean, var, var_grad, epsilon, mean_grad);
  conv_bnorm_bwd_data_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, out_delta, mean, mean_grad, var, var_grad, epsilon, in_delta);
}

__global__ void conv_bnorm_rfwd_var_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *in_r_act,
    const float *__restrict__ mean,
    const float *__restrict__ r_mean,
    float *r_var)
{
  __shared__ float r_var_cache[1024+32];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int bank_idx = OFFSET_BANK(threadIdx.x);
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int warp_idx = idx % 32;
  int c = (idx / 32) % num_channels;
  int u0 = warp_idx + ((idx / (32 * num_channels)) % block_spatial_dim) * (16*32);
  int batch_idx = idx / (32 * num_channels * block_spatial_dim);
  if (c < num_channels && u0 < spatial_dim && batch_idx < batch_size) {
    float mean_c = mean[c];
    float r_mean_c = r_mean[c];
    float y = 0.0f;
    int i0 = c * spatial_dim + batch_idx * spatial_dim * num_channels;
    int u_limit = min(spatial_dim, u0 + 16*32);
    for (int u = u0; u < u_limit; u += 32) {
      int i = i0 + u;
      float delta = in_act[i] - mean_c;
      float r_delta = in_r_act[i] - r_mean_c;
      y += 2.0f * delta * r_delta;
    }
    r_var_cache[bank_idx] = y;
  } else {
    r_var_cache[bank_idx] = 0.0f;
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 2 == 0) {
      r_var_cache[bank_idx] += r_var_cache[bank_idx+1];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 4 == 0) {
      r_var_cache[bank_idx] += r_var_cache[bank_idx+2];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 8 == 0) {
      r_var_cache[bank_idx] += r_var_cache[bank_idx+4];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 16 == 0) {
      r_var_cache[bank_idx] += r_var_cache[bank_idx+8];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 32 == 0 && u0 < spatial_dim) {
      float y = (r_var_cache[bank_idx] + r_var_cache[bank_idx+16]) / ((float)(spatial_dim-1) * (float)(batch_size-1));
      atomicAdd(&r_var[c], y);
    }
  }
}

extern "C" void rembrandt_conv_bnorm_rfwd_var_batch(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *in_r_act,
    const float *mean,
    const float *r_mean,
    float *r_var,
    cudaStream_t stream)
{
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int n = 32 * num_channels * block_spatial_dim * batch_size;
  conv_bnorm_rfwd_var_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, in_r_act, mean, r_mean, r_var);
}

__global__ void conv_bnorm_rfwd_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *in_r_act,
    const float *__restrict__ mean,
    const float *__restrict__ r_mean,
    const float *__restrict__ var,
    const float *__restrict__ r_var,
    float epsilon,
    float *out_r_act)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int u = idx % spatial_dim;
  int c = (idx / spatial_dim) % num_channels;
  int batch_idx = idx / (spatial_dim * num_channels);
  if (u < spatial_dim && c < num_channels && batch_idx < batch_size) {
    float sigma = var[c];
    float y = rsqrtf(sigma + epsilon) * ((in_r_act[idx] - r_mean[c]) - 0.5f / (sigma + epsilon) * r_var[c] * (in_act[idx] - mean[c]));
    out_r_act[idx] = y;
  }
}

extern "C" void rembrandt_conv_bnorm_rfwd_batch(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *in_r_act,
    const float *mean,
    const float *r_mean,
    const float *var,
    const float *r_var,
    float epsilon,
    float *out_r_act,
    cudaStream_t stream)
{
  int n = spatial_dim * num_channels * batch_size;
  conv_bnorm_rfwd_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, in_r_act, mean, r_mean, var, r_var, epsilon, out_r_act);
}

__global__ void conv_bnorm_rbwd_var_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *in_r_act,
    const float *out_delta,
    const float *out_r_delta,
    const float *__restrict__ mean,
    const float *__restrict__ r_mean,
    const float *__restrict__ var,
    const float *__restrict__ r_var,
    float epsilon,
    float *var_r_grad)
{
}

__global__ void conv_smooth_bnorm_fwd_mean_var_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *__restrict__ prev_mean,
    float alpha,
    float *mean,
    float *var)
{
  __shared__ float mu_cache[1024];
  __shared__ float sigma_cache[1024];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int warp_idx = idx % 32;
  int c = (idx / 32) % num_channels;
  int u0 = warp_idx + ((idx / (32 * num_channels)) % block_spatial_dim) * (16*32);
  int batch_idx = idx / (32 * num_channels * block_spatial_dim);
  if (c < num_channels && u0 < spatial_dim && batch_idx < batch_size) {
    float prev_mu = prev_mean[c];
    float mu = 0.0f;
    float sigma = 0.0f;
    int i0 = c * spatial_dim + batch_idx * spatial_dim * num_channels;
    int u_limit = min(spatial_dim, u0 + 16*32);
    for (int u = u0; u < u_limit; u += 32) {
      int i = i0 + u;
      float x = in_act[i];
      float r = x - prev_mu;
      mu += x;
      sigma += r * r;
    }
    mu_cache[threadIdx.x] = mu;
    sigma_cache[threadIdx.x] = sigma;
  } else {
    mu_cache[threadIdx.x] = 0.0f;
    sigma_cache[threadIdx.x] = 0.0f;
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 2 == 0) {
      mu_cache[threadIdx.x] += mu_cache[threadIdx.x+1];
      sigma_cache[threadIdx.x] += sigma_cache[threadIdx.x+1];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 4 == 0) {
      mu_cache[threadIdx.x] += mu_cache[threadIdx.x+2];
      sigma_cache[threadIdx.x] += sigma_cache[threadIdx.x+2];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 8 == 0) {
      mu_cache[threadIdx.x] += mu_cache[threadIdx.x+4];
      sigma_cache[threadIdx.x] += sigma_cache[threadIdx.x+4];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 16 == 0) {
      mu_cache[threadIdx.x] += mu_cache[threadIdx.x+8];
      sigma_cache[threadIdx.x] += sigma_cache[threadIdx.x+8];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 32 == 0 && u0 < spatial_dim) {
      float mu = alpha * (mu_cache[threadIdx.x] + mu_cache[threadIdx.x+16]) / ((float)(spatial_dim) * (float)(batch_size));
      float sigma = alpha * (sigma_cache[threadIdx.x] + sigma_cache[threadIdx.x+16]) / ((float)(spatial_dim) * (float)(batch_size));
      atomicAdd(&mean[c], mu);
      atomicAdd(&var[c], sigma);
    }
  }
}

extern "C" void rembrandt_conv_smooth_bnorm_fwd_mean_var_batch(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *prev_mean,
    float alpha,
    float *mean,
    float *var,
    cudaStream_t stream)
{
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int n = 32 * num_channels * block_spatial_dim * batch_size;
  conv_smooth_bnorm_fwd_mean_var_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, prev_mean, alpha, mean, var);
}

__global__ void conv_smooth_bnorm_bwd_mean_var_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *out_delta,
    const float *__restrict__ mean,
    const float *__restrict__ var,
    float epsilon,
    float *mean_grad,
    float *var_grad)
{
  __shared__ float d_mu_cache[1024];
  __shared__ float d_sigma_cache[1024];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  //int bank_idx = OFFSET_BANK(threadIdx.x);
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int warp_idx = idx % 32;
  int c = (idx / 32) % num_channels;
  int u0 = warp_idx + ((idx / (32 * num_channels)) % block_spatial_dim) * (16*32);
  int batch_idx = idx / (32 * num_channels * block_spatial_dim);
  if (c < num_channels && u0 < spatial_dim && batch_idx < batch_size) {
    float mu = mean[c];
    float sigma = var[c];
    float invroot_sigma = rsqrtf(sigma + epsilon);
    float d_mu = 0.0f;
    float d_sigma = 0.0f;
    int i0 = c * spatial_dim + batch_idx * spatial_dim * num_channels;
    int u_limit = min(spatial_dim, u0 + 16*32);
    for (int u = u0; u < u_limit; u += 32) {
      int i = i0 + u;
      float dy = out_delta[i];
      d_mu += dy * -invroot_sigma;
      d_sigma += dy * -0.5f * invroot_sigma / (sigma + epsilon) * (in_act[i] - mu);
    }
    d_mu_cache[threadIdx.x] = d_mu;
    d_sigma_cache[threadIdx.x] = d_sigma;
  } else {
    d_mu_cache[threadIdx.x] = 0.0f;
    d_sigma_cache[threadIdx.x] = 0.0f;
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 2 == 0) {
      d_mu_cache[threadIdx.x] += d_mu_cache[threadIdx.x+1];
      d_sigma_cache[threadIdx.x] += d_sigma_cache[threadIdx.x+1];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 4 == 0) {
      d_mu_cache[threadIdx.x] += d_mu_cache[threadIdx.x+2];
      d_sigma_cache[threadIdx.x] += d_sigma_cache[threadIdx.x+2];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 8 == 0) {
      d_mu_cache[threadIdx.x] += d_mu_cache[threadIdx.x+4];
      d_sigma_cache[threadIdx.x] += d_sigma_cache[threadIdx.x+4];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 16 == 0) {
      d_mu_cache[threadIdx.x] += d_mu_cache[threadIdx.x+8];
      d_sigma_cache[threadIdx.x] += d_sigma_cache[threadIdx.x+8];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 32 == 0 && u0 < spatial_dim) {
      float d_mu = d_mu_cache[threadIdx.x] + d_mu_cache[threadIdx.x+16];
      float d_sigma = d_sigma_cache[threadIdx.x] + d_sigma_cache[threadIdx.x+16];
      atomicAdd(&mean_grad[c], d_mu);
      atomicAdd(&var_grad[c], d_sigma);
    }
  }
}

__global__ void conv_smooth_bnorm_bwd_data_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *out_delta,
    const float *__restrict__ prev_mean,
    const float *__restrict__ mean_grad,
    const float *__restrict__ var,
    const float *__restrict__ var_grad,
    float epsilon,
    float alpha,
    float *in_delta)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int bank_idx = OFFSET_BANK(threadIdx.x);
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int warp_idx = idx % 32;
  int c = (idx / 32) % num_channels;
  int u0 = warp_idx + ((idx / (32 * num_channels)) % block_spatial_dim) * (16*32);
  int batch_idx = idx / (32 * num_channels * block_spatial_dim);
  if (c < num_channels && u0 < spatial_dim && batch_idx < batch_size) {
    float inv_mean_norm = 1.0f / ((float)(spatial_dim) * (float)(batch_size));
    float inv_var_norm = 1.0f / ((float)(spatial_dim - 1) * (float)(batch_size - 1));
    float mu = prev_mean[c];
    float d_mu = mean_grad[c];
    float sigma = var[c];
    float d_sigma = var_grad[c];
    int i0 = c * spatial_dim + batch_idx * spatial_dim * num_channels;
    int u_limit = min(spatial_dim, u0 + 16*32);
    for (int u = u0; u < u_limit; u += 32) {
      int i = i0 + u;
      in_delta[i] = out_delta[i] * rsqrtf(sigma + epsilon) + alpha * d_mu * inv_mean_norm + alpha * d_sigma * 2.0f * inv_var_norm * (in_act[i] - mu);
    }
  }
}

extern "C" void rembrandt_conv_smooth_bnorm_bwd_batch(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *out_delta,
    const float *prev_mean,
    const float *mean,
    const float *var,
    float epsilon,
    float alpha,
    float *mean_grad,
    float *var_grad,
    float *in_delta,
    cudaStream_t stream)
{
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int n = 32 * num_channels * block_spatial_dim * batch_size;
  conv_smooth_bnorm_bwd_mean_var_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, out_delta, mean, var, epsilon, mean_grad, var_grad);
  conv_smooth_bnorm_bwd_data_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, out_delta, prev_mean, mean_grad, var, var_grad, epsilon, alpha, in_delta);
}

__global__ void conv_smooth_bnorm_rfwd_mean_var_batch_kernel(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *in_r_act,
    const float *__restrict__ prev_mean,
    float alpha,
    float *r_mean,
    float *r_var)
{
  __shared__ float r_mu_cache[1024];
  __shared__ float r_var_cache[1024];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  //int bank_idx = OFFSET_BANK(threadIdx.x);
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int warp_idx = idx % 32;
  int c = (idx / 32) % num_channels;
  int u0 = warp_idx + ((idx / (32 * num_channels)) % block_spatial_dim) * (16*32);
  int batch_idx = idx / (32 * num_channels * block_spatial_dim);
  if (c < num_channels && u0 < spatial_dim && batch_idx < batch_size) {
    float mean_c = prev_mean[c];
    //float r_mean_c = r_mean[c];
    float mu = 0.0f;
    float y = 0.0f;
    int i0 = c * spatial_dim + batch_idx * spatial_dim * num_channels;
    int u_limit = min(spatial_dim, u0 + 16*32);
    for (int u = u0; u < u_limit; u += 32) {
      int i = i0 + u;
      float r_x = in_r_act[i];
      float delta = in_act[i] - mean_c;
      mu += r_x;
      y += delta * r_x;
    }
    r_mu_cache[threadIdx.x] = mu;
    r_var_cache[threadIdx.x] = y;
  } else {
    r_mu_cache[threadIdx.x] = 0.0f;
    r_var_cache[threadIdx.x] = 0.0f;
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 2 == 0) {
      r_mu_cache[threadIdx.x] += r_mu_cache[threadIdx.x+1];
      r_var_cache[threadIdx.x] += r_var_cache[threadIdx.x+1];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 4 == 0) {
      r_mu_cache[threadIdx.x] += r_mu_cache[threadIdx.x+2];
      r_var_cache[threadIdx.x] += r_var_cache[threadIdx.x+2];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 8 == 0) {
      r_mu_cache[threadIdx.x] += r_mu_cache[threadIdx.x+4];
      r_var_cache[threadIdx.x] += r_var_cache[threadIdx.x+4];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 16 == 0) {
      r_mu_cache[threadIdx.x] += r_mu_cache[threadIdx.x+8];
      r_var_cache[threadIdx.x] += r_var_cache[threadIdx.x+8];
    }
  }
  __syncthreads();
  if (c < num_channels && batch_idx < batch_size) {
    if (threadIdx.x % 32 == 0 && u0 < spatial_dim) {
      float mu = alpha * (r_mu_cache[threadIdx.x] + r_mu_cache[threadIdx.x+16]) / ((float)(spatial_dim) * (float)(batch_size));
      float y = 2.0f * alpha * (r_var_cache[threadIdx.x] + r_var_cache[threadIdx.x+16]) / ((float)(spatial_dim) * (float)(batch_size));
      atomicAdd(&r_mean[c], mu);
      atomicAdd(&r_var[c], y);
    }
  }
}

extern "C" void rembrandt_conv_smooth_bnorm_rfwd_batch(
    const float *in_act,
    int spatial_dim,
    int num_channels,
    int batch_size,
    const float *in_r_act,
    const float *prev_mean,
    const float *mean,
    const float *var,
    float epsilon,
    float alpha,
    float *r_mean,
    float *r_var,
    float *out_r_act,
    cudaStream_t stream)
{
  int block_spatial_dim = (spatial_dim+16*32-1)/(16*32);
  int block_n = 32 * num_channels * block_spatial_dim * batch_size;
  int n = spatial_dim * num_channels * batch_size;
  conv_smooth_bnorm_rfwd_mean_var_batch_kernel<<<(block_n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, in_r_act, prev_mean, alpha, r_mean, r_var);
  conv_bnorm_rfwd_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_act, spatial_dim, num_channels, batch_size, in_r_act, mean, r_mean, var, r_var, epsilon, out_r_act);
}
