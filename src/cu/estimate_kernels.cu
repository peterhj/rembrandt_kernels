#include <cuda_runtime_api.h>
#include <stdint.h>

__global__ void estimate_conv_mean_batch_kernel(
    const float *src,
    int spatial_dim,
    int channels,
    int batch_size,
    float *mean)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  /*int u = idx % spatial_dim;
  int c = (idx / spatial_dim) % channels;*/
  int c = idx % channels;
  int u = (idx / channels) % spatial_dim;
  int batch_idx = idx / (channels * spatial_dim);
  if (c < channels && u < spatial_dim && batch_idx < batch_size) {
    //float dy = src[idx];
    int i = u + c * spatial_dim + batch_idx * spatial_dim * channels;
    float dy = src[i];
    atomicAdd(&mean[c], dy);
  }
}

extern "C" void rembrandt_kernel_estimate_conv_mean_batch(
    const float *src,
    int spatial_dim,
    int channels,
    int batch_size,
    float *mean,
    cudaStream_t stream)
{
  int n = spatial_dim * channels * batch_size;
  estimate_conv_mean_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      src, spatial_dim, channels, batch_size, mean);
}

__global__ void estimate_conv_mean_fast_batch_kernel(
    const float *src,
    int spatial_dim,
    int channels,
    int batch_size,
    float *mean)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int unroll_spatial_dim = (spatial_dim+32-1)/32;
  int c = idx % channels;
  int u0 = ((idx / channels) % unroll_spatial_dim) * 32;
  int batch_idx = idx / (channels * unroll_spatial_dim);
  if (c < channels && u0 < spatial_dim && batch_idx < batch_size) {
    float dy = 0.0f;
    int i0 = c * spatial_dim + batch_idx * spatial_dim * channels;
    int u_limit = min(u0+32, spatial_dim);
    for (int u = u0; u < u_limit; u++) {
      int i = i0 + u;
      dy += src[i];
    }
    atomicAdd(&mean[c], dy);
  }
}

extern "C" void rembrandt_kernel_estimate_conv_mean_fast_batch(
    const float *src,
    int spatial_dim,
    int channels,
    int batch_size,
    float *mean,
    cudaStream_t stream)
{
  int n = ((spatial_dim+32-1)/32) * channels * batch_size;
  estimate_conv_mean_fast_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      src, spatial_dim, channels, batch_size, mean);
}

/*__global__ void estimate_online_mean_kernel(
    const float *mean_batch,
    int channels,
    int batch_size,
    int acc_size,
    float *mean_acc)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < channels) {
    float mean_a = mean_acc[idx];
    float delta = mean_batch[idx] - mean_a;
    float y = mean_a + ((float)(batch_size)) / ((float)(acc_size + batch_size)) * delta;
    mean_acc[idx] = y;
  }
}

extern "C" void rembrandt_kernel_estimate_online_mean(
    const float *mean_batch,
    int channels,
    int batch_size,
    int acc_size,
    float *mean_acc,
    cudaStream_t stream)
{
  estimate_online_mean_kernel<<<(channels+1024-1)/1024, 1024, 0, stream>>>(
      mean_batch, channels, batch_size, acc_size, mean_acc);
}*/

__global__ void estimate_conv_var_batch_kernel(
    const float *src,
    int spatial_dim,
    int channels,
    int batch_size,
    const float *mean,
    float *var)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  /*int u = idx % spatial_dim;
  int c = (idx / spatial_dim) % channels;*/
  int c = idx % channels;
  int u = (idx / channels) % spatial_dim;
  int batch_idx = idx / (channels * spatial_dim);
  if (c < channels && u < spatial_dim && batch_idx < batch_size) {
    int i = u + c * spatial_dim + batch_idx * spatial_dim * channels;
    float mean_c = mean[c] / ((float)(batch_size));
    //float delta = src[idx] - mean_c;
    float delta = src[i] - mean_c;
    float dy = delta * delta;
    atomicAdd(&var[c], dy);
  }
}

extern "C" void rembrandt_kernel_estimate_conv_var_batch(
    const float *src,
    int spatial_dim,
    int channels,
    int batch_size,
    const float *mean,
    float *var,
    cudaStream_t stream)
{
  int n = spatial_dim * channels * batch_size;
  estimate_conv_var_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      src, spatial_dim, channels, batch_size, mean, var);
}

__global__ void estimate_conv_var_fast_batch_kernel(
    const float *src,
    int spatial_dim,
    int channels,
    int batch_size,
    const float *mean,
    float *var)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int unroll_spatial_dim = (spatial_dim+32-1)/32;
  int c = idx % channels;
  int u0 = ((idx / channels) % unroll_spatial_dim) * 32;
  int batch_idx = idx / (channels * unroll_spatial_dim);
  if (c < channels && u0 < spatial_dim && batch_idx < batch_size) {
    float mean_c = mean[c] / ((float)(batch_size));
    float dy = 0.0f;
    int i0 = c * spatial_dim + batch_idx * spatial_dim * channels;
    int u_limit = min(u0+32, spatial_dim);
    for (int u = u0; u < u_limit; u++) {
      int i = i0 + u;
      float delta = src[i] - mean_c;
      dy += delta * delta;
    }
    atomicAdd(&var[c], dy);
  }
}

extern "C" void rembrandt_kernel_estimate_conv_var_fast_batch(
    const float *src,
    int spatial_dim,
    int channels,
    int batch_size,
    const float *mean,
    float *var,
    cudaStream_t stream)
{
  int n = ((spatial_dim+32-1)/32) * channels * batch_size;
  estimate_conv_var_fast_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      src, spatial_dim, channels, batch_size, mean, var);
}

__global__ void estimate_online_var_kernel(
    const float *mean_batch,
    int len,
    const float *var_batch,
    const float *mean_acc,
    int batch_size,
    int acc_size,
    float *var_acc)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) {
    float var_a = var_acc[idx];
    float var_b = var_batch[idx];
    float delta = mean_batch[idx] / ((float)(batch_size)) - mean_acc[idx] / ((float)(acc_size));
    float y = var_a + var_b + ((float)(acc_size)) / ((float)(acc_size + batch_size)) * ((float)(batch_size)) * delta * delta;
    var_acc[idx] = y;
  }
}

extern "C" void rembrandt_kernel_estimate_online_var(
    const float *mean_batch,
    int len,
    const float *var_batch,
    const float *mean_acc,
    int batch_size,
    int acc_size,
    float *var_acc,
    cudaStream_t stream)
{
  estimate_online_var_kernel<<<(len+1024-1)/1024, 1024, 0, stream>>>(
      mean_batch, len, var_batch, mean_acc, batch_size, acc_size, var_acc);
}

__global__ void estimate_invstd_kernel(
    const float *var,
    int len,
    float epsilon,
    float *invstd)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) {
    float y = 1.0f / sqrtf(epsilon + var[idx]);
    invstd[idx] = y;
  }
}

extern "C" void rembrandt_kernel_estimate_invstd(
    const float *var,
    int len,
    float epsilon,
    float *invstd,
    cudaStream_t stream)
{
  estimate_invstd_kernel<<<(len+1024-1)/1024, 1024, 0, stream>>>(
      var, len, epsilon, invstd);
}
