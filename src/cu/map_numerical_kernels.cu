#include "common.h"
#include <cuda_runtime_api.h>
#include <stdio.h>

__global__ void map_exp_kernel(
    float *x,
    int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    x[i] = expf(x[i]);
  }
}

extern "C" void rembrandt_kernel_map_exp(
    float *x,
    int n,
    cudaStream_t stream)
{
  map_exp_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(x, n);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_subtract_scalar_kernel(
    float *x,
    int n,
    float *scalar)
{
  __shared__ float c[1];
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (threadIdx.x == 0) {
    c[0] = *scalar;
  }
  __syncthreads();
  if (i < n) {
    x[i] -= c[0];
  }
}

extern "C" void rembrandt_kernel_map_subtract_scalar(
    float *x,
    int n,
    float *scalar,
    cudaStream_t stream)
{
  map_subtract_scalar_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(x, n, scalar);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_divide_scalar_kernel(
    float *x,
    int n,
    float *scalar)
{
  __shared__ float c[1];
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (threadIdx.x == 0) {
    c[0] = *scalar;
  }
  __syncthreads();
  if (i < n) {
    x[i] /= c[0];
  }
  /*__syncthreads();
  if (i == 0) {
    printf("CUDA DEBUG: %f %f %f %f %f %f %f %f %f %f\n",
        x[0], x[1], x[2], x[3], x[4],
        x[5], x[6], x[7], x[8], x[9]);
  }*/
}

extern "C" void rembrandt_kernel_map_divide_scalar(
    float *x,
    int n,
    float *scalar,
    cudaStream_t stream)
{
  map_divide_scalar_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(x, n, scalar);
  CUDA_POST_KERNEL_CHECK;
}

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
  map_relu_activation_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(n, x);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_relu_activation_backprop_kernel(
    int n,
    const float *z,
    float *delta)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    if (z[i] <= 0.0) {
      delta[i] = 0.0;
    }
  }
}

extern "C" void rembrandt_kernel_map_relu_activation_backprop(
    int n,
    const float *z,
    float *delta,
    cudaStream_t stream)
{
  map_relu_activation_backprop_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(n, z, delta);
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
  map_sigmoid_activation_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(n, x);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_sigmoid_activation_backprop_kernel(
    int n,
    const float *z,
    float *delta)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float z_i = z[i];
    delta[i] = delta[i] * z_i * (1.0 - z_i);
  }
}

extern "C" void rembrandt_kernel_map_sigmoid_activation_backprop(
    int n,
    const float *z,
    float *delta,
    cudaStream_t stream)
{
  map_sigmoid_activation_backprop_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(n, z, delta);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_tanh_activation_kernel(
    int n,
    float *x)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float x_i = x[i];
    x[i] = tanhf(x_i);
  }
}

extern "C" void rembrandt_kernel_map_tanh_activation(
    int n,
    float *x,
    cudaStream_t stream)
{
  map_tanh_activation_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(n, x);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_tanh_activation_backprop_kernel(
    int n,
    const float *z,
    float *delta)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float z_i = z[i];
    delta[i] = delta[i] * (1.0 - z_i * z_i);
  }
}

extern "C" void rembrandt_kernel_map_tanh_activation_backprop(
    int n,
    const float *z,
    float *delta,
    cudaStream_t stream)
{
  map_tanh_activation_backprop_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(n, z, delta);
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

__global__ void map_softmax_cross_entropy_loss_kernel(
    const float *z,
    int n,
    int truth_label,
    float *loss)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    if (i == truth_label) {
      loss[0] -= logf(z[i]);
      //printf("CUDA DEBUG: loss: %g\n", loss[0]);
    }
  }
}

extern "C" void rembrandt_kernel_map_softmax_cross_entropy_loss(
    const float *z,
    int n,
    int truth_label,
    float *loss,
    cudaStream_t stream)
{
  //printf("CUDA DEBUG: n %d label %d\n", n, truth_label);
  map_softmax_cross_entropy_loss_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      z, n, truth_label, loss);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_softmax_cross_entropy_loss_report_kernel(
    const float *loss)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i == 0) {
    printf("CUDA DEBUG: loss: %g\n", loss[0]);
  }
}

extern "C" void rembrandt_kernel_map_softmax_cross_entropy_loss_report(
    const float *loss,
    cudaStream_t stream)
{
  map_softmax_cross_entropy_loss_report_kernel<<<1, 32, 0, stream>>>(loss);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_softmax_cross_entropy_loss_backprop_kernel(
    const float *z,
    int n,
    int truth_label,
    float *delta)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float z_i = z[i];
    if (i == truth_label) {
      delta[i] = z_i - 1.0;
    } else {
      delta[i] = z_i;
    }
  }
}

extern "C" void rembrandt_kernel_map_softmax_cross_entropy_loss_backprop(
    const float *z,
    int n,
    int truth_label,
    float *delta,
    cudaStream_t stream)
{
  map_softmax_cross_entropy_loss_backprop_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(z, n, truth_label, delta);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_dropout_kernel(
    const float *x,
    int n,
    float threshold,
    float scale,
    const float *rand,
    float *z,
    int *mask)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    int m = rand[i] >= threshold;
    z[i] = scale * x[i] * m;
    mask[i] = m;
  }
}

extern "C" void rembrandt_kernel_map_dropout(
    const float *x,
    int n,
    float threshold,
    float scale,
    const float *rand,
    float *z,
    int *mask,
    cudaStream_t stream)
{
  map_dropout_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(x, n, threshold, scale, rand, z, mask);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_dropout_backprop_kernel(
    const float *z,
    int n,
    float threshold,
    float scale,
    const int *mask,
    float *delta)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    delta[i] = scale * z[i] * mask[i];
  }
}

extern "C" void rembrandt_kernel_map_dropout_backprop(
    const float *z,
    int n,
    float threshold,
    float scale,
    const int *mask,
    float *delta,
    cudaStream_t stream)
{
  map_dropout_backprop_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(z, n, threshold, scale, mask, delta);
  CUDA_POST_KERNEL_CHECK;
}
