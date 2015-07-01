#include "common.h"
#include <cuda_runtime_api.h>

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
  /*dim3 block_dim(CUDA_BLOCK_DIM_1D(n));
  dim3 grid_dim(CUDA_GRID_DIM_1D(n));
  map_relu_activation_kernel<<<grid_dim, block_dim, 0, stream>>>(n, x);*/
  map_relu_activation_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(n, x);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_relu_activation_backprop_kernel(
    const float *z,
    int n,
    float *delta)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    delta[i] = delta[i] * (z[i] > 0.0);
  }
}

extern "C" void rembrandt_kernel_map_relu_activation_backprop(
    const float *z,
    int n,
    float *delta,
    cudaStream_t stream)
{
  map_relu_activation_backprop_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(z, n, delta);
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
  /*dim3 block_dim(CUDA_BLOCK_DIM_1D(n));
  dim3 grid_dim(CUDA_GRID_DIM_1D(n));
  map_sigmoid_activation_kernel<<<grid_dim, block_dim, 0, stream>>>(n, x);*/
  map_sigmoid_activation_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(n, x);
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

/*__global__ void map_softmax_cross_entropy_kernel(
    int n
    )
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
  }
}

extern "C" void rembrandt_kernel_map_softmax_cross_entropy(
    cudaStream_t stream)
{
  map_softmax_cross_entropy_kernel<<<, , 0, stream>>>();
  CUDA_POST_KERNEL_CHECK;
}*/

__global__ void map_softmax_cross_entropy_backprop_kernel(
    const float *z,
    int n,
    int truth_label,
    float *delta)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    if (i == truth_label) {
      delta[i] = z[i] - 1.0;
    } else {
      delta[i] = z[i];
    }
  }
}

extern "C" void rembrandt_kernel_map_softmax_cross_entropy_backprop(
    const float *z,
    int n,
    int truth_label,
    float *delta,
    cudaStream_t stream)
{
  map_softmax_cross_entropy_backprop_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(z, n, truth_label, delta);
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
    int m = rand[i] > threshold;
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
