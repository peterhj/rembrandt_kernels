#include "common.h"
#include <cuda_runtime_api.h>
#include <stdint.h>

__global__ void map_noop_kernel() {
  // Do nothing.
}

extern "C" void rembrandt_kernel_map_noop(int n, cudaStream_t stream) {
  map_noop_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>();
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_zero_mask_kernel(
    float *xs,
    int n,
    const float *zero_mask)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    if (zero_mask[i] > 0.0) {
      xs[i] = 0.0;
    }
  }
}

__global__ void image_cast_byte_to_float_kernel(
    const uint8_t *image_bytes,
    int n,
    float *image)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    image[i] = (float)(image_bytes[i]);
  }
}

extern "C" void rembrandt_kernel_map_cast_byte_to_float(
    const uint8_t *image_bytes,
    int n,
    float *image,
    cudaStream_t stream)
{
  image_cast_byte_to_float_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      image_bytes, n, image);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void image_cast_byte_to_float_normalized_kernel(
    const uint8_t *image_bytes,
    int n,
    float *image)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    image[i] = (float)(image_bytes[i]) / 255.0f;
  }
}

extern "C" void rembrandt_kernel_map_cast_byte_to_float_normalized(
    const uint8_t *image_bytes,
    int n,
    float *image,
    cudaStream_t stream)
{
  image_cast_byte_to_float_normalized_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      image_bytes, n, image);
  CUDA_POST_KERNEL_CHECK;
}

extern "C" void rembrandt_kernel_map_zero_mask(float *xs, int n, const float *zero_mask, cudaStream_t stream) {
  map_zero_mask_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(xs, n, zero_mask);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_set_constant_float_kernel(
    int n,
    float *x,
    float c)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    x[i] = c;
  }
}

extern "C" void rembrandt_kernel_map_set_constant_float(
    int n,
    float *x,
    float c,
    cudaStream_t stream)
{
  map_set_constant_float_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(n, x, c);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_set_constant_i32_kernel(
    int n,
    int *x,
    int c)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    x[i] = c;
  }
}

extern "C" void rembrandt_kernel_map_set_constant_i32(
    int n,
    int *x,
    int c,
    cudaStream_t stream)
{
  map_set_constant_i32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(n, x, c);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_add_constant_float_kernel(
    float *x,
    int n,
    float c)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    x[i] += c;
  }
}

extern "C" void rembrandt_kernel_map_add_constant_float(
    float *x,
    int n,
    float c,
    cudaStream_t stream)
{
  map_add_constant_float_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(x, n, c);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void map_multiply_float_kernel(
    const float *x,
    int n,
    float *y)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float x_i = x[i];
    float y_i = y[i];
    y_i = x_i * y_i;
    y[i] = y_i;
  }
}

extern "C" void rembrandt_kernel_map_multiply_float(
    const float *x,
    int n,
    float *y,
    cudaStream_t stream)
{
  map_multiply_float_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(x, n, y);
  CUDA_POST_KERNEL_CHECK;
}
