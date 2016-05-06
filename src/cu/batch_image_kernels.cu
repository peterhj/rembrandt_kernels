#include <cuda_runtime_api.h>

__global__ void image3_crop(
    const float *in_pixels,
    int in_width,
    int in_height,
    int channels,
    int x_offset,
    int y_offset,
    float *out_pixels,
    int crop_width,
    int crop_height)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int x = idx % crop_width;
  int y = (idx / crop_width) % crop_height;
  int c = idx / (crop_width * crop_height);

  if ((x < crop_width) && (y < crop_height) && (c < channels)) {
    int in_idx = x_offset + x + (y_offset + y) * in_width + c * in_width * in_height;
    int out_idx = x + y * crop_width + c * crop_width * crop_height;

    out_pixels[out_idx] = in_pixels[in_idx];
  }
}

extern "C" void rembrandt_kernel_image3_crop(
    const float *in_pixels,
    int in_width,
    int in_height,
    int channels,
    int x_offset,
    int y_offset,
    float *out_pixels,
    int crop_width,
    int crop_height,
    cudaStream_t stream)
{
  int n = crop_width * crop_height * channels;
  image3_crop<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_pixels,
      in_width,
      in_height,
      channels,
      x_offset,
      y_offset,
      out_pixels,
      crop_width,
      crop_height);
}

__global__ void batch_image3_crop(
    const float *in_pixels,
    int in_stride,
    int batch_size,
    const int *in_widths,
    const int *in_heights,
    int channels,
    const int *in_x_offsets,
    const int *in_y_offsets,
    float *out_pixels,
    int crop_width,
    int crop_height)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int x = idx % crop_width;
  int y = (idx / crop_width) % crop_height;
  int c = (idx / (crop_width * crop_height)) % channels;
  int batch_idx = idx / (crop_width * crop_height * channels);

  if ((x < crop_width) && (y < crop_height) && (c < channels) && (batch_idx < batch_size)) {
    int in_width = in_widths[batch_idx];
    int in_height = in_heights[batch_idx];
    int x_offset = in_x_offsets[batch_idx];
    int y_offset = in_y_offsets[batch_idx];

    int in_idx = x_offset + x + (y_offset + y) * in_width + c * in_width * in_height + batch_idx * in_stride;
    int out_idx = x + y * crop_width + c * crop_width * crop_height + batch_idx * crop_width * crop_height * channels;

    out_pixels[out_idx] = in_pixels[in_idx];
  }
}

extern "C" void rembrandt_kernel_batch_image3_crop(
    const float *in_pixels,
    int in_stride,
    int batch_size,
    const int *in_widths,
    const int *in_heights,
    int channels,
    const int *in_x_offsets,
    const int *in_y_offsets,
    float *out_pixels,
    int crop_width,
    int crop_height,
    cudaStream_t stream)
{
  int n = crop_width * crop_height * channels * batch_size;
  batch_image3_crop<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_pixels,
      in_stride,
      batch_size,
      in_widths,
      in_heights,
      channels,
      in_x_offsets,
      in_y_offsets,
      out_pixels,
      crop_width,
      crop_height);
}
