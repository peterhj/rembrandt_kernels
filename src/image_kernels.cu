#include "common.h"
#include <cuda_runtime_api.h>
#include <float.h>
#include <stdint.h>

__global__ void image_cast_to_float_kernel(
    int width, int height, int channels,
    const uint8_t *image_bytes,
    float *image)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int n = width * height * channels;
  if (i < n) {
    image[i] = (float)(image_bytes[i]);
  }
}

extern "C" void rembrandt_kernel_image_cast_to_float(
    int width, int height, int channels,
    const uint8_t *image_bytes,
    float *image,
    cudaStream_t stream)
{
  int n = width * height * channels;
  dim3 block_dim(CUDA_BLOCK_DIM_1D(n));
  dim3 grid_dim(CUDA_GRID_DIM_1D(n));
  image_cast_to_float_kernel<<<grid_dim, block_dim, 0, stream>>>(
      width, height, channels,
      image_bytes,
      image);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void image_interleaved_cast_to_float_kernel(
    int width, int height, int channels,
    const uint8_t *image_bytes,
    float *image)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int n = width * height * channels;
  // FIXME(20150517)
  if (i < n) {
    image[i] = (float)(image_bytes[i]);
  }
}

extern "C" void rembrandt_kernel_image_interleaved_cast_to_float(
    int width, int height, int channels,
    const uint8_t *image_bytes,
    float *image,
    cudaStream_t stream)
{
  int n = width * height * channels;
  dim3 block_dim(CUDA_BLOCK_DIM_1D(n));
  dim3 grid_dim(CUDA_GRID_DIM_1D(n));
  image_interleaved_cast_to_float_kernel<<<grid_dim, block_dim, 0, stream>>>(
      width, height, channels,
      image_bytes,
      image);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void caffe_im2col_kernel(const int n, const float* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    float* data_col) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    int w_out = index % width_col;
    int h_index = index / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * kernel_h * kernel_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    float* data_col_ptr = data_col;
    data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    const float* data_im_ptr = data_im;
    data_im_ptr += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
            data_im_ptr[i * width + j] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

extern "C" void rembrandt_kernel_image_im2col(
    const float *data_im,
    int width, int height, int channels,
    int conv_diam, int conv_stride,
    float *data_col,
    cudaStream_t stream)
{
  int kernel_w = conv_diam;
  int kernel_h = conv_diam;
  int conv_pad = conv_diam / 2;
  int pad_w = conv_pad;
  int pad_h = conv_pad;
  int stride_w = conv_stride;
  int stride_h = conv_stride;

  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  caffe_im2col_kernel<<<(num_kernels+1024-1)/1024, 1024, 0, stream>>>(
      num_kernels,
      data_im,
      height, width,
      kernel_h, kernel_w,
      pad_h, pad_w,
      stride_h, stride_w,
      height_col, width_col,
      data_col);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void caffe_col2im_kernel(const int n, const float* data_col,
    const int height, const int width, const int channels,
    const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    float* data_im)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    float val = 0;
    int w = index % width + pad_w;
    int h = (index / width) % height + pad_h;
    int c = index / (width * height);
    // compute the start and end of the output
    int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
    int w_col_end = min(w / stride_w + 1, width_col);
    int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
    int h_col_end = min(h / stride_h + 1, height_col);
    /*
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        // the col location: [c * width * height + h_out, w_out]
        int c_col = c * patch_h * patch_w + (h - h_col * stride_h) * ksize
            + (w - w_col * stride_w);
        val += data_col[(c_col * height_col + h_col) * width_col + w_col];
      }
    }
    */
    // equivalent implementation
    int offset =
        (c * patch_h * patch_w + h * patch_w + w) * height_col * width_col;
    int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
    int coeff_w_col = (1 - stride_w * height_col * width_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}

extern "C" void rembrandt_kernel_image_col2im(
    //const float* data_col, const int channels,
    //const int height, const int width, const int patch_h, const int patch_w,
    //const int pad_h, const int pad_w, const int stride_h,
    //const int stride_w, float* data_im) {
    const float *data_col,
    int width, int height, int channels,
    int conv_diam, int conv_stride,
    float *data_im,
    cudaStream_t stream)
{
  int patch_w = conv_diam;
  int patch_h = conv_diam;
  int conv_pad = conv_diam / 2;
  int pad_w = conv_pad;
  int pad_h = conv_pad;
  int stride_w = conv_stride;
  int stride_h = conv_stride;

  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_col2im_kernel<<<(num_kernels+1024-1)/1024, 1024, 0, stream>>>(
      num_kernels, data_col, height, width, channels, patch_h, patch_w,
      pad_h, pad_w, stride_h, stride_w,
      height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void caffe_max_pool(
    const int nthreads, const float* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    //float* top_data, int* mask, float* top_mask)
    float *top_data)
{
  //CUDA_KERNEL_LOOP(index, nthreads) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    float maxval = -FLT_MAX;
    int maxidx = -1;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_data[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_data[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    /*if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }*/
  }
}

extern "C" void rembrandt_kernel_image_max_pool(
    const float *src_data,
    int width, int height, int channels,
    int pool_diam, int pool_stride, int pool_pad,
    float *dst_data,
    cudaStream_t stream)
{
  int pooled_width = (width + 2 * pool_pad - pool_diam + pool_stride - 1) / pool_stride + 1;
  int pooled_height = (height + 2 * pool_pad - pool_diam + pool_stride - 1) / pool_stride + 1;
  int count = pooled_width * pooled_height * channels;
  caffe_max_pool<<<(count+1024-1)/1024, 1024, 0, stream>>>(
      count, src_data,
      1, channels, height, width, 
      pooled_height, pooled_width,
      pool_diam, pool_diam,
      pool_stride, pool_stride,
      pool_pad, pool_pad,
      dst_data);
  CUDA_POST_KERNEL_CHECK;
}
