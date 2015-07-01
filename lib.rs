extern crate cuda;

use cuda::ffi::runtime::{cudaStream_t};

#[link(name = "rembrandt_kernels_cuda", kind = "static")]
extern "C" {
  // Image processing kernels.
  pub fn rembrandt_kernel_image_cast_to_float(
      width: i32, height: i32, channels: i32,
      image_bytes: *const u8,
      image: *mut f32,
      stream: cudaStream_t);
  pub fn rembrandt_kernel_image_interleaved_cast_to_float(
      width: i32, height: i32, channels: i32,
      image_bytes: *const u8,
      image: *mut f32,
      stream: cudaStream_t);
  pub fn rembrandt_kernel_image_im2col(
      image: *const f32,
      width: i32, height: i32, channels: i32,
      conv_diam: i32, conv_stride: i32,
      col: *mut f32,
      stream: cudaStream_t);
  pub fn rembrandt_kernel_image_col2im(
      col: *const f32,
      width: i32, height: i32, channels: i32,
      conv_diam: i32, conv_stride: i32,
      image: *mut f32,
      stream: cudaStream_t);
  pub fn rembrandt_kernel_image_max_pool(
      src_data: *const f32,
      width: i32, height: i32, channels: i32,
      pool_diam: i32, pool_stride: i32, pool_pad: i32,
      dst_data: *mut f32,
      dst_mask: *mut i32,
      stream: cudaStream_t);
  pub fn rembrandt_kernel_image_max_pool_backward(
      src_data: *const f32,
      src_mask: *const i32,
      width: i32, height: i32, channels: i32,
      pool_diam: i32, pool_stride: i32, pool_pad: i32,
      dst_data: *mut f32,
      stream: cudaStream_t);

  // General purpose map kernels.
  pub fn rembrandt_kernel_map_set_constant_float(
      n: i32,
      x: *mut f32,
      c: f32,
      stream: cudaStream_t);
  pub fn rembrandt_kernel_map_set_constant_i32(
      n: i32,
      x: *mut i32,
      c: i32,
      stream: cudaStream_t);

  // Numerical (single-precision) map kernels.
  pub fn rembrandt_kernel_map_exp(
      x: *mut f32,
      n: i32,
      stream: cudaStream_t);
  pub fn rembrandt_kernel_map_subtract_scalar(
      x: *mut f32,
      n: i32,
      scalar: *const f32,
      stream: cudaStream_t);
  pub fn rembrandt_kernel_map_divide_scalar(
      x: *mut f32,
      n: i32,
      scalar: *const f32,
      stream: cudaStream_t);
  pub fn rembrandt_kernel_map_relu_activation(
      n: i32,
      x: *mut f32,
      stream: cudaStream_t);
  pub fn rembrandt_kernel_map_relu_activation_backprop(
      z: *const f32,
      n: i32,
      delta: *mut f32,
      stream: cudaStream_t);
  pub fn rembrandt_kernel_map_sigmoid_activation(
      n: i32,
      x: *mut f32,
      stream: cudaStream_t);
  pub fn rembrandt_kernel_map_kahan_sum_update(
      n: i32,
      x: *const f32,
      y_sum: *mut f32,
      y_err: *mut f32,
      stream: cudaStream_t);
  pub fn rembrandt_kernel_map_kahan_sum_finish(
      n: i32,
      y_sum: *const f32,
      y_err: *const f32,
      s: *mut f32,
      stream: cudaStream_t);
  pub fn rembrandt_kernel_map_softmax_cross_entropy_backprop(
      z: *const f32,
      n: i32,
      truth_label: i32,
      delta: *mut f32,
      stream: cudaStream_t);
  pub fn rembrandt_kernel_map_dropout(
      x: *const f32,
      n: i32,
      threshold: f32,
      scale: f32,
      rand: *const f32,
      z: *mut f32,
      mask: *mut i32,
      stream: cudaStream_t);
  pub fn rembrandt_kernel_map_dropout_backprop(
      z: *const f32,
      n: i32,
      threshold: f32,
      scale: f32,
      mask: *const i32,
      delta: *mut f32,
      stream: cudaStream_t);

  // General purpose reduction kernels.
  pub fn rembrandt_kernel_blockreduce_argmax_float(
      n: i32,
      x: *const f32,
      max_block: *mut f32,
      //idx_block: *mut i32,
      idx_block: *mut f32,
      stream: cudaStream_t);
  /*pub fn rembrandt_kernel_blockreduce_argmin_float(
      n: i32,
      x: *const f32,
      min_block: *mut f32,
      idx_block: *mut i32,
      stream: cudaStream_t);*/
  pub fn rembrandt_kernel_blockreduce_argmax(
      n: i32,
      x: *const f32,
      max_block: *mut f32,
      idx_block: *mut i32,
      stream: cudaStream_t);
  pub fn rembrandt_kernel_blockreduce_sum(
      n: i32,
      x: *const f32,
      sum_block: *mut f32,
      stream: cudaStream_t);
}
