extern crate cuda;

use cuda::ffi::runtime::{cudaStream_t};

/*#[repr(C)]
struct cudaStream_s;
#[allow(non_camel_case_types)]
pub type cudaStream_t = *mut cudaStream_s;*/

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
  pub fn rembrandt_kernel_map_relu_activation(
      n: i32,
      x: *mut f32,
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

  // General purpose reduction kernels.
  pub fn rembrandt_kernel_blockreduce_argmax_float(
      n: i32,
      x: *const f32,
      max_block: *mut f32,
      idx_block: *mut i32,
      stream: cudaStream_t);
  pub fn rembrandt_kernel_blockreduce_argmin_float(
      n: i32,
      x: *const f32,
      min_block: *mut f32,
      idx_block: *mut i32,
      stream: cudaStream_t);
}
