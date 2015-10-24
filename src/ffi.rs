use cuda::ffi::runtime::{cudaStream_t};
use libc::{c_int};

#[link(name = "rembrandt_kernels_cuda", kind = "static")]
extern "C" {
  // Batch map kernels.
  pub fn rembrandt_kernel_batch_map_zero_mask_inplace(
      z: *mut f32,
      num_channels: c_int,
      batch_size: c_int,
      zero_mask: *const f32,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_batch_map_softmax_cross_entropy_loss_backprop(
      z: *const f32,
      num_channels: c_int,
      batch_size: c_int,
      labels: *const i32,
      delta: *mut f32,
      minibatch_size: f32,
      stream: cudaStream_t,
  );

  // Batch reduce and scan kernels.
  pub fn rembrandt_kernel_batch_blockreduce_argmax(
      xs: *const f32,
      len: c_int,
      batch_size: c_int,
      xs_max: *mut f32,
      xs_idx: *mut i32,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_batch_blockscan_prefix_sum(
      xs: *const f32,
      len: c_int,
      batch_size: c_int,
      xs_prefix_sum: *mut f32,
      stream: cudaStream_t,
  );

  // Batch sort kernels.
  pub fn rembrandt_kernel_batch_blocksort_bitonic_argrevsort(
      xs: *const f32,
      len: c_int,
      batch_size: c_int,
      xs_value_block: *mut f32,
      xs_index_block: *mut i32,
      stream: cudaStream_t,
  );
}
