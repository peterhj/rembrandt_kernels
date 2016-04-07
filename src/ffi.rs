use cuda::ffi::runtime::{cudaStream_t};
use libc::{c_int};

#[link(name = "rembrandt_kernels_cuda", kind = "static")]
extern "C" {
  // Map kernels.
  pub fn rembrandt_kernel_map_cast_byte_to_float(
      x: *const u8,
      n: c_int,
      y: *mut f32,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_map_cast_byte_to_float_normalized(
      x: *const u8,
      n: c_int,
      y: *mut f32,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_map_multiply_float(
      x: *const f32,
      n: c_int,
      y: *mut f32,
      stream: cudaStream_t,
  );

  // Reduce kernels.
  pub fn rembrandt_kernel_reduce_count_nonzero(
      xs: *const f32,
      n: c_int,
      count_block_tmp: *mut f32,
      count: *mut f32,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_reduce_count_sparse(
      xs: *const f32,
      n: c_int,
      count_block_tmp: *mut f32,
      count: *mut f32,
      threshold: f32,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_reduce_count_range(
      xs: *const f32,
      n: c_int,
      count_block_tmp: *mut f32,
      count: *mut f32,
      lo_threshold: f32,
      hi_threshold: f32,
      stream: cudaStream_t,
  );

  // Batch map kernels.
  pub fn rembrandt_kernel_batch_blockmap_normalize(
      xs: *mut f32,
      num_channels: c_int,
      batch_size: c_int,
      norm: *const f32,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_batch_map_zero_mask_inplace(
      z: *mut f32,
      num_channels: c_int,
      batch_size: c_int,
      zero_mask: *const f32,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_batch_map_pos_mask_inplace(
      z: *mut f32,
      num_channels: c_int,
      batch_size: c_int,
      pos_mask: *const f32,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_batch_map_rect_inplace(
      z: *mut f32,
      num_channels: c_int,
      batch_size: c_int,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_batch_map_rect_backprop_inplace(
      out_act: *const f32,
      num_channels: c_int,
      batch_size: c_int,
      out_delta: *mut f32,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_batch_map_bounded_rect_inplace(
      z: *mut f32,
      num_channels: c_int,
      batch_size: c_int,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_batch_map_bounded_rect_backprop_inplace(
      out_act: *const f32,
      num_channels: c_int,
      batch_size: c_int,
      out_delta: *mut f32,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_batch_map_sigmoid_inplace_forward(
      z: *mut f32,
      num_channels: c_int,
      batch_size: c_int,
      beta: f32,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_batch_map_sigmoid_inplace_backward(
      out_act: *const f32,
      num_channels: c_int,
      batch_size: c_int,
      out_delta: *mut f32,
      beta: f32,
      stream: cudaStream_t,
  );
  /*pub fn rembrandt_kernel_batch_map_boltzmann_q_transform(
      probs: *const f32,
      num_channels: c_int,
      batch_size: c_int,
      beta: f32,
      qvalues: *mut f32,
      stream: cudaStream_t,
  );*/
  pub fn rembrandt_kernel_batch_map_softmax_cross_entropy_loss(
      probs: *const f32,
      num_channels: c_int,
      batch_size: c_int,
      labels: *const i32,
      loss_accum: *mut f32,
      minibatch_size: f32,
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
  pub fn rembrandt_kernel_batch_map_multi_bin_logistic(
      in_act: *const f32,
      num_channels: c_int,
      batch_size: c_int,
      out_act: *mut f32,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_batch_map_multi_bin_logistic_xent_loss_backprop(
      out_act: *const f32,
      num_channels: c_int,
      batch_size: c_int,
      cat_labels: *const i32,
      bin_labels: *const i32,
      in_delta: *mut f32,
      maybe_loss: *mut f32,
      minibatch_size: f32,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_batch_map_softmax_kl_loss(
      out_act:      *const f32,
      num_channels: c_int,
      batch_size:   c_int,
      labels:       *const i32,
      weights:      *const f32,
      loss1:        *mut f32,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_batch_map_softmax_kl_backward(
      out_act:      *const f32,
      num_channels: c_int,
      batch_size:   c_int,
      labels:       *const i32,
      weights:      *const f32,
      in_delta:     *mut f32,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_batch_map_softmax_ind_backward(
      out_act:      *const f32,
      num_channels: c_int,
      batch_size:   c_int,
      labels:       *const i32,
      weights:      *const f32,
      in_delta:     *mut f32,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_batch_map_marginalized_softmax_ind_backward(
      out_act:      *const f32,
      num_channels: c_int,
      batch_size:   c_int,
      weights:      *const f32,
      cat_weights:  *const f32,
      in_delta:     *mut f32,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_batch_map_logistic_forward(
      in_values:    *const f32,
      num_channels: c_int,
      batch_size:   c_int,
      out_values:   *mut f32,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_batch_map_logistic_ind_backward(
      out_values:   *const f32,
      num_channels: c_int,
      batch_size:   c_int,
      labels:       *const i32,
      weights:      *const f32,
      in_delta:     *mut f32,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_batch_map_antilogistic_forward(
      in_values:    *const f32,
      num_channels: c_int,
      batch_size:   c_int,
      logit_sums:   *const f32,
      out_values:   *mut f32,
      stream: cudaStream_t,
  );
  pub fn rembrandt_kernel_batch_map_antilogistic_kl_backward(
      out_values:   *const f32,
      num_channels: c_int,
      batch_size:   c_int,
      labels:       *const i32,
      weights:      *const f32,
      in_delta:     *mut f32,
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
  pub fn rembrandt_kernel_batch_blockreduce_sum(
      xs: *const f32,
      len: c_int,
      batch_size: c_int,
      xs_sum: *mut f32,
      alpha:  f32,
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
