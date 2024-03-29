use cuda::ffi::runtime::{cudaStream_t};
use libc::{c_int};

#[link(name = "rembrandt_kernels_cuda", kind = "static")]
extern "C" {
  pub fn rembrandt_kernel_div_map_inplace(
      xs:       *mut f32,
      n:        c_int,
      ys:       *const f32,
      stream:   cudaStream_t,
  );
  pub fn rembrandt_kernel_exp_map(
      xs:       *const f32,
      n:        c_int,
      ys:       *mut f32,
      stream:   cudaStream_t,
  );
  pub fn rembrandt_kernel_scalar_sub_map_batch_inplace(
      xs:           *mut f32,
      frame_len:    c_int,
      batch_size:   c_int,
      scalars:      *const f32,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_kernel_scalar_div_map_batch(
      xs:           *const f32,
      frame_len:    c_int,
      batch_size:   c_int,
      scalars:      *const f32,
      ys:           *mut f32,
      stream:       cudaStream_t,
  );

  pub fn rembrandt_conv_diag_affine_white_var_fwd_batch(
      in_act:       *const f32,
      spatial_dim:  c_int,
      num_channels: c_int,
      batch_size:   c_int,
      mean:         *const f32,
      var:          *const f32,
      epsilon:      f32,
      out_act:      *mut f32,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_conv_diag_affine_white_fwd_batch(
      in_act:       *const f32,
      spatial_dim:  c_int,
      num_channels: c_int,
      batch_size:   c_int,
      mean:         *const f32,
      istd:         *const f32,
      out_act:      *mut f32,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_conv_diag_affine_fwd_batch(
      in_act:       *const f32,
      spatial_dim:  c_int,
      num_channels: c_int,
      batch_size:   c_int,
      scale:        *const f32,
      bias:         *const f32,
      out_act:      *mut f32,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_conv_diag_affine_fwd_inplace_batch(
      out_act:      *mut f32,
      spatial_dim:  c_int,
      num_channels: c_int,
      batch_size:   c_int,
      scale:        *const f32,
      bias:         *const f32,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_conv_diag_linear_fwd_batch(
      in_act:       *const f32,
      spatial_dim:  c_int,
      num_channels: c_int,
      batch_size:   c_int,
      scale:        *const f32,
      out_act:      *mut f32,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_conv_diag_affine_bwd_data_batch(
      in_act:       *const f32,
      spatial_dim:  c_int,
      num_channels: c_int,
      batch_size:   c_int,
      out_delta:    *const f32,
      scale:        *const f32,
      in_delta:     *mut f32,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_conv_diag_affine_bwd_batch(
      in_act:       *const f32,
      spatial_dim:  c_int,
      num_channels: c_int,
      batch_size:   c_int,
      out_delta:    *const f32,
      scale:        *const f32,
      scale_grad:   *mut f32,
      bias_grad:    *mut f32,
      in_delta:     *mut f32,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_conv_bnorm_inferfwd_batch(
      in_act:       *const f32,
      spatial_dim:  c_int,
      num_channels: c_int,
      batch_size:   c_int,
      mean:         *const f32,
      var:          *const f32,
      scale:        *const f32,
      bias:         *const f32,
      epsilon:      f32,
      out_act:      *mut f32,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_conv_bnorm_bwd_batch(
      in_act:       *const f32,
      spatial_dim:  c_int,
      num_channels: c_int,
      batch_size:   c_int,
      out_delta:    *const f32,
      mean:         *const f32,
      var:          *const f32,
      epsilon:      f32,
      mean_grad:    *mut f32,
      var_grad:     *mut f32,
      in_delta:     *mut f32,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_conv_bnorm_rfwd_var_batch(
      in_act:       *const f32,
      spatial_dim:  c_int,
      num_channels: c_int,
      batch_size:   c_int,
      in_r_act:     *const f32,
      mean:         *const f32,
      r_mean:       *const f32,
      r_var:        *mut f32,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_conv_bnorm_rfwd_batch(
      in_act:       *const f32,
      spatial_dim:  c_int,
      num_channels: c_int,
      batch_size:   c_int,
      in_r_act:     *const f32,
      mean:         *const f32,
      r_mean:       *const f32,
      var:          *const f32,
      r_var:        *const f32,
      epsilon:      f32,
      out_r_act:    *mut f32,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_conv_smooth_bnorm_fwd_mean_var_batch(
      in_act:       *const f32,
      spatial_dim:  c_int,
      num_channels: c_int,
      batch_size:   c_int,
      prev_mean:    *const f32,
      alpha:        f32,
      mean:         *mut f32,
      var:          *mut f32,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_conv_smooth_bnorm_bwd_batch(
      in_act:       *const f32,
      spatial_dim:  c_int,
      num_channels: c_int,
      batch_size:   c_int,
      out_delta:    *const f32,
      prev_mean:    *const f32,
      mean:         *const f32,
      var:          *const f32,
      epsilon:      f32,
      alpha:        f32,
      mean_grad:    *mut f32,
      var_grad:     *mut f32,
      in_delta:     *mut f32,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_conv_smooth_bnorm_rfwd_batch(
      in_act:       *const f32,
      spatial_dim:  c_int,
      num_channels: c_int,
      batch_size:   c_int,
      in_r_act:     *const f32,
      prev_mean:    *const f32,
      mean:         *const f32,
      var:          *const f32,
      epsilon:      f32,
      alpha:        f32,
      r_mean:       *mut f32,
      r_var:        *mut f32,
      out_r_act:    *mut f32,
      stream:       cudaStream_t,
  );

  pub fn rembrandt_kernel_estimate_conv_mean_batch(
      src:          *const f32,
      spatial_dim:  c_int,
      num_channels: c_int,
      batch_size:   c_int,
      mean:         *mut f32,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_kernel_estimate_conv_mean_fast_batch(
      src:          *const f32,
      spatial_dim:  c_int,
      num_channels: c_int,
      batch_size:   c_int,
      mean:         *mut f32,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_kernel_estimate_conv_var_batch(
      src:          *const f32,
      spatial_dim:  c_int,
      num_channels: c_int,
      batch_size:   c_int,
      mean:         *const f32,
      var:          *mut f32,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_kernel_estimate_conv_var_fast_batch(
      src:          *const f32,
      spatial_dim:  c_int,
      num_channels: c_int,
      batch_size:   c_int,
      mean:         *const f32,
      var:          *mut f32,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_kernel_estimate_online_var(
      mean_batch:   *const f32,
      len:          c_int,
      var_batch:    *const f32,
      mean_acc:     *const f32,
      batch_size:   c_int,
      acc_size:     c_int,
      var_acc:      *mut f32,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_kernel_estimate_invstd(
      var:          *const f32,
      len:          c_int,
      epsilon:      f32,
      istd:         *mut f32,
      stream:       cudaStream_t,
  );

  pub fn rembrandt_kernel_inner_prod_blockreduce_batch(
      xs:           *const f32,
      frame_len:    c_int,
      batch_size:   c_int,
      ws:           *const f32,
      alpha:        f32,
      sum:          *mut f32,
      stream:       cudaStream_t,
  );

  pub fn rembrandt_rect_fwd(
      in_act:       *const f32,
      dim:          c_int,
      out_act:      *mut f32,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_rect_bwd(
      in_act:       *const f32,
      dim:          c_int,
      out_delta:    *const f32,
      in_delta:     *mut f32,
      stream:       cudaStream_t,
  );

  pub fn rembrandt_kernel_softmax_kl_loss_fwd_batch(
      out_act:      *const f32,
      frame_len:    c_int,
      batch_size:   c_int,
      label_cats:   *const i32,
      weights:      *const f32,
      targets:      *const f32,
      out_loss:     *mut f32,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_kernel_softmax_kl_loss_bwd_batch(
      out_act:      *const f32,
      frame_len:    c_int,
      batch_size:   c_int,
      label_cats:   *const i32,
      weights:      *const f32,
      targets:      *const f32,
      in_delta:     *mut f32,
      stream:       cudaStream_t,
  );
  // XXX: This actually computes the "normalized" R-activation.
  pub fn rembrandt_kernel_softmax_r_fwd_batch(
      in_r_act:     *const f32,
      frame_len:    c_int,
      batch_size:   c_int,
      mix_in_r_act: *const f32,
      out_r_act:    *mut f32,
      stream:       cudaStream_t,
  );
  // XXX: This actually uses the "normalized" R-activation.
  pub fn rembrandt_kernel_softmax_kl_loss_r_fwd_batch(
      out_r_act:    *const f32,
      frame_len:    c_int,
      batch_size:   c_int,
      label_cats:   *const i32,
      //r_weights:    *const f32,
      out_r_loss:   *mut f32,
      stream:       cudaStream_t,
  );

  // Image kernels.
  pub fn rembrandt_kernel_image3_bicubic_scale(
      in_pixels:    *const f32,
      in_width:     c_int,
      in_height:    c_int,
      channels:     c_int,
      out_pixels:   *mut f32,
      out_width:    c_int,
      out_height:   c_int,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_kernel_image3_catmullrom_scale(
      in_pixels:    *const f32,
      in_width:     c_int,
      in_height:    c_int,
      channels:     c_int,
      out_pixels:   *mut f32,
      out_width:    c_int,
      out_height:   c_int,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_kernel_image3_2x2_bilinear_scale(
      in_pixels:    *const f32,
      in_width:     c_int,
      in_height:    c_int,
      channels:     c_int,
      out_pixels:   *mut f32,
      out_width:    c_int,
      out_height:   c_int,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_kernel_image3_crop(
      in_pixels:    *const f32,
      in_width:     c_int,
      in_height:    c_int,
      channels:     c_int,
      x_offset:     c_int,
      y_offset:     c_int,
      out_pixels:   *mut f32,
      crop_width:   c_int,
      crop_height:  c_int,
      stream:       cudaStream_t,
  );
  pub fn rembrandt_kernel_batch_image3_crop(
      in_pixels:    *const f32,
      in_stride:    c_int,
      batch_size:   c_int,
      in_widths:    *const c_int,
      in_heights:   *const c_int,
      channels:     c_int,
      in_x_offsets: *const c_int,
      in_y_offsets: *const c_int,
      out_pixels:   *mut f32,
      crop_width:   c_int,
      crop_height:  c_int,
      stream:       cudaStream_t,
  );

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
  pub fn rembrandt_kernel_batch_map_preproc_pca3_noise(
      src:          *const f32,
      spatial_size: c_int,
      batch_size:   c_int,
      alphas:       *const f32,
      evals:        *const f32,
      evecs:        *const f32,
      dst:          *mut f32,
      stream:       cudaStream_t,
  );
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
  pub fn rembrandt_kernel_batch_map_softmax_kl_loss1(
      out_act:      *const f32,
      num_channels: c_int,
      batch_size:   c_int,
      labels:       *const i32,
      weights:      *const f32,
      loss_factor:  f32,
      loss1:        *mut f32,
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

  // Batch block map kernels.
  pub fn rembrandt_kernel_batch_blockmap256_flip(
      src:          *const f32,
      num_channels: c_int,
      batch_size:   c_int,
      dst:          *mut f32,
      stream:       cudaStream_t,
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
