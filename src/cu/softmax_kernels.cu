#include <cuda_runtime_api.h>
#include <stdint.h>

__global__ void softmax_kl_loss_fwd_batch_kernel(
    const float *out_act,
    int frame_len,
    int batch_size,
    const int32_t *label_cats,
    const float *weights,
    const float *targets,
    float *out_loss)
{
  int batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (batch_idx < batch_size) {
    int cat_i = label_cats[batch_idx];
    int idx = cat_i + batch_idx * frame_len;
    float x = -logf(out_act[idx]) * weights[cat_i] * targets[cat_i];
    out_loss[batch_idx] = x;
  }
}

extern "C" void rembrandt_kernel_softmax_kl_loss_fwd_batch(
    const float *out_act,
    int frame_len,
    int batch_size,
    const int32_t *label_cats,
    const float *weights,
    const float *targets,
    float *out_loss,
    cudaStream_t stream)
{
  int n = batch_size;
  softmax_kl_loss_fwd_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      out_act, frame_len, batch_size,
      label_cats,
      weights,
      targets,
      out_loss);
}

__global__ void softmax_kl_loss_bwd_batch_kernel(
    const float *out_act,
    int frame_len,
    int batch_size,
    const int32_t *label_cats,
    const float *weights,
    const float *targets,
    float *in_delta)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int i = idx % frame_len;
  int batch_idx = idx / frame_len;
  if ((i < frame_len) && (batch_idx < batch_size)) {
    int cat_i = label_cats[batch_idx];
    float dx = out_act[idx];
    if (i == cat_i) {
      dx -= 1.0f;
    }
    dx *= weights[cat_i] * targets[cat_i];
    in_delta[idx] = dx;
  }
}

extern "C" void rembrandt_kernel_softmax_kl_loss_bwd_batch(
    const float *out_act,
    int frame_len,
    int batch_size,
    const int32_t *label_cats,
    const float *weights,
    const float *targets,
    float *in_delta,
    cudaStream_t stream)
{
  int n = frame_len * batch_size;
  softmax_kl_loss_bwd_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      out_act, frame_len, batch_size,
      label_cats,
      weights,
      targets,
      in_delta);
}

__global__ void softmax_r_fwd_batch_kernel(
    const float *out_act,
    int frame_len,
    int batch_size,
    const float *in_r_act,
    const float *mix_in_r_act,
    float *out_r_act)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int i = idx % frame_len;
  int batch_idx = idx / frame_len;
  if ((i < frame_len) && (batch_idx < batch_size)) {
    float x = out_act[idx];
    x *= in_r_act[idx] - mix_in_r_act[batch_idx];
    out_r_act[idx] = x;
  }
}

extern "C" void rembrandt_kernel_softmax_r_fwd_batch(
    const float *out_act,
    int frame_len,
    int batch_size,
    const float *in_r_act,
    const float *mix_in_r_act,
    float *out_r_act,
    cudaStream_t stream)
{
  int n = frame_len * batch_size;
  softmax_r_fwd_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      out_act, frame_len, batch_size,
      in_r_act,
      mix_in_r_act,
      out_r_act);
}

__global__ void softmax_kl_loss_r_fwd_batch_kernel(
    const float *out_act,
    int frame_len,
    int batch_size,
    const float *out_r_act,
    const int32_t *label_cats,
    float *out_r_loss)
{
  int batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (batch_idx < batch_size) {
    int cat_i = label_cats[batch_idx];
    int idx = cat_i + batch_idx * frame_len;
    float x = -out_r_act[idx] / out_act[idx];
    out_r_loss[batch_idx] = x;
  }
}

extern "C" void rembrandt_kernel_softmax_kl_loss_r_fwd_batch(
    const float *out_act,
    int frame_len,
    int batch_size,
    const float *out_r_act,
    const int32_t *label_cats,
    float *out_r_loss,
    cudaStream_t stream)
{
  int n = batch_size;
  softmax_kl_loss_r_fwd_batch_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      out_act, frame_len, batch_size,
      out_r_act,
      label_cats,
      out_r_loss);
}
