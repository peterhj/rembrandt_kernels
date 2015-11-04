#include "common.h"
#include <cuda_runtime_api.h>

__global__ void reduce_sum_level_1_kernel(
    const float *xs,
    int n,
    float *sum_block)
{
  __shared__ float cache[1024];
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  if (i < n) {
    cache[tid] = xs[i];
  } else {
    cache[tid] = 0.0;
  }
  __syncthreads();
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0 && (i + s) < n) {
      cache[tid] += cache[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    sum_block[blockIdx.x] = cache[0];
  }
}

__global__ void reduce_count_nonzero_level_1_kernel(
    const float *xs,
    int n,
    float *count_block)
{
  __shared__ float cache[1024];
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  if (i < n) {
    float abs_x_i = fabsf(xs[i]);
    if (abs_x_i > 0.0) {
      cache[tid] = 1.0;
    } else {
      cache[tid] = 0.0;
    }
  } else {
    cache[tid] = 0.0;
  }
  __syncthreads();
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0 && (i + s) < n) {
      cache[tid] += cache[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    count_block[blockIdx.x] = cache[0];
  }
}

__global__ void reduce_count_sparse_level_1_kernel(
    const float *xs,
    int n,
    float *count_block,
    float threshold)
{
  __shared__ float cache[1024];
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  if (i < n) {
    float abs_x_i = fabsf(xs[i]);
    if (abs_x_i > threshold) {
      cache[tid] = 1.0;
    } else {
      cache[tid] = 0.0;
    }
  } else {
    cache[tid] = 0.0;
  }
  __syncthreads();
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0 && (i + s) < n) {
      cache[tid] += cache[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    count_block[blockIdx.x] = cache[0];
  }
}

__global__ void reduce_count_range_level_1_kernel(
    const float *xs,
    int n,
    float *count_block,
    float lo_threshold,
    float hi_threshold)
{
  __shared__ float cache[1024];
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  if (i < n) {
    float abs_x_i = fabsf(xs[i]);
    if (abs_x_i > lo_threshold && abs_x_i < hi_threshold) {
      cache[tid] = 1.0;
    } else {
      cache[tid] = 0.0;
    }
  } else {
    cache[tid] = 0.0;
  }
  __syncthreads();
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0 && (i + s) < n) {
      cache[tid] += cache[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    count_block[blockIdx.x] = cache[0];
  }
}

extern "C" void rembrandt_kernel_reduce_count_nonzero(
    const float *xs,
    int n,
    float *count_block_tmp,
    float *count,
    cudaStream_t stream)
{
  float *count_block_tmp1 = count_block_tmp;
  float *count_block_tmp2 = count_block_tmp + (n+1024-1)/1024;
  int num_levels = 0;
  int level_size = n;
  for (;;) {
    num_levels += 1;
    level_size = (level_size+1024-1)/1024;
    if (level_size <= 1) {
      break;
    }
  }
  if (num_levels == 1) {
    reduce_count_nonzero_level_1_kernel<<<1, (n+32-1)/32*32, 0, stream>>>(
        xs, n, count);
  } else {
    reduce_count_nonzero_level_1_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
        xs, n, count_block_tmp1);
    // TODO(20151103)
    int level_size = (n+1024-1)/1024;
    for (int level = 1; level < num_levels - 1; level += 1) {
      if ((level % 2) == 1) {
        reduce_sum_level_1_kernel<<<(level_size+1024-1)/1024, 1024, 0, stream>>>(
            count_block_tmp1, level_size, count_block_tmp2);
      } else {
        reduce_sum_level_1_kernel<<<(level_size+1024-1)/1024, 1024, 0, stream>>>(
            count_block_tmp2, level_size, count_block_tmp1);
      }
      level_size = (level_size+1024-1)/1024;
    }
    if (((num_levels - 1) % 2) == 1) {
      reduce_sum_level_1_kernel<<<1, (level_size+32-1)/32*32, 0, stream>>>(
          count_block_tmp1, level_size, count);
    } else {
      reduce_sum_level_1_kernel<<<1, (level_size+32-1)/32*32, 0, stream>>>(
          count_block_tmp2, level_size, count);
    }
  }
  CUDA_POST_KERNEL_CHECK;
}

extern "C" void rembrandt_kernel_reduce_count_sparse(
    const float *xs,
    int n,
    float *count_block_tmp,
    float *count,
    float threshold,
    cudaStream_t stream)
{
  float *count_block_tmp1 = count_block_tmp;
  float *count_block_tmp2 = count_block_tmp + (n+1024-1)/1024;
  int num_levels = 0;
  int level_size = n;
  for (;;) {
    num_levels += 1;
    level_size = (level_size+1024-1)/1024;
    if (level_size <= 1) {
      break;
    }
  }
  if (num_levels == 1) {
    reduce_count_sparse_level_1_kernel<<<1, (n+32-1)/32*32, 0, stream>>>(
        xs, n, count, threshold);
  } else {
    reduce_count_sparse_level_1_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
        xs, n, count_block_tmp1, threshold);
    // TODO(20151103)
    int level_size = (n+1024-1)/1024;
    for (int level = 1; level < num_levels - 1; level += 1) {
      if ((level % 2) == 1) {
        reduce_sum_level_1_kernel<<<(level_size+1024-1)/1024, 1024, 0, stream>>>(
            count_block_tmp1, level_size, count_block_tmp2);
      } else {
        reduce_sum_level_1_kernel<<<(level_size+1024-1)/1024, 1024, 0, stream>>>(
            count_block_tmp2, level_size, count_block_tmp1);
      }
      level_size = (level_size+1024-1)/1024;
    }
    if (((num_levels - 1) % 2) == 1) {
      reduce_sum_level_1_kernel<<<1, (level_size+32-1)/32*32, 0, stream>>>(
          count_block_tmp1, level_size, count);
    } else {
      reduce_sum_level_1_kernel<<<1, (level_size+32-1)/32*32, 0, stream>>>(
          count_block_tmp2, level_size, count);
    }
  }
  CUDA_POST_KERNEL_CHECK;
}

extern "C" void rembrandt_kernel_reduce_count_range(
    const float *xs,
    int n,
    float *count_block_tmp,
    float *count,
    float lo_threshold,
    float hi_threshold,
    cudaStream_t stream)
{
  float *count_block_tmp1 = count_block_tmp;
  float *count_block_tmp2 = count_block_tmp + (n+1024-1)/1024;
  int num_levels = 0;
  int level_size = n;
  for (;;) {
    num_levels += 1;
    level_size = (level_size+1024-1)/1024;
    if (level_size <= 1) {
      break;
    }
  }
  if (num_levels == 1) {
    reduce_count_range_level_1_kernel<<<1, (n+32-1)/32*32, 0, stream>>>(
        xs, n, count, lo_threshold, hi_threshold);
  } else {
    reduce_count_range_level_1_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
        xs, n, count_block_tmp1, lo_threshold, hi_threshold);
    // TODO(20151103)
    int level_size = (n+1024-1)/1024;
    for (int level = 1; level < num_levels - 1; level += 1) {
      if ((level % 2) == 1) {
        reduce_sum_level_1_kernel<<<(level_size+1024-1)/1024, 1024, 0, stream>>>(
            count_block_tmp1, level_size, count_block_tmp2);
      } else {
        reduce_sum_level_1_kernel<<<(level_size+1024-1)/1024, 1024, 0, stream>>>(
            count_block_tmp2, level_size, count_block_tmp1);
      }
      level_size = (level_size+1024-1)/1024;
    }
    if (((num_levels - 1) % 2) == 1) {
      reduce_sum_level_1_kernel<<<1, (level_size+32-1)/32*32, 0, stream>>>(
          count_block_tmp1, level_size, count);
    } else {
      reduce_sum_level_1_kernel<<<1, (level_size+32-1)/32*32, 0, stream>>>(
          count_block_tmp2, level_size, count);
    }
  }
  CUDA_POST_KERNEL_CHECK;
}

__global__ void blockreduce_argmax_float_kernel(
    const int n,
    const float *x,
    float *x_max_block,
    //int *x_argmax_block)
    float *x_argmax_block)
{
  __shared__ float cache[1024];
  __shared__ int cache_idx[1024];
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  if (i < n) {
    cache[tid]      = x[i];
    cache_idx[tid]  = i;
  }
  __syncthreads();
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0 && (i + s) < n && cache[tid] < cache[tid + s]) {
      cache[tid]      = cache[tid + s];
      cache_idx[tid]  = cache_idx[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    x_max_block[blockIdx.x] = cache[0];
    if (x_argmax_block != NULL) {
      //x_argmax_block[blockIdx.x] = cache_idx[0];
      x_argmax_block[blockIdx.x] = (float)(cache_idx[0]);
    }
  }
}

extern "C" void rembrandt_kernel_blockreduce_argmax_float(
    int n,
    const float *x,
    float *max_block,
    //int *idx_block,
    float *idx_block,
    cudaStream_t stream)
{
  blockreduce_argmax_float_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      n, x, max_block, idx_block);
  CUDA_POST_KERNEL_CHECK;
}

extern "C" void rembrandt_kernel_blockreduce_argmin_float(
    int n,
    const float *x,
    float *min_block,
    int *idx_block,
    cudaStream_t stream)
{
  assert(0 && "unimplemented!");
}

__global__ void blockreduce_argmax_kernel(
    const int n,
    const float *x,
    float *x_max_block,
    int *x_argmax_block)
{
  __shared__ float cache[1024];
  __shared__ int cache_idx[1024];
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  if (i < n) {
    cache[tid]      = x[i];
    cache_idx[tid]  = i;
  }
  __syncthreads();
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0 && (i + s) < n && cache[tid] < cache[tid + s]) {
      cache[tid]      = cache[tid + s];
      cache_idx[tid]  = cache_idx[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    x_max_block[blockIdx.x] = cache[0];
    if (x_argmax_block != NULL) {
      x_argmax_block[blockIdx.x] = cache_idx[0];
    }
  }
}

extern "C" void rembrandt_kernel_blockreduce_argmax(
    int n,
    const float *x,
    float *max_block,
    int *idx_block,
    cudaStream_t stream)
{
  blockreduce_argmax_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      n, x, max_block, idx_block);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void blockreduce_sum_kernel(
    const int n,
    const float *x,
    float *x_sum_block)
{
  __shared__ float cache[1024];
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  if (i < n) {
    cache[tid] = x[i];
  } else {
    cache[tid] = 0.0;
  }
  __syncthreads();
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0 && (i + s) < n) {
      cache[tid] += cache[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    x_sum_block[blockIdx.x] = cache[0];
  }
}

extern "C" void rembrandt_kernel_blockreduce_sum(
    int n,
    const float *x,
    float *sum_block,
    cudaStream_t stream)
{
  blockreduce_sum_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      n, x, sum_block);
  CUDA_POST_KERNEL_CHECK;
}
