#include <cuda_runtime_api.h>
#include <stdint.h>

__device__ float w0(float a) {
  return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);
}

__device__ float w1(float a) {
  return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

__device__ float w2(float a) {
  return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

__device__ float w3(float a) {
  return (1.0f/6.0f)*(a*a*a);
}

__device__ float image3_tex2d_clamp(const float *pixels, int width, int height, int u, int v, int c) {
  int clamp_u = min(max(0, u), width-1);
  int clamp_v = min(max(0, v), height-1);
  return pixels[clamp_u + clamp_v * width + c * width * height];
}

__device__ float image3_bicubic_filter(
    float x,
    float a0,
    float a1,
    float a2,
    float a3)
{
  float r = a0 * w0(x);
  r += a1 * w1(x);
  r += a2 * w2(x);
  r += a3 * w3(x);
  return r;
}

__device__ float image3_bicubic_interpolate(
    const float *pixels,
    int width,
    int height,
    float u,
    float v,
    int c)
{
  u -= 0.5f;
  v -= 0.5f;
  float px = floorf(u);
  float py = floorf(v);
  float fx = u - px;
  float fy = v - py;
  int ipx = (int)px;
  int ipy = (int)py;
  return image3_bicubic_filter(fy,
      image3_bicubic_filter(fx,
          image3_tex2d_clamp(pixels, width, height, ipx-1, ipy-1, c),
          image3_tex2d_clamp(pixels, width, height, ipx,   ipy-1, c),
          image3_tex2d_clamp(pixels, width, height, ipx+1, ipy-1, c),
          image3_tex2d_clamp(pixels, width, height, ipx+2, ipy-1, c)),
      image3_bicubic_filter(fx,
          image3_tex2d_clamp(pixels, width, height, ipx-1, ipy,   c),
          image3_tex2d_clamp(pixels, width, height, ipx,   ipy,   c),
          image3_tex2d_clamp(pixels, width, height, ipx+1, ipy,   c),
          image3_tex2d_clamp(pixels, width, height, ipx+2, ipy,   c)),
      image3_bicubic_filter(fx,
          image3_tex2d_clamp(pixels, width, height, ipx-1, ipy+1, c),
          image3_tex2d_clamp(pixels, width, height, ipx,   ipy+1, c),
          image3_tex2d_clamp(pixels, width, height, ipx+1, ipy+1, c),
          image3_tex2d_clamp(pixels, width, height, ipx+2, ipy+1, c)),
      image3_bicubic_filter(fx,
          image3_tex2d_clamp(pixels, width, height, ipx-1, ipy+2, c),
          image3_tex2d_clamp(pixels, width, height, ipx,   ipy+2, c),
          image3_tex2d_clamp(pixels, width, height, ipx+1, ipy+2, c),
          image3_tex2d_clamp(pixels, width, height, ipx+2, ipy+2, c)));
}

__global__ void image3_bicubic_scale_kernel(
    const float *in_pixels,
    int in_width,
    int in_height,
    int channels,
    float *out_pixels,
    int out_width,
    int out_height)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int x = idx % out_width;
  int y = (idx / out_width) % out_height;
  int c = idx / (out_width * out_height);

  if ((x < out_width) && (y < out_height) && (c < channels)) {
    float u = ((float)x) / ((float)out_width) * ((float)in_width);
    float v = ((float)y) / ((float)out_height) * ((float)in_height);

    float interp_value = image3_bicubic_interpolate(in_pixels, in_width, in_height, u, v, c);

    out_pixels[x + y * out_width + c * out_width * out_height] = interp_value;
  }
}

extern "C" void rembrandt_kernel_image3_bicubic_scale(
    const float *in_pixels,
    int in_width,
    int in_height,
    int channels,
    float *out_pixels,
    int out_width,
    int out_height,
    cudaStream_t stream)
{
  int n = out_width * out_height * channels;
  image3_bicubic_scale_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_pixels,
      in_width,
      in_height,
      channels,
      out_pixels,
      out_width,
      out_height);
}
