#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define HDQUAL __host__ __device__
#else
#define HDQUAL
#endif

namespace lowdisc {

HDQUAL inline std::uint32_t reverse_bits32(std::uint32_t x) {
  x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);
  x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);
  x = ((x & 0x0F0F0F0Fu) << 4) | ((x >> 4) & 0x0F0F0F0Fu);
  x = ((x & 0x00FF00FFu) << 8) | ((x >> 8) & 0x00FF00FFu);
  return (x << 16) | (x >> 16);
}

HDQUAL inline std::uint32_t mix32(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

HDQUAL inline float sobol_uniform(std::uint64_t sample_idx, int dim, std::uint64_t seed) {
  const std::uint32_t scramble = mix32(static_cast<std::uint32_t>(seed) ^ (0x9e3779b9u * static_cast<std::uint32_t>(dim + 1)));
  const std::uint32_t code = reverse_bits32(static_cast<std::uint32_t>(sample_idx + 1u)) ^ scramble;
  const float u = (static_cast<float>(code) + 0.5f) * 2.3283064365386963e-10f;  // 2^-32
  return (u < 1.0e-7f) ? 1.0e-7f : ((u > (1.0f - 1.0e-7f)) ? (1.0f - 1.0e-7f) : u);
}

// Acklam rational approximation for inverse normal CDF.
HDQUAL inline float inv_norm(float p) {
  const float a1 = -3.9696830e+01f;
  const float a2 = 2.2094609e+02f;
  const float a3 = -2.7592851e+02f;
  const float a4 = 1.3835775e+02f;
  const float a5 = -3.0664798e+01f;
  const float a6 = 2.5066283e+00f;

  const float b1 = -5.4476099e+01f;
  const float b2 = 1.6158584e+02f;
  const float b3 = -1.5569898e+02f;
  const float b4 = 6.6801313e+01f;
  const float b5 = -1.3280682e+01f;

  const float c1 = -7.7848940e-03f;
  const float c2 = -3.2239646e-01f;
  const float c3 = -2.4007583e+00f;
  const float c4 = -2.5497324e+00f;
  const float c5 = 4.3746643e+00f;
  const float c6 = 2.9381640e+00f;

  const float d1 = 7.7846957e-03f;
  const float d2 = 3.2246712e-01f;
  const float d3 = 2.4451342e+00f;
  const float d4 = 3.7544087e+00f;

  const float plow = 0.02425f;
  const float phigh = 1.0f - plow;

  if (p < plow) {
    const float q = sqrtf(-2.0f * logf(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0f);
  }
  if (p > phigh) {
    const float q = sqrtf(-2.0f * logf(1.0f - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0f);
  }

  const float q = p - 0.5f;
  const float r = q * q;
  return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
         (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0f);
}

HDQUAL inline float sobol_normal(std::uint64_t sample_idx, int dim, std::uint64_t seed) {
  return inv_norm(sobol_uniform(sample_idx, dim, seed));
}

}  // namespace lowdisc

#undef HDQUAL
