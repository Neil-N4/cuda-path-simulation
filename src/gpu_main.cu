#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "argparse.hpp"

namespace {

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t err__ = (call);                                           \
    if (err__ != cudaSuccess) {                                                 \
      std::cerr << "CUDA error: " << cudaGetErrorString(err__) << " @ "        \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

constexpr int kThreadsPerBlock = 256;
constexpr int kStreams = 4;
constexpr double kZ95 = 1.959963984540054;

double normal_cdf(double x) {
  return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double black_scholes_call(double s0, double strike, double rate, double vol, double maturity) {
  if (maturity <= 0.0 || vol <= 0.0 || s0 <= 0.0 || strike <= 0.0) {
    return std::max(s0 - strike, 0.0);
  }
  const double sqrt_t = std::sqrt(maturity);
  const double d1 =
      (std::log(s0 / strike) + (rate + 0.5 * vol * vol) * maturity) / (vol * sqrt_t);
  const double d2 = d1 - vol * sqrt_t;
  return s0 * normal_cdf(d1) - strike * std::exp(-rate * maturity) * normal_cdf(d2);
}

struct WorkShape {
  std::uint64_t trajectories = 0;
  std::uint64_t sample_count = 0;
};

WorkShape compute_work_shape(const SimConfig& cfg) {
  WorkShape s;
  if (cfg.antithetic) {
    if (cfg.paths < 2) return s;
    s.trajectories = (cfg.paths % 2 == 0) ? cfg.paths : (cfg.paths - 1);
    s.sample_count = s.trajectories / 2;
  } else {
    s.trajectories = cfg.paths;
    s.sample_count = cfg.paths;
  }
  return s;
}

__global__ void init_states(curandStatePhilox4_32_10_t* states,
                            std::uint64_t n,
                            std::uint64_t seed,
                            std::uint64_t offset) {
  const std::uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    curand_init(seed, idx, offset, &states[idx]);
  }
}

__global__ void simulate_gbm_kernel(const curandStatePhilox4_32_10_t* states_in,
                                    curandStatePhilox4_32_10_t* states_out,
                                    float* block_sums,
                                    float* block_sums_sq,
                                    std::uint64_t samples,
                                    int steps,
                                    float s0,
                                    float strike,
                                    float rate,
                                    float vol,
                                    float maturity,
                                    int antithetic) {
  extern __shared__ float shared[];
  float* s_sum = shared;
  float* s_sum_sq = shared + blockDim.x;

  const std::uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  float sample_payoff = 0.0f;

  if (idx < samples) {
    curandStatePhilox4_32_10_t local_state = states_in[idx];
    const float dt = maturity / static_cast<float>(steps);
    const float drift = (rate - 0.5f * vol * vol) * dt;
    const float vol_step = vol * sqrtf(dt);

    float st_a = s0;
    float st_b = s0;
    int step = 0;
    for (; step + 4 <= steps; step += 4) {
      const float4 z4 = curand_normal4(&local_state);
      st_a = st_a * __expf(drift + vol_step * z4.x);
      st_a = st_a * __expf(drift + vol_step * z4.y);
      st_a = st_a * __expf(drift + vol_step * z4.z);
      st_a = st_a * __expf(drift + vol_step * z4.w);
      if (antithetic) {
        st_b = st_b * __expf(drift - vol_step * z4.x);
        st_b = st_b * __expf(drift - vol_step * z4.y);
        st_b = st_b * __expf(drift - vol_step * z4.z);
        st_b = st_b * __expf(drift - vol_step * z4.w);
      }
    }
    for (; step < steps; ++step) {
      const float z = curand_normal(&local_state);
      st_a = st_a * __expf(drift + vol_step * z);
      if (antithetic) {
        st_b = st_b * __expf(drift - vol_step * z);
      }
    }

    const float payoff_a = fmaxf(st_a - strike, 0.0f);
    if (antithetic) {
      const float payoff_b = fmaxf(st_b - strike, 0.0f);
      sample_payoff = 0.5f * (payoff_a + payoff_b);
    } else {
      sample_payoff = payoff_a;
    }

    states_out[idx] = local_state;
  }

  s_sum[threadIdx.x] = sample_payoff;
  s_sum_sq[threadIdx.x] = sample_payoff * sample_payoff;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
      s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    block_sums[blockIdx.x] = s_sum[0];
    block_sums_sq[blockIdx.x] = s_sum_sq[0];
  }
}

struct StreamBuffers {
  std::uint64_t chunk_samples = 0;
  std::uint64_t blocks = 0;
  curandStatePhilox4_32_10_t* d_state_a = nullptr;
  curandStatePhilox4_32_10_t* d_state_b = nullptr;
  float* d_partial_sum = nullptr;
  float* d_partial_sq = nullptr;
  float* h_partial_sum = nullptr;
  float* h_partial_sq = nullptr;
};

void allocate_stream_buffers(StreamBuffers& b) {
  CUDA_CHECK(cudaMalloc(&b.d_state_a, b.chunk_samples * sizeof(curandStatePhilox4_32_10_t)));
  CUDA_CHECK(cudaMalloc(&b.d_state_b, b.chunk_samples * sizeof(curandStatePhilox4_32_10_t)));
  CUDA_CHECK(cudaMalloc(&b.d_partial_sum, b.blocks * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b.d_partial_sq, b.blocks * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&b.h_partial_sum, b.blocks * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&b.h_partial_sq, b.blocks * sizeof(float)));
}

void free_stream_buffers(StreamBuffers& b) {
  if (b.d_state_a) CUDA_CHECK(cudaFree(b.d_state_a));
  if (b.d_state_b) CUDA_CHECK(cudaFree(b.d_state_b));
  if (b.d_partial_sum) CUDA_CHECK(cudaFree(b.d_partial_sum));
  if (b.d_partial_sq) CUDA_CHECK(cudaFree(b.d_partial_sq));
  if (b.h_partial_sum) CUDA_CHECK(cudaFreeHost(b.h_partial_sum));
  if (b.h_partial_sq) CUDA_CHECK(cudaFreeHost(b.h_partial_sq));
}

SimResult run_gpu(const SimConfig& cfg) {
  const WorkShape shape = compute_work_shape(cfg);
  if (shape.sample_count == 0) {
    return {};
  }

  const std::uint64_t total_samples = shape.sample_count;
  const std::uint64_t chunk_samples = (total_samples + kStreams - 1) / kStreams;

  std::vector<cudaStream_t> streams(kStreams);
  std::vector<StreamBuffers> buffers(kStreams);

  for (int s = 0; s < kStreams; ++s) {
    CUDA_CHECK(cudaStreamCreate(&streams[s]));

    const std::uint64_t start = static_cast<std::uint64_t>(s) * chunk_samples;
    const std::uint64_t remaining = (start < total_samples) ? (total_samples - start) : 0;
    const std::uint64_t this_chunk = std::min(chunk_samples, remaining);
    buffers[s].chunk_samples = this_chunk;
    buffers[s].blocks = (this_chunk + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (this_chunk > 0) {
      allocate_stream_buffers(buffers[s]);
    }
  }

  const auto t0 = std::chrono::high_resolution_clock::now();

  for (int s = 0; s < kStreams; ++s) {
    auto& b = buffers[s];
    if (b.chunk_samples == 0) continue;

    const std::uint64_t global_offset = static_cast<std::uint64_t>(s) * chunk_samples;
    const dim3 grid(static_cast<unsigned int>(b.blocks));
    const dim3 block(kThreadsPerBlock);

    init_states<<<grid, block, 0, streams[s]>>>(b.d_state_a, b.chunk_samples, cfg.seed, global_offset);

    const std::size_t shared_bytes = 2 * kThreadsPerBlock * sizeof(float);
    simulate_gbm_kernel<<<grid, block, shared_bytes, streams[s]>>>(
        b.d_state_a,
        b.d_state_b,
        b.d_partial_sum,
        b.d_partial_sq,
        b.chunk_samples,
        cfg.steps,
        cfg.s0,
        cfg.strike,
        cfg.rate,
        cfg.volatility,
        cfg.maturity,
        cfg.antithetic ? 1 : 0);

    CUDA_CHECK(cudaMemcpyAsync(
        b.h_partial_sum, b.d_partial_sum, b.blocks * sizeof(float), cudaMemcpyDeviceToHost, streams[s]));
    CUDA_CHECK(cudaMemcpyAsync(
        b.h_partial_sq, b.d_partial_sq, b.blocks * sizeof(float), cudaMemcpyDeviceToHost, streams[s]));
  }

  double sum = 0.0;
  double sum_sq = 0.0;
  for (int s = 0; s < kStreams; ++s) {
    auto& b = buffers[s];
    if (b.chunk_samples == 0) continue;
    CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    for (std::uint64_t i = 0; i < b.blocks; ++i) {
      sum += b.h_partial_sum[i];
      sum_sq += b.h_partial_sq[i];
    }
  }

  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  for (int s = 0; s < kStreams; ++s) {
    free_stream_buffers(buffers[s]);
    CUDA_CHECK(cudaStreamDestroy(streams[s]));
  }

  const double n = static_cast<double>(shape.sample_count);
  const double mean = sum / n;
  const double variance = std::max(sum_sq / n - mean * mean, 0.0);
  const double stddev = std::sqrt(variance);
  const double stderr = stddev / std::sqrt(n);
  const double discount = std::exp(-cfg.rate * cfg.maturity);
  const double price = discount * mean;
  const double bs = black_scholes_call(cfg.s0, cfg.strike, cfg.rate, cfg.volatility, cfg.maturity);

  SimResult out;
  out.payoff_mean = mean;
  out.payoff_stddev = stddev;
  out.payoff_stderr = stderr;
  out.option_price = price;
  out.ci95_low = discount * (mean - kZ95 * stderr);
  out.ci95_high = discount * (mean + kZ95 * stderr);
  out.bs_price = bs;
  out.abs_error_bs = std::abs(price - bs);
  out.sample_count = shape.sample_count;
  out.trajectory_count = shape.trajectories;
  out.runtime_ms = ms;
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  SimConfig cfg;
  if (!parse_args(argc, argv, cfg)) return 1;

  CUDA_CHECK(cudaSetDevice(0));
  const SimResult result = run_gpu(cfg);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "engine=gpu\n";
  std::cout << "paths=" << cfg.paths << "\n";
  std::cout << "trajectories=" << result.trajectory_count << "\n";
  std::cout << "samples=" << result.sample_count << "\n";
  std::cout << "antithetic=" << (cfg.antithetic ? 1 : 0) << "\n";
  std::cout << "steps=" << cfg.steps << "\n";
  std::cout << "price=" << result.option_price << "\n";
  std::cout << "ci95_low=" << result.ci95_low << "\n";
  std::cout << "ci95_high=" << result.ci95_high << "\n";
  std::cout << "bs_price=" << result.bs_price << "\n";
  std::cout << "abs_error_bs=" << result.abs_error_bs << "\n";
  std::cout << "payoff_mean=" << result.payoff_mean << "\n";
  std::cout << "payoff_stddev=" << result.payoff_stddev << "\n";
  std::cout << "payoff_stderr=" << result.payoff_stderr << "\n";
  std::cout << "runtime_ms=" << result.runtime_ms << "\n";
  return 0;
}
