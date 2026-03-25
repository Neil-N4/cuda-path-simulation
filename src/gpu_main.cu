#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
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

double normal_pdf(double x) {
  static constexpr double kInvSqrt2Pi = 0.39894228040143267794;
  return kInvSqrt2Pi * std::exp(-0.5 * x * x);
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

double black_scholes_delta(double s0, double strike, double rate, double vol, double maturity) {
  if (maturity <= 0.0 || vol <= 0.0 || s0 <= 0.0 || strike <= 0.0) {
    return (s0 > strike) ? 1.0 : 0.0;
  }
  const double sqrt_t = std::sqrt(maturity);
  const double d1 =
      (std::log(s0 / strike) + (rate + 0.5 * vol * vol) * maturity) / (vol * sqrt_t);
  return normal_cdf(d1);
}

double black_scholes_vega(double s0, double strike, double rate, double vol, double maturity) {
  if (maturity <= 0.0 || vol <= 0.0 || s0 <= 0.0 || strike <= 0.0) return 0.0;
  const double sqrt_t = std::sqrt(maturity);
  const double d1 =
      (std::log(s0 / strike) + (rate + 0.5 * vol * vol) * maturity) / (vol * sqrt_t);
  return s0 * normal_pdf(d1) * sqrt_t;
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

__device__ float payoff_value_device(int payoff_type, float terminal, float mean_path, int knocked_out, float strike) {
  if (payoff_type == static_cast<int>(PayoffType::EuropeanCall)) {
    return fmaxf(terminal - strike, 0.0f);
  }
  if (payoff_type == static_cast<int>(PayoffType::AsianCall)) {
    return fmaxf(mean_path - strike, 0.0f);
  }
  if (payoff_type == static_cast<int>(PayoffType::UpAndOutCall)) {
    if (knocked_out) return 0.0f;
    return fmaxf(terminal - strike, 0.0f);
  }
  return 0.0f;
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

__global__ void simulate_kernel(
    const curandStatePhilox4_32_10_t* states_in,
    curandStatePhilox4_32_10_t* states_out,
    float* block_y,
    float* block_y2,
    float* block_x,
    float* block_x2,
    float* block_yx,
    float* block_delta,
    float* block_vega,
    std::uint64_t samples,
    int steps,
    float s0,
    float strike,
    float rate,
    float vol,
    float maturity,
    float barrier,
    int antithetic,
    int payoff_type) {
  extern __shared__ float sh[];
  float* sh_y = sh;
  float* sh_y2 = sh + blockDim.x;
  float* sh_x = sh + 2 * blockDim.x;
  float* sh_x2 = sh + 3 * blockDim.x;
  float* sh_yx = sh + 4 * blockDim.x;
  float* sh_delta = sh + 5 * blockDim.x;
  float* sh_vega = sh + 6 * blockDim.x;

  const std::uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  float y = 0.0f, x = 0.0f, delta = 0.0f, vega = 0.0f;

  if (idx < samples) {
    curandStatePhilox4_32_10_t state = states_in[idx];
    const float dt = maturity / static_cast<float>(steps);
    const float sqrt_dt = sqrtf(dt);
    const float drift = (rate - 0.5f * vol * vol) * dt;
    const float vol_step = vol * sqrt_dt;
    const float discount = expf(-rate * maturity);
    const int greek_enabled = (payoff_type == static_cast<int>(PayoffType::EuropeanCall));

    float st_a = s0, st_b = s0;
    float mean_acc_a = 0.0f, mean_acc_b = 0.0f;
    int knock_a = 0, knock_b = 0;
    float w_a = 0.0f;

    int step = 0;
    for (; step + 4 <= steps; step += 4) {
      const float4 z4 = curand_normal4(&state);
      const float zvals[4] = {z4.x, z4.y, z4.z, z4.w};
      #pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float z = zvals[k];
        w_a += z * sqrt_dt;
        st_a = st_a * __expf(drift + vol_step * z);
        if (antithetic) st_b = st_b * __expf(drift - vol_step * z);
        mean_acc_a += st_a;
        if (antithetic) mean_acc_b += st_b;
        knock_a = knock_a || (st_a >= barrier);
        if (antithetic) knock_b = knock_b || (st_b >= barrier);
      }
    }
    for (; step < steps; ++step) {
      const float z = curand_normal(&state);
      w_a += z * sqrt_dt;
      st_a = st_a * __expf(drift + vol_step * z);
      if (antithetic) st_b = st_b * __expf(drift - vol_step * z);
      mean_acc_a += st_a;
      if (antithetic) mean_acc_b += st_b;
      knock_a = knock_a || (st_a >= barrier);
      if (antithetic) knock_b = knock_b || (st_b >= barrier);
    }

    const float mean_path_a = mean_acc_a / steps;
    const float payoff_a = payoff_value_device(payoff_type, st_a, mean_path_a, knock_a, strike);

    if (antithetic) {
      const float mean_path_b = mean_acc_b / steps;
      const float payoff_b = payoff_value_device(payoff_type, st_b, mean_path_b, knock_b, strike);
      y = 0.5f * discount * (payoff_a + payoff_b);
      x = 0.5f * discount * (st_a + st_b);
      if (greek_enabled) {
        const float ind_a = (st_a > strike) ? 1.0f : 0.0f;
        const float ind_b = (st_b > strike) ? 1.0f : 0.0f;
        const float d_a = discount * ind_a * st_a / s0;
        const float d_b = discount * ind_b * st_b / s0;
        const float v_a = discount * ind_a * st_a * (w_a - vol * maturity);
        const float v_b = discount * ind_b * st_b * (-w_a - vol * maturity);
        delta = 0.5f * (d_a + d_b);
        vega = 0.5f * (v_a + v_b);
      }
    } else {
      y = discount * payoff_a;
      x = discount * st_a;
      if (greek_enabled) {
        const float ind = (st_a > strike) ? 1.0f : 0.0f;
        delta = discount * ind * st_a / s0;
        vega = discount * ind * st_a * (w_a - vol * maturity);
      }
    }
    states_out[idx] = state;
  }

  sh_y[threadIdx.x] = y;
  sh_y2[threadIdx.x] = y * y;
  sh_x[threadIdx.x] = x;
  sh_x2[threadIdx.x] = x * x;
  sh_yx[threadIdx.x] = y * x;
  sh_delta[threadIdx.x] = delta;
  sh_vega[threadIdx.x] = vega;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sh_y[threadIdx.x] += sh_y[threadIdx.x + stride];
      sh_y2[threadIdx.x] += sh_y2[threadIdx.x + stride];
      sh_x[threadIdx.x] += sh_x[threadIdx.x + stride];
      sh_x2[threadIdx.x] += sh_x2[threadIdx.x + stride];
      sh_yx[threadIdx.x] += sh_yx[threadIdx.x + stride];
      sh_delta[threadIdx.x] += sh_delta[threadIdx.x + stride];
      sh_vega[threadIdx.x] += sh_vega[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    block_y[blockIdx.x] = sh_y[0];
    block_y2[blockIdx.x] = sh_y2[0];
    block_x[blockIdx.x] = sh_x[0];
    block_x2[blockIdx.x] = sh_x2[0];
    block_yx[blockIdx.x] = sh_yx[0];
    block_delta[blockIdx.x] = sh_delta[0];
    block_vega[blockIdx.x] = sh_vega[0];
  }
}

struct StreamBuffers {
  std::uint64_t chunk_samples = 0;
  std::uint64_t blocks = 0;
  curandStatePhilox4_32_10_t* d_state_a = nullptr;
  curandStatePhilox4_32_10_t* d_state_b = nullptr;
  float* d_y = nullptr;
  float* d_y2 = nullptr;
  float* d_x = nullptr;
  float* d_x2 = nullptr;
  float* d_yx = nullptr;
  float* d_delta = nullptr;
  float* d_vega = nullptr;
  float* h_y = nullptr;
  float* h_y2 = nullptr;
  float* h_x = nullptr;
  float* h_x2 = nullptr;
  float* h_yx = nullptr;
  float* h_delta = nullptr;
  float* h_vega = nullptr;
};

void alloc_metric_pair(float*& d, float*& h, std::uint64_t blocks) {
  CUDA_CHECK(cudaMalloc(&d, blocks * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h, blocks * sizeof(float)));
}

void free_metric_pair(float*& d, float*& h) {
  if (d) CUDA_CHECK(cudaFree(d));
  if (h) CUDA_CHECK(cudaFreeHost(h));
}

void allocate_stream_buffers(StreamBuffers& b) {
  CUDA_CHECK(cudaMalloc(&b.d_state_a, b.chunk_samples * sizeof(curandStatePhilox4_32_10_t)));
  CUDA_CHECK(cudaMalloc(&b.d_state_b, b.chunk_samples * sizeof(curandStatePhilox4_32_10_t)));
  alloc_metric_pair(b.d_y, b.h_y, b.blocks);
  alloc_metric_pair(b.d_y2, b.h_y2, b.blocks);
  alloc_metric_pair(b.d_x, b.h_x, b.blocks);
  alloc_metric_pair(b.d_x2, b.h_x2, b.blocks);
  alloc_metric_pair(b.d_yx, b.h_yx, b.blocks);
  alloc_metric_pair(b.d_delta, b.h_delta, b.blocks);
  alloc_metric_pair(b.d_vega, b.h_vega, b.blocks);
}

void free_stream_buffers(StreamBuffers& b) {
  if (b.d_state_a) CUDA_CHECK(cudaFree(b.d_state_a));
  if (b.d_state_b) CUDA_CHECK(cudaFree(b.d_state_b));
  free_metric_pair(b.d_y, b.h_y);
  free_metric_pair(b.d_y2, b.h_y2);
  free_metric_pair(b.d_x, b.h_x);
  free_metric_pair(b.d_x2, b.h_x2);
  free_metric_pair(b.d_yx, b.h_yx);
  free_metric_pair(b.d_delta, b.h_delta);
  free_metric_pair(b.d_vega, b.h_vega);
}

SimResult run_gpu(const SimConfig& cfg) {
  const WorkShape shape = compute_work_shape(cfg);
  if (shape.sample_count == 0) return {};

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
    if (this_chunk > 0) allocate_stream_buffers(buffers[s]);
  }

  const auto t0 = std::chrono::high_resolution_clock::now();

  for (int s = 0; s < kStreams; ++s) {
    auto& b = buffers[s];
    if (b.chunk_samples == 0) continue;
    const std::uint64_t global_offset = static_cast<std::uint64_t>(s) * chunk_samples;
    const dim3 grid(static_cast<unsigned int>(b.blocks));
    const dim3 block(kThreadsPerBlock);
    init_states<<<grid, block, 0, streams[s]>>>(b.d_state_a, b.chunk_samples, cfg.seed, global_offset);

    const std::size_t shared_bytes = 7 * kThreadsPerBlock * sizeof(float);
    simulate_kernel<<<grid, block, shared_bytes, streams[s]>>>(
        b.d_state_a, b.d_state_b, b.d_y, b.d_y2, b.d_x, b.d_x2, b.d_yx, b.d_delta, b.d_vega,
        b.chunk_samples, cfg.steps, cfg.s0, cfg.strike, cfg.rate, cfg.volatility, cfg.maturity,
        cfg.barrier, cfg.antithetic ? 1 : 0, static_cast<int>(cfg.payoff));

    CUDA_CHECK(cudaMemcpyAsync(b.h_y, b.d_y, b.blocks * sizeof(float), cudaMemcpyDeviceToHost, streams[s]));
    CUDA_CHECK(cudaMemcpyAsync(b.h_y2, b.d_y2, b.blocks * sizeof(float), cudaMemcpyDeviceToHost, streams[s]));
    CUDA_CHECK(cudaMemcpyAsync(b.h_x, b.d_x, b.blocks * sizeof(float), cudaMemcpyDeviceToHost, streams[s]));
    CUDA_CHECK(cudaMemcpyAsync(b.h_x2, b.d_x2, b.blocks * sizeof(float), cudaMemcpyDeviceToHost, streams[s]));
    CUDA_CHECK(cudaMemcpyAsync(b.h_yx, b.d_yx, b.blocks * sizeof(float), cudaMemcpyDeviceToHost, streams[s]));
    CUDA_CHECK(cudaMemcpyAsync(b.h_delta, b.d_delta, b.blocks * sizeof(float), cudaMemcpyDeviceToHost, streams[s]));
    CUDA_CHECK(cudaMemcpyAsync(b.h_vega, b.d_vega, b.blocks * sizeof(float), cudaMemcpyDeviceToHost, streams[s]));
  }

  double sum_y = 0.0, sum_y2 = 0.0, sum_x = 0.0, sum_x2 = 0.0, sum_yx = 0.0;
  double sum_delta = 0.0, sum_vega = 0.0;

  for (int s = 0; s < kStreams; ++s) {
    auto& b = buffers[s];
    if (b.chunk_samples == 0) continue;
    CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    for (std::uint64_t i = 0; i < b.blocks; ++i) {
      sum_y += b.h_y[i];
      sum_y2 += b.h_y2[i];
      sum_x += b.h_x[i];
      sum_x2 += b.h_x2[i];
      sum_yx += b.h_yx[i];
      sum_delta += b.h_delta[i];
      sum_vega += b.h_vega[i];
    }
  }

  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  for (int s = 0; s < kStreams; ++s) {
    free_stream_buffers(buffers[s]);
    CUDA_CHECK(cudaStreamDestroy(streams[s]));
  }

  const double n = static_cast<double>(shape.sample_count);
  const double mean_y = sum_y / n;
  const double var_y = std::max(sum_y2 / n - mean_y * mean_y, 0.0);
  const double stddev_y = std::sqrt(var_y);
  const double stderr_y = stddev_y / std::sqrt(n);

  const double mean_x = sum_x / n;
  const double var_x = std::max(sum_x2 / n - mean_x * mean_x, 0.0);
  const double cov_yx = (sum_yx / n) - mean_y * mean_x;
  const double beta = (var_x > 1e-18) ? (cov_yx / var_x) : 0.0;
  const double cv_target = cfg.s0;
  const double mean_y_cv = mean_y - beta * (mean_x - cv_target);
  const double var_y_cv = std::max(var_y + beta * beta * var_x - 2.0 * beta * cov_yx, 0.0);
  const double stderr_y_cv = std::sqrt(var_y_cv) / std::sqrt(n);

  SimResult out;
  out.option_price = mean_y;
  out.option_price_cv = cfg.control_variate ? mean_y_cv : mean_y;
  out.cv_beta = beta;
  out.payoff_mean = mean_y;
  out.payoff_stddev = stddev_y;
  out.payoff_stderr = stderr_y;
  out.ci95_low = mean_y - kZ95 * stderr_y;
  out.ci95_high = mean_y + kZ95 * stderr_y;
  out.ci95_low_cv = (cfg.control_variate ? mean_y_cv : mean_y) - kZ95 * (cfg.control_variate ? stderr_y_cv : stderr_y);
  out.ci95_high_cv = (cfg.control_variate ? mean_y_cv : mean_y) + kZ95 * (cfg.control_variate ? stderr_y_cv : stderr_y);
  out.sample_count = shape.sample_count;
  out.trajectory_count = shape.trajectories;
  out.runtime_ms = ms;

  if (cfg.payoff == PayoffType::EuropeanCall) {
    out.bs_price = black_scholes_call(cfg.s0, cfg.strike, cfg.rate, cfg.volatility, cfg.maturity);
    out.abs_error_bs = std::abs((cfg.control_variate ? out.option_price_cv : out.option_price) - out.bs_price);
    out.delta = sum_delta / n;
    out.vega = sum_vega / n;
    out.bs_delta = black_scholes_delta(cfg.s0, cfg.strike, cfg.rate, cfg.volatility, cfg.maturity);
    out.bs_vega = black_scholes_vega(cfg.s0, cfg.strike, cfg.rate, cfg.volatility, cfg.maturity);
    out.abs_error_delta_bs = std::abs(out.delta - out.bs_delta);
    out.abs_error_vega_bs = std::abs(out.vega - out.bs_vega);
  } else {
    out.bs_price = std::numeric_limits<double>::quiet_NaN();
    out.abs_error_bs = std::numeric_limits<double>::quiet_NaN();
    out.delta = std::numeric_limits<double>::quiet_NaN();
    out.vega = std::numeric_limits<double>::quiet_NaN();
    out.bs_delta = std::numeric_limits<double>::quiet_NaN();
    out.bs_vega = std::numeric_limits<double>::quiet_NaN();
    out.abs_error_delta_bs = std::numeric_limits<double>::quiet_NaN();
    out.abs_error_vega_bs = std::numeric_limits<double>::quiet_NaN();
  }
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
  std::cout << "control_variate=" << (cfg.control_variate ? 1 : 0) << "\n";
  std::cout << "payoff=" << static_cast<int>(cfg.payoff) << "\n";
  std::cout << "steps=" << cfg.steps << "\n";
  std::cout << "price=" << result.option_price << "\n";
  std::cout << "price_cv=" << result.option_price_cv << "\n";
  std::cout << "cv_beta=" << result.cv_beta << "\n";
  std::cout << "ci95_low=" << result.ci95_low << "\n";
  std::cout << "ci95_high=" << result.ci95_high << "\n";
  std::cout << "ci95_low_cv=" << result.ci95_low_cv << "\n";
  std::cout << "ci95_high_cv=" << result.ci95_high_cv << "\n";
  std::cout << "bs_price=" << result.bs_price << "\n";
  std::cout << "abs_error_bs=" << result.abs_error_bs << "\n";
  std::cout << "delta=" << result.delta << "\n";
  std::cout << "vega=" << result.vega << "\n";
  std::cout << "bs_delta=" << result.bs_delta << "\n";
  std::cout << "bs_vega=" << result.bs_vega << "\n";
  std::cout << "abs_error_delta_bs=" << result.abs_error_delta_bs << "\n";
  std::cout << "abs_error_vega_bs=" << result.abs_error_vega_bs << "\n";
  std::cout << "payoff_mean=" << result.payoff_mean << "\n";
  std::cout << "payoff_stddev=" << result.payoff_stddev << "\n";
  std::cout << "payoff_stderr=" << result.payoff_stderr << "\n";
  std::cout << "runtime_ms=" << result.runtime_ms << "\n";
  return 0;
}
