#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>

#include "argparse.hpp"

namespace {

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

std::uint64_t adjusted_paths(const SimConfig& cfg) {
  if (!cfg.antithetic) return cfg.paths;
  if (cfg.paths < 2) return 0;
  return (cfg.paths % 2 == 0) ? cfg.paths : (cfg.paths - 1);
}

}  // namespace

SimResult run_cpu(const SimConfig& cfg) {
  const std::uint64_t trajectories = adjusted_paths(cfg);
  if (trajectories == 0) {
    return {};
  }

  const float dt = cfg.maturity / static_cast<float>(cfg.steps);
  const float drift = (cfg.rate - 0.5f * cfg.volatility * cfg.volatility) * dt;
  const float vol_step = cfg.volatility * std::sqrt(dt);

  std::mt19937_64 rng(cfg.seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);

  double sum = 0.0;
  double sum_sq = 0.0;
  std::uint64_t sample_count = 0;

  const auto t0 = std::chrono::high_resolution_clock::now();

  if (cfg.antithetic) {
    for (std::uint64_t p = 0; p < trajectories; p += 2) {
      float st_a = cfg.s0;
      float st_b = cfg.s0;
      for (int step = 0; step < cfg.steps; ++step) {
        const float z = normal(rng);
        st_a = st_a * std::exp(drift + vol_step * z);
        st_b = st_b * std::exp(drift - vol_step * z);
      }
      const double payoff_a = std::max(st_a - cfg.strike, 0.0f);
      const double payoff_b = std::max(st_b - cfg.strike, 0.0f);
      const double sample_payoff = 0.5 * (payoff_a + payoff_b);
      sum += sample_payoff;
      sum_sq += sample_payoff * sample_payoff;
      ++sample_count;
    }
  } else {
    for (std::uint64_t p = 0; p < trajectories; ++p) {
      float st = cfg.s0;
      for (int step = 0; step < cfg.steps; ++step) {
        const float z = normal(rng);
        st = st * std::exp(drift + vol_step * z);
      }
      const double sample_payoff = std::max(st - cfg.strike, 0.0f);
      sum += sample_payoff;
      sum_sq += sample_payoff * sample_payoff;
      ++sample_count;
    }
  }

  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  const double n = static_cast<double>(sample_count);
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
  out.sample_count = sample_count;
  out.trajectory_count = trajectories;
  out.runtime_ms = ms;
  return out;
}

int main(int argc, char** argv) {
  SimConfig cfg;
  if (!parse_args(argc, argv, cfg)) return 1;

  const SimResult result = run_cpu(cfg);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "engine=cpu\n";
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
