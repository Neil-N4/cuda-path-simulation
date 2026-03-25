#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

#include "argparse.hpp"

SimResult run_cpu(const SimConfig& cfg) {
  const float dt = cfg.maturity / static_cast<float>(cfg.steps);
  const float drift = (cfg.rate - 0.5f * cfg.volatility * cfg.volatility) * dt;
  const float vol_step = cfg.volatility * std::sqrt(dt);

  std::mt19937_64 rng(cfg.seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);

  double sum = 0.0;
  double sum_sq = 0.0;

  const auto t0 = std::chrono::high_resolution_clock::now();

  for (std::uint64_t p = 0; p < cfg.paths; ++p) {
    float st = cfg.s0;
    for (int step = 0; step < cfg.steps; ++step) {
      const float z = normal(rng);
      st = st * std::exp(drift + vol_step * z);
    }
    const float payoff = std::max(st - cfg.strike, 0.0f);
    sum += payoff;
    sum_sq += static_cast<double>(payoff) * static_cast<double>(payoff);
  }

  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  const double n = static_cast<double>(cfg.paths);
  const double mean = sum / n;
  const double variance = std::max(sum_sq / n - mean * mean, 0.0);
  const double discount = std::exp(-cfg.rate * cfg.maturity);

  SimResult out;
  out.payoff_mean = mean;
  out.payoff_stddev = std::sqrt(variance);
  out.option_price = discount * mean;
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
  std::cout << "steps=" << cfg.steps << "\n";
  std::cout << "price=" << result.option_price << "\n";
  std::cout << "payoff_mean=" << result.payoff_mean << "\n";
  std::cout << "payoff_stddev=" << result.payoff_stddev << "\n";
  std::cout << "runtime_ms=" << result.runtime_ms << "\n";

  return 0;
}
