#pragma once

#include <cstdint>

enum class PayoffType : int {
  EuropeanCall = 0,
  AsianCall = 1,
  UpAndOutCall = 2,
};

enum class RngMode : int {
  Philox = 0,
  Sobol = 1,
};

enum class MathMode : int {
  FP32 = 0,
  Mixed = 1,
};

struct SimConfig {
  std::uint64_t paths = 10000000;
  int steps = 365;
  float s0 = 100.0f;
  float strike = 100.0f;
  float rate = 0.03f;
  float volatility = 0.2f;
  float maturity = 1.0f;
  float barrier = 130.0f;
  std::uint64_t seed = 42;
  bool antithetic = false;
  bool control_variate = false;
  PayoffType payoff = PayoffType::EuropeanCall;
  RngMode rng_mode = RngMode::Philox;
  MathMode math_mode = MathMode::FP32;
};

struct SimResult {
  double option_price = 0.0;
  double option_price_cv = 0.0;
  double cv_beta = 0.0;
  double payoff_mean = 0.0;
  double payoff_stddev = 0.0;
  double payoff_stderr = 0.0;
  double ci95_low = 0.0;
  double ci95_high = 0.0;
  double ci95_low_cv = 0.0;
  double ci95_high_cv = 0.0;
  double bs_price = 0.0;
  double abs_error_bs = 0.0;
  double delta = 0.0;
  double vega = 0.0;
  double bs_delta = 0.0;
  double bs_vega = 0.0;
  double abs_error_delta_bs = 0.0;
  double abs_error_vega_bs = 0.0;
  std::uint64_t sample_count = 0;
  std::uint64_t trajectory_count = 0;
  double runtime_ms = 0.0;
};
