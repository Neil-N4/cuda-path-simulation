#pragma once

#include <cstdint>

struct SimConfig {
  std::uint64_t paths = 10000000;
  int steps = 365;
  float s0 = 100.0f;
  float strike = 100.0f;
  float rate = 0.03f;
  float volatility = 0.2f;
  float maturity = 1.0f;
  std::uint64_t seed = 42;
};

struct SimResult {
  double option_price = 0.0;
  double payoff_mean = 0.0;
  double payoff_stddev = 0.0;
  double runtime_ms = 0.0;
};
