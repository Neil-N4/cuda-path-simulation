#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>

#include "argparse.hpp"
#include "low_discrepancy.hpp"

namespace {

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

std::uint64_t adjusted_paths(const SimConfig& cfg) {
  if (!cfg.antithetic) return cfg.paths;
  if (cfg.paths < 2) return 0;
  return (cfg.paths % 2 == 0) ? cfg.paths : (cfg.paths - 1);
}

double payoff_value(PayoffType payoff, double terminal, double mean_path, bool knocked_out, double strike) {
  if (payoff == PayoffType::EuropeanCall) {
    return std::max(terminal - strike, 0.0);
  }
  if (payoff == PayoffType::AsianCall) {
    return std::max(mean_path - strike, 0.0);
  }
  if (payoff == PayoffType::UpAndOutCall) {
    if (knocked_out) return 0.0;
    return std::max(terminal - strike, 0.0);
  }
  return 0.0;
}

}  // namespace

SimResult run_cpu(const SimConfig& cfg) {
  const std::uint64_t trajectories = adjusted_paths(cfg);
  if (trajectories == 0) return {};

  const float dt = cfg.maturity / static_cast<float>(cfg.steps);
  const float sqrt_dt = std::sqrt(dt);
  const float drift = (cfg.rate - 0.5f * cfg.volatility * cfg.volatility) * dt;
  const float vol_step = cfg.volatility * sqrt_dt;
  const double discount = std::exp(-cfg.rate * cfg.maturity);
  const bool greek_enabled = (cfg.payoff == PayoffType::EuropeanCall);

  std::mt19937_64 rng(cfg.seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);

  double sum_y = 0.0, sum_y2 = 0.0;
  double sum_x = 0.0, sum_x2 = 0.0, sum_yx = 0.0;
  double sum_delta = 0.0, sum_vega = 0.0;
  std::uint64_t sample_count = 0;

  const auto t0 = std::chrono::high_resolution_clock::now();

  if (cfg.antithetic) {
    for (std::uint64_t p = 0; p < trajectories; p += 2) {
      float st_a = cfg.s0, st_b = cfg.s0;
      float mean_acc_a = 0.0f, mean_acc_b = 0.0f;
      bool knock_a = false, knock_b = false;
      float w_a = 0.0f;

      for (int step = 0; step < cfg.steps; ++step) {
        const float z = (cfg.rng_mode == RngMode::Sobol)
                            ? lowdisc::sobol_normal(sample_count, step, cfg.seed)
                            : normal(rng);
        w_a += z * sqrt_dt;
        st_a = st_a * std::exp(drift + vol_step * z);
        st_b = st_b * std::exp(drift - vol_step * z);
        mean_acc_a += st_a;
        mean_acc_b += st_b;
        knock_a = knock_a || (st_a >= cfg.barrier);
        knock_b = knock_b || (st_b >= cfg.barrier);
      }

      const double mean_path_a = static_cast<double>(mean_acc_a) / cfg.steps;
      const double mean_path_b = static_cast<double>(mean_acc_b) / cfg.steps;
      const double payoff_a = payoff_value(cfg.payoff, st_a, mean_path_a, knock_a, cfg.strike);
      const double payoff_b = payoff_value(cfg.payoff, st_b, mean_path_b, knock_b, cfg.strike);

      const double y = 0.5 * discount * (payoff_a + payoff_b);
      const double x = 0.5 * discount * (static_cast<double>(st_a) + static_cast<double>(st_b));
      sum_y += y;
      sum_y2 += y * y;
      sum_x += x;
      sum_x2 += x * x;
      sum_yx += y * x;

      if (greek_enabled) {
        const double ind_a = (st_a > cfg.strike) ? 1.0 : 0.0;
        const double ind_b = (st_b > cfg.strike) ? 1.0 : 0.0;
        const double delta_a = discount * ind_a * st_a / cfg.s0;
        const double delta_b = discount * ind_b * st_b / cfg.s0;
        const double vega_a = discount * ind_a * st_a * (w_a - cfg.volatility * cfg.maturity);
        const double vega_b = discount * ind_b * st_b * (-w_a - cfg.volatility * cfg.maturity);
        sum_delta += 0.5 * (delta_a + delta_b);
        sum_vega += 0.5 * (vega_a + vega_b);
      }
      ++sample_count;
    }
  } else {
    for (std::uint64_t p = 0; p < trajectories; ++p) {
      float st = cfg.s0;
      float mean_acc = 0.0f;
      bool knocked = false;
      float w = 0.0f;

      for (int step = 0; step < cfg.steps; ++step) {
        const float z = (cfg.rng_mode == RngMode::Sobol)
                            ? lowdisc::sobol_normal(sample_count, step, cfg.seed)
                            : normal(rng);
        w += z * sqrt_dt;
        st = st * std::exp(drift + vol_step * z);
        mean_acc += st;
        knocked = knocked || (st >= cfg.barrier);
      }

      const double mean_path = static_cast<double>(mean_acc) / cfg.steps;
      const double payoff = payoff_value(cfg.payoff, st, mean_path, knocked, cfg.strike);
      const double y = discount * payoff;
      const double x = discount * st;
      sum_y += y;
      sum_y2 += y * y;
      sum_x += x;
      sum_x2 += x * x;
      sum_yx += y * x;

      if (greek_enabled) {
        const double ind = (st > cfg.strike) ? 1.0 : 0.0;
        sum_delta += discount * ind * st / cfg.s0;
        sum_vega += discount * ind * st * (w - cfg.volatility * cfg.maturity);
      }
      ++sample_count;
    }
  }

  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  const double n = static_cast<double>(sample_count);
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
  out.sample_count = sample_count;
  out.trajectory_count = trajectories;
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

int main(int argc, char** argv) {
  SimConfig cfg;
  if (!parse_args(argc, argv, cfg)) return 1;
  if (cfg.rng_mode == RngMode::Sobol && cfg.math_mode == MathMode::Mixed) {
    std::cerr << "unsupported mode: --rng sobol cannot be combined with --math mixed"
              << std::endl;
    return 2;
  }
  const SimResult result = run_cpu(cfg);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "engine=cpu\n";
  std::cout << "paths=" << cfg.paths << "\n";
  std::cout << "trajectories=" << result.trajectory_count << "\n";
  std::cout << "samples=" << result.sample_count << "\n";
  std::cout << "antithetic=" << (cfg.antithetic ? 1 : 0) << "\n";
  std::cout << "control_variate=" << (cfg.control_variate ? 1 : 0) << "\n";
  std::cout << "rng_mode=" << (cfg.rng_mode == RngMode::Sobol ? "sobol" : "philox") << "\n";
  std::cout << "math_mode=" << (cfg.math_mode == MathMode::Mixed ? "mixed" : "fp32") << "\n";
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
