#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

#include "sim_config.hpp"

inline bool parse_payoff(const std::string& s, PayoffType& payoff) {
  if (s == "european") {
    payoff = PayoffType::EuropeanCall;
    return true;
  }
  if (s == "asian") {
    payoff = PayoffType::AsianCall;
    return true;
  }
  if (s == "upout") {
    payoff = PayoffType::UpAndOutCall;
    return true;
  }
  return false;
}

inline bool parse_rng_mode(const std::string& s, RngMode& mode) {
  if (s == "philox") {
    mode = RngMode::Philox;
    return true;
  }
  if (s == "sobol") {
    mode = RngMode::Sobol;
    return true;
  }
  return false;
}

inline bool parse_math_mode(const std::string& s, MathMode& mode) {
  if (s == "fp32") {
    mode = MathMode::FP32;
    return true;
  }
  if (s == "mixed") {
    mode = MathMode::Mixed;
    return true;
  }
  return false;
}

inline void print_usage(const char* bin) {
  std::cerr
      << "Usage: " << bin
      << " [--paths N] [--steps N] [--s0 X] [--strike X] [--rate X]"
      << " [--vol X] [--maturity X] [--barrier X] [--seed N]"
      << " [--payoff european|asian|upout]"
      << " [--rng philox|sobol]"
      << " [--math fp32|mixed]"
      << " [--antithetic|--no-antithetic]"
      << " [--control-variate|--no-control-variate]" << std::endl;
}

inline bool parse_args(int argc, char** argv, SimConfig& cfg) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--antithetic") {
      cfg.antithetic = true;
      continue;
    }
    if (arg == "--no-antithetic") {
      cfg.antithetic = false;
      continue;
    }
    if (arg == "--control-variate") {
      cfg.control_variate = true;
      continue;
    }
    if (arg == "--no-control-variate") {
      cfg.control_variate = false;
      continue;
    }
    if (i + 1 >= argc) {
      print_usage(argv[0]);
      return false;
    }
    const std::string value = argv[++i];
    try {
      if (arg == "--paths") cfg.paths = std::stoull(value);
      else if (arg == "--steps") cfg.steps = std::stoi(value);
      else if (arg == "--s0") cfg.s0 = std::stof(value);
      else if (arg == "--strike") cfg.strike = std::stof(value);
      else if (arg == "--rate") cfg.rate = std::stof(value);
      else if (arg == "--vol") cfg.volatility = std::stof(value);
      else if (arg == "--maturity") cfg.maturity = std::stof(value);
      else if (arg == "--barrier") cfg.barrier = std::stof(value);
      else if (arg == "--seed") cfg.seed = std::stoull(value);
      else if (arg == "--payoff") {
        if (!parse_payoff(value, cfg.payoff)) {
          print_usage(argv[0]);
          return false;
        }
      } else if (arg == "--rng") {
        if (!parse_rng_mode(value, cfg.rng_mode)) {
          print_usage(argv[0]);
          return false;
        }
      } else if (arg == "--math") {
        if (!parse_math_mode(value, cfg.math_mode)) {
          print_usage(argv[0]);
          return false;
        }
      }
      else {
        print_usage(argv[0]);
        return false;
      }
    } catch (...) {
      print_usage(argv[0]);
      return false;
    }
  }
  return true;
}
