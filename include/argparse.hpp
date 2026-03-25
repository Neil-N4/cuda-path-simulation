#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

#include "sim_config.hpp"

inline void print_usage(const char* bin) {
  std::cerr
      << "Usage: " << bin
      << " [--paths N] [--steps N] [--s0 X] [--strike X] [--rate X]"
      << " [--vol X] [--maturity X] [--seed N]" << std::endl;
}

inline bool parse_args(int argc, char** argv, SimConfig& cfg) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
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
      else if (arg == "--seed") cfg.seed = std::stoull(value);
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
