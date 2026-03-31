BUILD_DIR ?= build

.PHONY: configure build build-cpu run-cpu run-gpu run-gpu-cv benchmark benchmark-anti benchmark-cv benchmark-anti-cv mode-sweep plot validate validate-anti validate-cv convergence-gpu convergence-cpu stress perf-gate nsight-compute nsight-csv clean

configure:
	cmake -S . -B $(BUILD_DIR)

build: configure
	cmake --build $(BUILD_DIR) -j

build-cpu:
	cmake -S . -B $(BUILD_DIR) -DBUILD_CUDA=OFF
	cmake --build $(BUILD_DIR) -j

run-cpu: build
	./$(BUILD_DIR)/mc_cpu --paths 1000000 --steps 365

run-gpu: build
	./$(BUILD_DIR)/mc_gpu --paths 10000000 --steps 365

run-gpu-cv: build
	./$(BUILD_DIR)/mc_gpu --paths 10000000 --steps 365 --antithetic --control-variate

benchmark: build
	python3 scripts/benchmark.py --build-dir $(BUILD_DIR) --paths 10000000 --steps 365 --runs 3

benchmark-anti: build
	python3 scripts/benchmark.py --build-dir $(BUILD_DIR) --paths 10000000 --steps 365 --runs 3 --antithetic

benchmark-cv: build
	python3 scripts/benchmark.py --build-dir $(BUILD_DIR) --paths 10000000 --steps 365 --runs 3 --control-variate

benchmark-anti-cv: build
	python3 scripts/benchmark.py --build-dir $(BUILD_DIR) --paths 10000000 --steps 365 --runs 3 --antithetic --control-variate

mode-sweep: build
	python3 scripts/mode_sweep.py --build-dir $(BUILD_DIR) --paths 2000000 --steps 365 --payoff european --antithetic --control-variate

plot:
	python3 scripts/plot_benchmark.py --csv results/benchmark_results.csv --out results/runtime_comparison.png

validate: build
	python3 scripts/validate_parity.py --build-dir $(BUILD_DIR) --paths 2000000 --steps 365 --price-tol 0.05

validate-anti: build
	python3 scripts/validate_parity.py --build-dir $(BUILD_DIR) --paths 2000000 --steps 365 --price-tol 0.05 --antithetic --require-ci-overlap

validate-cv: build
	python3 scripts/validate_parity.py --build-dir $(BUILD_DIR) --paths 2000000 --steps 365 --price-tol 0.05 --antithetic --control-variate --require-ci-overlap

convergence-gpu: build
	python3 scripts/convergence_report.py --build-dir $(BUILD_DIR) --engine gpu --steps 365

convergence-cpu: build
	python3 scripts/convergence_report.py --build-dir $(BUILD_DIR) --engine cpu --steps 365

stress: build
	python3 scripts/stress_suite.py --build-dir $(BUILD_DIR)

nsight-compute: build
	bash scripts/profile_ncu.sh $(BUILD_DIR) 10000000 365 european

nsight-csv: build
	bash scripts/profile_ncu_csv.sh $(BUILD_DIR) results/ncu_metrics.csv 10000000 365

perf-gate:
	python3 scripts/perf_gate.py --benchmark-csv results/benchmark_results.csv --convergence-csv results/convergence/gpu_convergence.csv --thresholds configs/perf_gate_thresholds.json

clean:
	rm -rf $(BUILD_DIR) results/*.csv results/*.png results/*.qdrep results/*.nsys-rep results/*.ncu-rep results/convergence
