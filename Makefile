BUILD_DIR ?= build

.PHONY: configure build run-cpu run-gpu benchmark plot validate clean

configure:
	cmake -S . -B $(BUILD_DIR)

build: configure
	cmake --build $(BUILD_DIR) -j

run-cpu: build
	./$(BUILD_DIR)/mc_cpu --paths 1000000 --steps 365

run-gpu: build
	./$(BUILD_DIR)/mc_gpu --paths 10000000 --steps 365

benchmark: build
	python3 scripts/benchmark.py --build-dir $(BUILD_DIR) --paths 10000000 --steps 365 --runs 3

plot:
	python3 scripts/plot_benchmark.py --csv results/benchmark_results.csv --out results/runtime_comparison.png

validate: build
	python3 scripts/validate_parity.py --build-dir $(BUILD_DIR) --paths 2000000 --steps 365 --price-tol 0.05

clean:
	rm -rf $(BUILD_DIR) results/*.csv results/*.png results/*.qdrep results/*.nsys-rep
