/**
 * BCI Edge Engine: Zero-overhead real-time sliding-window neural state inference.
 * C++23 implementation targeting ultra-low latency (<5ms) edge BCI deployment.
 */

#include <iostream>
#include <vector>
#include <array>
#include <span>
#include <ranges>
#include <chrono>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>

constexpr size_t NUM_REGIONS = 68;       // Desikan-Killiany parcellation size
constexpr size_t STATE_DIM = 4;          // Excitatory, Inhibitory, Phase, Derivative
constexpr size_t WINDOW_SIZE = 128;      // Sliding window sample depth
constexpr size_t BENCHMARK_ITERS = 1000; // Benchmark iterations

/**
 * Ring-buffer for sliding-window multi-channel neural stream.
 */
template <typename T, size_t Channels, size_t Capacity>
class SlidingWindowBuffer {
public:
    SlidingWindowBuffer() : head_(0), size_(0) {
        buffer_.fill(T{0});
    }

    void push_sample(std::span<const T, Channels> sample) {
        size_t offset = head_ * Channels;
        for (size_t c = 0; c < Channels; ++c) {
            buffer_[offset + c] = sample[c];
        }
        head_ = (head_ + 1) % Capacity;
        if (size_ < Capacity) {
            ++size_;
        }
    }

    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] bool is_full() const noexcept { return size_ == Capacity; }

    // Extracts flattened view of current window
    void get_latest_window(std::span<T> out) const {
        size_t start = (head_ + Capacity - size_) % Capacity;
        for (size_t i = 0; i < size_; ++i) {
            size_t src_idx = ((start + i) % Capacity) * Channels;
            size_t dst_idx = i * Channels;
            for (size_t c = 0; c < Channels; ++c) {
                out[dst_idx + c] = buffer_[src_idx + c];
            }
        }
    }

private:
    std::array<T, Channels * Capacity> buffer_;
    size_t head_;
    size_t size_;
};

/**
 * Neural Connectome continuous-time vector field evaluation.
 */
class ConnectomeDynamicsEngine {
public:
    ConnectomeDynamicsEngine(float alpha = 0.15f, float kappa = 0.85f)
        : alpha_(alpha), kappa_(kappa) {
        initialize_synthetic_laplacian();
    }

    void initialize_synthetic_laplacian() {
        laplacian_.fill(0.0f);
        // Fill diagonally dominant symmetric normalized Laplacian
        for (size_t i = 0; i < NUM_REGIONS; ++i) {
            laplacian_[i * NUM_REGIONS + i] = 1.0f;
            // Nearest neighbors and modular connections
            size_t left = (i + NUM_REGIONS - 1) % NUM_REGIONS;
            size_t right = (i + 1) % NUM_REGIONS;
            laplacian_[i * NUM_REGIONS + left] = -0.15f;
            laplacian_[i * NUM_REGIONS + right] = -0.15f;
        }
    }

    // Evaluates dX/dt = -alpha * X - kappa * L_sym * X + tanh(W * X)
    void compute_derivative(
        std::span<const float> state, // (NUM_REGIONS * STATE_DIM)
        std::span<float> dxdt
    ) const noexcept {
        // 1. Damping: -alpha * X
        for (size_t idx = 0; idx < NUM_REGIONS * STATE_DIM; ++idx) {
            dxdt[idx] = -alpha_ * state[idx];
        }

        // 2. Inter-regional diffusion: -kappa * (L_sym @ X)
        for (size_t i = 0; i < NUM_REGIONS; ++i) {
            for (size_t d = 0; d < STATE_DIM; ++d) {
                float lap_sum = 0.0f;
                for (size_t j = 0; j < NUM_REGIONS; ++j) {
                    float l_ij = laplacian_[i * NUM_REGIONS + j];
                    if (l_ij != 0.0f) {
                        lap_sum += l_ij * state[j * STATE_DIM + d];
                    }
                }
                dxdt[i * STATE_DIM + d] -= kappa_ * lap_sum;

                // 3. Intrinsic non-linear saturation
                dxdt[i * STATE_DIM + d] += 0.1f * std::tanh(state[i * STATE_DIM + d]);
            }
        }
    }

    // Single Runge-Kutta 4th order continuous integration step
    void rk4_step(
        std::span<float> state,
        float dt
    ) {
        constexpr size_t TOTAL = NUM_REGIONS * STATE_DIM;
        std::array<float, TOTAL> k1, k2, k3, k4, temp;

        // k1 = f(t, y)
        compute_derivative(state, k1);

        // k2 = f(t + dt/2, y + dt/2 * k1)
        for (size_t i = 0; i < TOTAL; ++i) temp[i] = state[i] + 0.5f * dt * k1[i];
        compute_derivative(temp, k2);

        // k3 = f(t + dt/2, y + dt/2 * k2)
        for (size_t i = 0; i < TOTAL; ++i) temp[i] = state[i] + 0.5f * dt * k2[i];
        compute_derivative(temp, k3);

        // k4 = f(t + dt, y + dt * k3)
        for (size_t i = 0; i < TOTAL; ++i) temp[i] = state[i] + dt * k3[i];
        compute_derivative(temp, k4);

        // Update state
        for (size_t i = 0; i < TOTAL; ++i) {
            state[i] += (dt / 6.0f) * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
        }
    }

private:
    float alpha_;
    float kappa_;
    std::array<float, NUM_REGIONS * NUM_REGIONS> laplacian_;
};

int main(int argc, char* argv[]) {
    std::cout << "========================================================\n";
    std::cout << "   Neural-Flow BCI Edge Engine (C++23 Real-Time Core)   \n";
    std::cout << "   Target Latency: < 5.0 ms per sliding window cycle    \n";
    std::cout << "========================================================\n\n";

    SlidingWindowBuffer<float, NUM_REGIONS, WINDOW_SIZE> ring_buffer;
    ConnectomeDynamicsEngine engine(0.15f, 0.85f);

    // Initial state vector
    std::array<float, NUM_REGIONS * STATE_DIM> state;
    for (size_t i = 0; i < state.size(); ++i) {
        state[i] = static_cast<float>(std::sin(i * 0.1));
    }

    // Warm-up cycles
    for (size_t i = 0; i < 50; ++i) {
        engine.rk4_step(state, 0.02f);
    }

    std::vector<double> latencies_us;
    latencies_us.reserve(BENCHMARK_ITERS);

    std::array<float, NUM_REGIONS> incoming_sample;
    incoming_sample.fill(0.5f);

    // Main real-time streaming benchmark loop
    for (size_t iter = 0; iter < BENCHMARK_ITERS; ++iter) {
        auto start = std::chrono::high_resolution_clock::now();

        // 1. Ingest new multi-channel EEG / cortical sensor sample into ring buffer
        ring_buffer.push_sample(incoming_sample);

        // 2. Perform continuous-time connectome integration step
        engine.rk4_step(state, 0.02f);

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies_us.push_back(elapsed_us);
    }

    // Compute latency statistics
    std::ranges::sort(latencies_us);
    double sum = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
    double mean_us = sum / static_cast<double>(latencies_us.size());
    double median_us = latencies_us[latencies_us.size() / 2];
    double p95_us = latencies_us[static_cast<size_t>(latencies_us.size() * 0.95)];
    double p99_us = latencies_us[static_cast<size_t>(latencies_us.size() * 0.99)];
    double min_us = latencies_us.front();
    double max_us = latencies_us.back();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Benchmark Results (" << BENCHMARK_ITERS << " sliding window iterations):\n";
    std::cout << "  Mean Latency:    " << (mean_us / 1000.0) << " ms (" << mean_us << " us)\n";
    std::cout << "  Median Latency:  " << (median_us / 1000.0) << " ms (" << median_us << " us)\n";
    std::cout << "  95th Percentile: " << (p95_us / 1000.0) << " ms (" << p95_us << " us)\n";
    std::cout << "  99th Percentile: " << (p99_us / 1000.0) << " ms (" << p99_us << " us)\n";
    std::cout << "  Min Latency:     " << (min_us / 1000.0) << " ms (" << min_us << " us)\n";
    std::cout << "  Max Latency:     " << (max_us / 1000.0) << " ms (" << max_us << " us)\n";

    bool meets_target = (p95_us / 1000.0) < 5.0;
    std::cout << "\nTarget Check (<5.0 ms): " << (meets_target ? "PASS [OK]" : "FAIL") << "\n";

    // Write benchmark JSON
    std::ofstream out("results/cpp_bci_benchmark.json");
    if (out.is_open()) {
        out << "{\n";
        out << "  \"engine\": \"C++23 BCI Edge Engine\",\n";
        out << "  \"num_regions\": " << NUM_REGIONS << ",\n";
        out << "  \"state_dim\": " << STATE_DIM << ",\n";
        out << "  \"iterations\": " << BENCHMARK_ITERS << ",\n";
        out << "  \"mean_latency_ms\": " << (mean_us / 1000.0) << ",\n";
        out << "  \"median_latency_ms\": " << (median_us / 1000.0) << ",\n";
        out << "  \"p95_latency_ms\": " << (p95_us / 1000.0) << ",\n";
        out << "  \"p99_latency_ms\": " << (p99_us / 1000.0) << ",\n";
        out << "  \"target_met\": " << (meets_target ? "true" : "false") << "\n";
        out << "}\n";
        out.close();
        std::cout << "Metrics exported to results/cpp_bci_benchmark.json\n";
    }

    return 0;
}
