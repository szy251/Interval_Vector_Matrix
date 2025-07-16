#include <benchmark/benchmark.h>
#include "BatchSwitchMatrixAVX_Grouped.hpp"
#include "BatchSwitchMatrixAVX512_Grouped.hpp"
#include "Utilities.hpp"


static constexpr size_t N_25 = 25;
static constexpr size_t N_50 = 50;
static constexpr size_t N_75 = 75;
static constexpr size_t N_100 = 100;
static constexpr size_t N_125 = 125;
static constexpr size_t N_150 = 150;
static constexpr size_t N_175 = 175;
static constexpr size_t N_200 = 200;


static void BM_ADD_25_AVX(benchmark::State& state) {
    const size_t num_intervals = N_25 * N_25;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX_Grouped<N_25, N_25> matrix1(intervals1_data);
    BatchSwitchMatrixAVX_Grouped<N_25, N_25> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 + matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ADD_25_AVX);

static void BM_ADD_50_AVX(benchmark::State& state) {
    const size_t num_intervals = N_50 * N_50;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX_Grouped<N_50, N_50> matrix1(intervals1_data);
    BatchSwitchMatrixAVX_Grouped<N_50, N_50> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 + matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ADD_50_AVX);

static void BM_ADD_75_AVX(benchmark::State& state) {
    const size_t num_intervals = N_75 * N_75;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX_Grouped<N_75, N_75> matrix1(intervals1_data);
    BatchSwitchMatrixAVX_Grouped<N_75, N_75> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 + matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ADD_75_AVX);

static void BM_ADD_100_AVX(benchmark::State& state) {
    const size_t num_intervals = N_100 * N_100;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX_Grouped<N_100, N_100> matrix1(intervals1_data);
    BatchSwitchMatrixAVX_Grouped<N_100, N_100> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 + matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ADD_100_AVX);

static void BM_ADD_125_AVX(benchmark::State& state) {
    const size_t num_intervals = N_125 * N_125;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX_Grouped<N_125, N_125> matrix1(intervals1_data);
    BatchSwitchMatrixAVX_Grouped<N_125, N_125> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 + matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ADD_125_AVX);

static void BM_ADD_150_AVX(benchmark::State& state) {
    const size_t num_intervals = N_150 * N_150;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX_Grouped<N_150, N_150> matrix1(intervals1_data);
    BatchSwitchMatrixAVX_Grouped<N_150, N_150> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 + matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ADD_150_AVX);

static void BM_ADD_175_AVX(benchmark::State& state) {
    const size_t num_intervals = N_175 * N_175;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX_Grouped<N_175, N_175> matrix1(intervals1_data);
    BatchSwitchMatrixAVX_Grouped<N_175, N_175> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 + matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ADD_175_AVX);

static void BM_ADD_200_AVX(benchmark::State& state) {
    const size_t num_intervals = N_200 * N_200;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX_Grouped<N_200, N_200> matrix1(intervals1_data);
    BatchSwitchMatrixAVX_Grouped<N_200, N_200> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 + matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ADD_200_AVX);

// --- Benchmarki dla BatchSwitchMatrixAVX512_Grouped - Dodawanie ---

static void BM_ADD_25_AVX512(benchmark::State& state) {
    const size_t num_intervals = N_25 * N_25;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX512_Grouped<N_25, N_25> matrix1(intervals1_data);
    BatchSwitchMatrixAVX512_Grouped<N_25, N_25> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 + matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ADD_25_AVX512);

static void BM_ADD_50_AVX512(benchmark::State& state) {
    const size_t num_intervals = N_50 * N_50;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX512_Grouped<N_50, N_50> matrix1(intervals1_data);
    BatchSwitchMatrixAVX512_Grouped<N_50, N_50> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 + matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ADD_50_AVX512);

static void BM_ADD_75_AVX512(benchmark::State& state) {
    const size_t num_intervals = N_75 * N_75;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX512_Grouped<N_75, N_75> matrix1(intervals1_data);
    BatchSwitchMatrixAVX512_Grouped<N_75, N_75> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 + matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ADD_75_AVX512);

static void BM_ADD_100_AVX512(benchmark::State& state) {
    const size_t num_intervals = N_100 * N_100;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX512_Grouped<N_100, N_100> matrix1(intervals1_data);
    BatchSwitchMatrixAVX512_Grouped<N_100, N_100> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 + matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ADD_100_AVX512);

static void BM_ADD_125_AVX512(benchmark::State& state) {
    const size_t num_intervals = N_125 * N_125;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX512_Grouped<N_125, N_125> matrix1(intervals1_data);
    BatchSwitchMatrixAVX512_Grouped<N_125, N_125> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 + matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ADD_125_AVX512);

static void BM_ADD_150_AVX512(benchmark::State& state) {
    const size_t num_intervals = N_150 * N_150;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX512_Grouped<N_150, N_150> matrix1(intervals1_data);
    BatchSwitchMatrixAVX512_Grouped<N_150, N_150> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 + matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ADD_150_AVX512);

static void BM_ADD_175_AVX512(benchmark::State& state) {
    const size_t num_intervals = N_175 * N_175;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX512_Grouped<N_175, N_175> matrix1(intervals1_data);
    BatchSwitchMatrixAVX512_Grouped<N_175, N_175> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 + matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ADD_175_AVX512);

static void BM_ADD_200_AVX512(benchmark::State& state) {
    const size_t num_intervals = N_200 * N_200;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX512_Grouped<N_200, N_200> matrix1(intervals1_data);
    BatchSwitchMatrixAVX512_Grouped<N_200, N_200> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 + matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ADD_200_AVX512);

BENCHMARK_MAIN();