#include <benchmark/benchmark.h>
#include "BatchSwitchMatrixAVX_Grouped.hpp"
#include "BatchSwitchMatrixAVX512_Grouped.hpp"
#include "Utilities.hpp"


static constexpr size_t M_10 = 10;
static constexpr size_t M_15 = 15;
static constexpr size_t M_20 = 20;
static constexpr size_t M_25 = 25;
static constexpr size_t M_30 = 30;
static constexpr size_t M_35 = 35;
static constexpr size_t M_40 = 40;
static constexpr size_t M_45 = 45;
static constexpr size_t M_50 = 50;




static void BM_MLT_MAT_10_AVX(benchmark::State& state) {
    const size_t num_intervals = M_10 * M_10;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX_Grouped<M_10, M_10> matrix1(intervals1_data);
    BatchSwitchMatrixAVX_Grouped<M_10, M_10> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 * matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MLT_MAT_10_AVX);


static void BM_MLT_MAT_15_AVX(benchmark::State& state) {
    const size_t num_intervals = M_15 * M_15;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX_Grouped<M_15, M_15> matrix1(intervals1_data);
    BatchSwitchMatrixAVX_Grouped<M_15, M_15> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 * matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MLT_MAT_15_AVX);


static void BM_MLT_MAT_20_AVX(benchmark::State& state) {
    const size_t num_intervals = M_20 * M_20;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX_Grouped<M_20, M_20> matrix1(intervals1_data);
    BatchSwitchMatrixAVX_Grouped<M_20, M_20> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 * matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MLT_MAT_20_AVX);


static void BM_MLT_MAT_25_AVX(benchmark::State& state) {
    const size_t num_intervals = M_25 * M_25;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX_Grouped<M_25, M_25> matrix1(intervals1_data);
    BatchSwitchMatrixAVX_Grouped<M_25, M_25> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 * matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MLT_MAT_25_AVX);


static void BM_MLT_MAT_30_AVX(benchmark::State& state) {
    const size_t num_intervals = M_30 * M_30;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX_Grouped<M_30, M_30> matrix1(intervals1_data);
    BatchSwitchMatrixAVX_Grouped<M_30, M_30> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 * matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MLT_MAT_30_AVX);


static void BM_MLT_MAT_35_AVX(benchmark::State& state) {
    const size_t num_intervals = M_35 * M_35;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX_Grouped<M_35, M_35> matrix1(intervals1_data);
    BatchSwitchMatrixAVX_Grouped<M_35, M_35> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 * matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MLT_MAT_35_AVX);


static void BM_MLT_MAT_40_AVX(benchmark::State& state) {
    const size_t num_intervals = M_40 * M_40;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX_Grouped<M_40, M_40> matrix1(intervals1_data);
    BatchSwitchMatrixAVX_Grouped<M_40, M_40> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 * matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MLT_MAT_40_AVX);


static void BM_MLT_MAT_45_AVX(benchmark::State& state) {
    const size_t num_intervals = M_45 * M_45;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX_Grouped<M_45, M_45> matrix1(intervals1_data);
    BatchSwitchMatrixAVX_Grouped<M_45, M_45> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 * matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MLT_MAT_45_AVX);


static void BM_MLT_MAT_50_AVX(benchmark::State& state) {
    const size_t num_intervals = M_50 * M_50;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX_Grouped<M_50, M_50> matrix1(intervals1_data);
    BatchSwitchMatrixAVX_Grouped<M_50, M_50> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 * matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MLT_MAT_50_AVX);




static void BM_MLT_MAT_10_AVX512(benchmark::State& state) {
    const size_t num_intervals = M_10 * M_10;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX512_Grouped<M_10, M_10> matrix1(intervals1_data);
    BatchSwitchMatrixAVX512_Grouped<M_10, M_10> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 * matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MLT_MAT_10_AVX512);


static void BM_MLT_MAT_15_AVX512(benchmark::State& state) {
    const size_t num_intervals = M_15 * M_15;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX512_Grouped<M_15, M_15> matrix1(intervals1_data);
    BatchSwitchMatrixAVX512_Grouped<M_15, M_15> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 * matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MLT_MAT_15_AVX512);


static void BM_MLT_MAT_20_AVX512(benchmark::State& state) {
    const size_t num_intervals = M_20 * M_20;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX512_Grouped<M_20, M_20> matrix1(intervals1_data);
    BatchSwitchMatrixAVX512_Grouped<M_20, M_20> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 * matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MLT_MAT_20_AVX512);


static void BM_MLT_MAT_25_AVX512(benchmark::State& state) {
    const size_t num_intervals = M_25 * M_25;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX512_Grouped<M_25, M_25> matrix1(intervals1_data);
    BatchSwitchMatrixAVX512_Grouped<M_25, M_25> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 * matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MLT_MAT_25_AVX512);


static void BM_MLT_MAT_30_AVX512(benchmark::State& state) {
    const size_t num_intervals = M_30 * M_30;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX512_Grouped<M_30, M_30> matrix1(intervals1_data);
    BatchSwitchMatrixAVX512_Grouped<M_30, M_30> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 * matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MLT_MAT_30_AVX512);


static void BM_MLT_MAT_35_AVX512(benchmark::State& state) {
    const size_t num_intervals = M_35 * M_35;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX512_Grouped<M_35, M_35> matrix1(intervals1_data);
    BatchSwitchMatrixAVX512_Grouped<M_35, M_35> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 * matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MLT_MAT_35_AVX512);


static void BM_MLT_MAT_40_AVX512(benchmark::State& state) {
    const size_t num_intervals = M_40 * M_40;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX512_Grouped<M_40, M_40> matrix1(intervals1_data);
    BatchSwitchMatrixAVX512_Grouped<M_40, M_40> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 * matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MLT_MAT_40_AVX512);


static void BM_MLT_MAT_45_AVX512(benchmark::State& state) {
    const size_t num_intervals = M_45 * M_45;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX512_Grouped<M_45, M_45> matrix1(intervals1_data);
    BatchSwitchMatrixAVX512_Grouped<M_45, M_45> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 * matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MLT_MAT_45_AVX512);


static void BM_MLT_MAT_50_AVX512(benchmark::State& state) {
    const size_t num_intervals = M_50 * M_50;
    auto intervals1_data = generateArray(num_intervals, 42);
    auto intervals2_data = generateArray(num_intervals, 143);
    BatchSwitchMatrixAVX512_Grouped<M_50, M_50> matrix1(intervals1_data);
    BatchSwitchMatrixAVX512_Grouped<M_50, M_50> matrix2(intervals2_data);
    for (auto _ : state) {
        auto result = matrix1 * matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MLT_MAT_50_AVX512);

BENCHMARK_MAIN();