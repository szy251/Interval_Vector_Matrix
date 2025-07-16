#include <benchmark/benchmark.h>
#include "BatchSwitchMatrixAVX_Grouped.hpp"
#include "BatchSwitchMatrixAVX512_Grouped.hpp"

static constexpr size_t N_25 = 25;
static constexpr size_t N_50 = 50;
static constexpr size_t N_75 = 75;
static constexpr size_t N_100 = 100;
static constexpr size_t N_125 = 125;
static constexpr size_t N_150 = 150;
static constexpr size_t N_175 = 175;
static constexpr size_t N_200 = 200;

static void BM_CTOR_DEFAULT_25_AVX512(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixAVX512_Grouped<N_25, N_25> matrix;
        benchmark::DoNotOptimize(matrix);
    }
}

BENCHMARK(BM_CTOR_DEFAULT_25_AVX512);
static void BM_CTOR_DEFAULT_50_AVX512(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixAVX512_Grouped<N_50, N_50> matrix;
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(BM_CTOR_DEFAULT_50_AVX512);
static void BM_CTOR_DEFAULT_75_AVX512(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixAVX512_Grouped<N_75, N_75> matrix;
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(BM_CTOR_DEFAULT_75_AVX512);
static void BM_CTOR_DEFAULT_100_AVX512(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixAVX512_Grouped<N_100, N_100> matrix;
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(BM_CTOR_DEFAULT_100_AVX512);
static void BM_CTOR_DEFAULT_125_AVX512(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixAVX512_Grouped<N_125, N_125> matrix;
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(BM_CTOR_DEFAULT_125_AVX512);
static void BM_CTOR_DEFAULT_150_AVX512(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixAVX512_Grouped<N_150, N_150> matrix;
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(BM_CTOR_DEFAULT_150_AVX512);
static void BM_CTOR_DEFAULT_175_AVX512(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixAVX512_Grouped<N_175, N_175> matrix;
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(BM_CTOR_DEFAULT_175_AVX512);
static void BM_CTOR_DEFAULT_200_AVX512(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixAVX512_Grouped<N_200, N_200> matrix;
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(BM_CTOR_DEFAULT_200_AVX512);
static void BM_CTOR_DEFAULT_25_AVX(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixAVX_Grouped<N_25, N_25> matrix;
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(BM_CTOR_DEFAULT_25_AVX);
static void BM_CTOR_DEFAULT_50_AVX(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixAVX_Grouped<N_50, N_50> matrix;
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(BM_CTOR_DEFAULT_50_AVX);
static void BM_CTOR_DEFAULT_75_AVX(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixAVX_Grouped<N_75, N_75> matrix;
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(BM_CTOR_DEFAULT_75_AVX);
static void BM_CTOR_DEFAULT_100_AVX(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixAVX_Grouped<N_100, N_100> matrix;
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(BM_CTOR_DEFAULT_100_AVX);
static void BM_CTOR_DEFAULT_125_AVX(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixAVX_Grouped<N_125, N_125> matrix;
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(BM_CTOR_DEFAULT_125_AVX);
static void BM_CTOR_DEFAULT_150_AVX(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixAVX_Grouped<N_150, N_150> matrix;
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(BM_CTOR_DEFAULT_150_AVX);
static void BM_CTOR_DEFAULT_175_AVX(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixAVX_Grouped<N_175, N_175> matrix;
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(BM_CTOR_DEFAULT_175_AVX);
static void BM_CTOR_DEFAULT_200_AVX(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixAVX_Grouped<N_200, N_200> matrix;
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(BM_CTOR_DEFAULT_200_AVX);

BENCHMARK(BM_CTOR_DEFAULT_200_AVX);




BENCHMARK_MAIN();