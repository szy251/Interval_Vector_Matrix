#include<benchmark/benchmark.h>
#include<Interval.hpp>
#include<MatrixBasic.hpp>
#include<BatchSwitchMatrixMixed.hpp>
#include<BatchSwitchMatrix_Grouped.hpp>
#include<BatchSwitchMatrixAVX_Grouped.hpp>
#include<BatchSwitchMatrixAVX512_Grouped.hpp>
#include <capd/rounding/DoubleRounding.h>
#include<capd/filib/Interval.h>
#include<Utilities.hpp>

static constexpr size_t n1 = 16;
static constexpr size_t n2 = 104;
static constexpr size_t n3 = 304;

static void BM_MLT_16_16_UPP(benchmark::State& state) {
    const size_t array_size = n1*n1;
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    

    MatrixBasic<opt::Interval, n1, n1> matrix1(intervals);
    MatrixBasic<opt::Interval, n1, n1> matrix2(intervals2);
    capd::rounding::DoubleRounding l;
    l.roundUp();

    for (auto _ : state) {
        auto a = matrix1 * matrix2;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_16_16_UPP);

static void BM_MLT_16_16_CAPD(benchmark::State& state) {
    const size_t array_size = n1*n1;
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);

    MatrixBasic<capd::filib::Interval<double>, n1,n1> matrix1(intervals);
    MatrixBasic<capd::filib::Interval<double>, n1,n1> matrix2(intervals2);

    for (auto _ : state) {
        auto a = matrix1 * matrix2;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_16_16_CAPD);

static void BM_MLT_16_16_BATCH_MIX(benchmark::State& state) {
    const size_t array_size = n1*n1;
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    
    BatchSwitchMatrixMixed<n1, n1> matrix1(intervals);
    BatchSwitchMatrixMixed<n1, n1> matrix2(intervals2);
    for (auto _ : state) {
        auto a = matrix1 * matrix2;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_16_16_BATCH_MIX);

static void BM_MLT_16_16_BATCH_GRP(benchmark::State& state) {
    const size_t array_size = n1*n1; 
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    

    BatchSwitchMatrix_Grouped<n1, n1> matrix1(intervals);
    BatchSwitchMatrix_Grouped<n1, n1> matrix2(intervals2);
    for (auto _ : state) {
        auto a = matrix1 * matrix2;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_16_16_BATCH_GRP);

static void BM_MLT_16_16_BATCH_GRP_AVX(benchmark::State& state) {
    const size_t array_size = n1*n1; 
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    

    BatchSwitchMatrixAVX_Grouped<n1, n1> matrix1(intervals);
    BatchSwitchMatrixAVX_Grouped<n1, n1> matrix2(intervals2);
    for (auto _ : state) {
        auto a = matrix1 * matrix2;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_16_16_BATCH_GRP_AVX);

static void BM_MLT_16_16_BATCH_GRP_AVX512(benchmark::State& state) {
    const size_t array_size = n1*n1;
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    

    BatchSwitchMatrixAVX512_Grouped<n1, n1> matrix1(intervals);
    BatchSwitchMatrixAVX512_Grouped<n1, n1> matrix2(intervals2);
    for (auto _ : state) {
        auto a = matrix1 * matrix2;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_16_16_BATCH_GRP_AVX512);


static void BM_MLT_104_104_UPP(benchmark::State& state) {
    const size_t array_size = n2*n2;
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    MatrixBasic<opt::Interval,n2, n2> matrix1(intervals);
    MatrixBasic<opt::Interval,n2, n2> matrix2(intervals2);
    capd::rounding::DoubleRounding l;
    l.roundUp();
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_104_104_UPP);

static void BM_MLT_104_104_CAPD(benchmark::State& state) {
    const size_t array_size = n2*n2;
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    MatrixBasic<capd::filib::Interval<double>,n2, n2> matrix1(intervals);
    MatrixBasic<capd::filib::Interval<double>,n2, n2> matrix2(intervals2);

    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_104_104_CAPD);

static void BM_MLT_104_104_BATCH_MIX(benchmark::State& state) {
    const size_t array_size = n2*n2;
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    BatchSwitchMatrixMixed<n2, n2> matrix1(intervals);
    BatchSwitchMatrixMixed<n2, n2> matrix2(intervals2);
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_104_104_BATCH_MIX);

static void BM_MLT_104_104_BATCH_GRP(benchmark::State& state) {
    const size_t array_size = n2*n2;
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    BatchSwitchMatrix_Grouped<n2, n2> matrix1(intervals);
    BatchSwitchMatrix_Grouped<n2, n2> matrix2(intervals2);
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_104_104_BATCH_GRP);

static void BM_MLT_104_104_BATCH_GRP_AVX(benchmark::State& state) {
    const size_t array_size = n2*n2;
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    BatchSwitchMatrixAVX_Grouped<n2, n2> matrix1(intervals);
    BatchSwitchMatrixAVX_Grouped<n2, n2> matrix2(intervals2);
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_104_104_BATCH_GRP_AVX);

static void BM_MLT_104_104_BATCH_GRP_AVX512(benchmark::State& state) {
    const size_t array_size = n2*n2;
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    BatchSwitchMatrixAVX512_Grouped<n2, n2> matrix1(intervals);
    BatchSwitchMatrixAVX512_Grouped<n2, n2> matrix2(intervals2);
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_104_104_BATCH_GRP_AVX512);

static void BM_MLT_304_304_UPP(benchmark::State& state) {
    const size_t array_size = n3*n3; 
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    MatrixBasic<opt::Interval,n3, n3> matrix1(intervals);
    MatrixBasic<opt::Interval,n3, n3> matrix2(intervals2);
    capd::rounding::DoubleRounding l;

    l.roundUp();
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_304_304_UPP);

static void BM_MLT_304_304_CAPD(benchmark::State& state) {
    const size_t array_size = n3*n3;
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    MatrixBasic<capd::filib::Interval<double>,n3, n3> matrix1(intervals);
    MatrixBasic<capd::filib::Interval<double>,n3, n3> matrix2(intervals2);

    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_304_304_CAPD);


static void BM_MLT_304_304_BATCH_MIX(benchmark::State& state) {
    const size_t array_size = n3*n3; 
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    BatchSwitchMatrixMixed<n3, n3> matrix1(intervals);
    BatchSwitchMatrixMixed<n3, n3> matrix2(intervals2);
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_304_304_BATCH_MIX);

static void BM_MLT_304_304_BATCH_GRP(benchmark::State& state) {
    const size_t array_size = n3*n3; 
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    BatchSwitchMatrix_Grouped<n3, n3> matrix1(intervals);
    BatchSwitchMatrix_Grouped<n3, n3> matrix2(intervals2);
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_304_304_BATCH_GRP);

static void BM_MLT_304_304_BATCH_GRP_AVX(benchmark::State& state) {
    const size_t array_size = n3*n3; 
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    BatchSwitchMatrixAVX_Grouped<n3, n3> matrix1(intervals);
    BatchSwitchMatrixAVX_Grouped<n3, n3> matrix2(intervals2);
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_304_304_BATCH_GRP_AVX);

static void BM_MLT_304_304_BATCH_GRP_AVX512(benchmark::State& state) {
    const size_t array_size = n3*n3; 
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    BatchSwitchMatrixAVX512_Grouped<n3, n3> matrix1(intervals);
    BatchSwitchMatrixAVX512_Grouped<n3, n3> matrix2(intervals2);
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_304_304_BATCH_GRP_AVX512);



BENCHMARK_MAIN();
  