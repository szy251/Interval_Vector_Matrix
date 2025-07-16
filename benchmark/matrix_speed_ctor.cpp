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

static void BM_CTOR_16_16_UPP(benchmark::State& state) {
    capd::rounding::DoubleRounding l;
    l.roundUp();

    for (auto _ : state) {
        MatrixBasic<opt::Interval, n1, n1> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_16_16_UPP);

static void BM_CTOR_16_16_CAPD(benchmark::State& state) {

    for (auto _ : state) {
        MatrixBasic<capd::filib::Interval<double>, n1,n1> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_16_16_CAPD);

static void BM_CTOR_16_16_BATCH_MIX(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixMixed<n1, n1> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_16_16_BATCH_MIX);

static void BM_CTOR_16_16_BATCH_GRP(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrix_Grouped<n1, n1> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_16_16_BATCH_GRP);

static void BM_CTOR_16_16_BATCH_GRP_AVX(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixAVX_Grouped<n1, n1> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_16_16_BATCH_GRP_AVX);

static void BM_CTOR_16_16_BATCH_GRP_AVX512(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixAVX512_Grouped<n1, n1> a ;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_16_16_BATCH_GRP_AVX512);


static void BM_CTOR_104_104_UPP(benchmark::State& state) {
    capd::rounding::DoubleRounding l;
    l.roundUp();
    for (auto _ : state) {
        MatrixBasic<opt::Interval,n2, n2> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_104_104_UPP);

static void BM_CTOR_104_104_CAPD(benchmark::State& state) {

    for (auto _ : state) {
       MatrixBasic<capd::filib::Interval<double>,n2, n2> a; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_104_104_CAPD);

static void BM_CTOR_104_104_BATCH_MIX(benchmark::State& state) {

    for (auto _ : state) {
        BatchSwitchMatrixMixed<n2, n2> a; 
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_104_104_BATCH_MIX);

static void BM_CTOR_104_104_BATCH_GRP(benchmark::State& state) {

    for (auto _ : state) {
       BatchSwitchMatrix_Grouped<n2, n2> a;
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_104_104_BATCH_GRP);

static void BM_CTOR_104_104_BATCH_GRP_AVX(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixAVX_Grouped<n2, n2> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_104_104_BATCH_GRP_AVX);

static void BM_CTOR_104_104_BATCH_GRP_AVX512(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixAVX512_Grouped<n2, n2> a; 
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_104_104_BATCH_GRP_AVX512);


static void BM_CTOR_304_304_UPP(benchmark::State& state) {
    capd::rounding::DoubleRounding l;
    l.roundUp();
    for (auto _ : state) {
        MatrixBasic<opt::Interval,n3, n3> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_304_304_UPP);

static void BM_CTOR_304_304_CAPD(benchmark::State& state) {

    for (auto _ : state) {
       MatrixBasic<capd::filib::Interval<double>,n3, n3> a; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_304_304_CAPD);

static void BM_CTOR_304_304_BATCH_MIX(benchmark::State& state) {

    for (auto _ : state) {
        BatchSwitchMatrixMixed<n3, n3> a; 
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_304_304_BATCH_MIX);

static void BM_CTOR_304_304_BATCH_GRP(benchmark::State& state) {

    for (auto _ : state) {
       BatchSwitchMatrix_Grouped<n3, n3> a;
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_304_304_BATCH_GRP);

static void BM_CTOR_304_304_BATCH_GRP_AVX(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixAVX_Grouped<n3, n3> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_304_304_BATCH_GRP_AVX);

static void BM_CTOR_304_304_BATCH_GRP_AVX512(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchMatrixAVX512_Grouped<n3, n3> a; 
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_304_304_BATCH_GRP_AVX512);


BENCHMARK_MAIN();
  