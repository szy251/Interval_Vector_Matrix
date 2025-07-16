#include <benchmark/benchmark.h>
#include <Interval.hpp>
#include <VectorBasic.hpp>
#include <BatchSwitchVectorMixed.hpp>
#include <BatchSwitchVector_Grouped.hpp>
#include <BatchSwitchVectorAVX_G256.hpp>
#include <BatchSwitchVectorAVX_Grouped.hpp>
#include <BatchSwitchVectorAVX512_Grouped.hpp>
#include <capd/rounding/DoubleRounding.h>
#include <capd/filib/Interval.h>


static constexpr size_t n1 = 16;
static constexpr size_t n2 = 104;
static constexpr size_t n3 = 304;


static void BM_CTOR_16_16_UPP(benchmark::State& state) {
    capd::rounding::DoubleRounding l;
    l.roundUp();               

    for (auto _ : state) {
        VectorBasic<opt::Interval, n1 * n1> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_16_16_UPP);

static void BM_CTOR_16_16_CAPD(benchmark::State& state) {
    for (auto _ : state) {
        VectorBasic<capd::filib::Interval<double>, n1 * n1> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_16_16_CAPD);

static void BM_CTOR_16_16_BATCH_MIX(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchVectorMixed<n1 * n1> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_16_16_BATCH_MIX);

static void BM_CTOR_16_16_BATCH_GRP(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchVector_Grouped<n1 * n1> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_16_16_BATCH_GRP);

static void BM_CTOR_16_16_BATCH_GRP_AVX(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchVectorAVX_Grouped<n1 * n1> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_16_16_BATCH_GRP_AVX);

static void BM_CTOR_16_16_BATCH_AVX_G256(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchVectorAVX_G256<n1 * n1> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_16_16_BATCH_AVX_G256);


static void BM_CTOR_16_16_BATCH_GRP_AVX512(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchVectorAVX512_Grouped<n1 * n1> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_16_16_BATCH_GRP_AVX512);



static void BM_CTOR_104_104_UPP(benchmark::State& state) {
    capd::rounding::DoubleRounding l;
    l.roundUp();
    for (auto _ : state) {
        VectorBasic<opt::Interval, n2 * n2> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_104_104_UPP);

static void BM_CTOR_104_104_CAPD(benchmark::State& state) {
    for (auto _ : state) {
        VectorBasic<capd::filib::Interval<double>, n2 * n2> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_104_104_CAPD);

static void BM_CTOR_104_104_BATCH_MIX(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchVectorMixed<n2 * n2> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_104_104_BATCH_MIX);

static void BM_CTOR_104_104_BATCH_GRP(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchVector_Grouped<n2 * n2> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_104_104_BATCH_GRP);

static void BM_CTOR_104_104_BATCH_GRP_AVX(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchVectorAVX_Grouped<n2 * n2> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_104_104_BATCH_GRP_AVX);

static void BM_CTOR_104_104_BATCH_AVX_G256(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchVectorAVX_G256<n2 * n2> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_104_104_BATCH_AVX_G256);


static void BM_CTOR_104_104_BATCH_GRP_AVX512(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchVectorAVX512_Grouped<n2 * n2> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_104_104_BATCH_GRP_AVX512);


static void BM_CTOR_304_304_UPP(benchmark::State& state) {
    capd::rounding::DoubleRounding l;
    l.roundUp();
    for (auto _ : state) {
        VectorBasic<opt::Interval, n3 * n3> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_304_304_UPP);

static void BM_CTOR_304_304_CAPD(benchmark::State& state) {
    for (auto _ : state) {
        VectorBasic<capd::filib::Interval<double>, n3 * n3> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_304_304_CAPD);

static void BM_CTOR_304_304_BATCH_MIX(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchVectorMixed<n3 * n3> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_304_304_BATCH_MIX);

static void BM_CTOR_304_304_BATCH_GRP(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchVector_Grouped<n3 * n3> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_304_304_BATCH_GRP);

static void BM_CTOR_304_304_BATCH_GRP_AVX(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchVectorAVX_Grouped<n3 * n3> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_304_304_BATCH_GRP_AVX);

static void BM_CTOR_304_304_BATCH_AVX_G256(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchVectorAVX_G256<n3 * n3> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_304_304_BATCH_AVX_G256);

static void BM_CTOR_304_304_BATCH_GRP_AVX512(benchmark::State& state) {
    for (auto _ : state) {
        BatchSwitchVectorAVX512_Grouped<n3 * n3> a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_304_304_BATCH_GRP_AVX512);


BENCHMARK_MAIN();