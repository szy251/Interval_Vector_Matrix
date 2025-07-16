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
#include <Utilities.hpp>


static constexpr size_t n1 = 16;
static constexpr size_t n2 = 104;
static constexpr size_t n3 = 304;


static void BM_CTOR_16_16_UPP(benchmark::State& state) {
    const size_t array_size = n1 * n1;
    auto intervals_vec = generateArray(array_size, 42);
    capd::rounding::DoubleRounding l;
    l.roundUp();
    for (auto _ : state) {
        VectorBasic<opt::Interval, array_size> a(intervals_vec);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_16_16_UPP);

static void BM_CTOR_16_16_CAPD(benchmark::State& state) {
    const size_t array_size = n1 * n1;
    auto intervals_vec = generateArray(array_size, 42);
    for (auto _ : state) {
        VectorBasic<capd::filib::Interval<double>, array_size> a(intervals_vec);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_16_16_CAPD);

static void BM_CTOR_16_16_BATCH_MIX(benchmark::State& state) {
    const size_t array_size = n1 * n1;
    auto intervals_vec = generateArray(array_size, 42);
    for (auto _ : state) {
        BatchSwitchVectorMixed<array_size> a(intervals_vec);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_16_16_BATCH_MIX);

static void BM_CTOR_16_16_BATCH_GRP(benchmark::State& state) {
    const size_t array_size = n1 * n1;
    auto intervals_vec = generateArray(array_size, 42);
    for (auto _ : state) {
        BatchSwitchVector_Grouped<array_size> a(intervals_vec);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_16_16_BATCH_GRP);

static void BM_CTOR_16_16_BATCH_GRP_AVX(benchmark::State& state) {
    const size_t array_size = n1 * n1;
    auto intervals_vec = generateArray(array_size, 42);
    for (auto _ : state) {
        BatchSwitchVectorAVX_Grouped<array_size> a(intervals_vec);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_16_16_BATCH_GRP_AVX);

static void BM_CTOR_16_16_BATCH_AVX_G256(benchmark::State& state) {
    const size_t array_size = n1 * n1;
    auto intervals_vec = generateArray(array_size, 42);
    for (auto _ : state) {
        BatchSwitchVectorAVX_G256<array_size> a(intervals_vec);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_16_16_BATCH_AVX_G256);


static void BM_CTOR_16_16_BATCH_GRP_AVX512(benchmark::State& state) {
    const size_t array_size = n1 * n1;
    auto intervals_vec = generateArray(array_size, 42);
    for (auto _ : state) {
        BatchSwitchVectorAVX512_Grouped<array_size> a(intervals_vec);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_16_16_BATCH_GRP_AVX512);



static void BM_CTOR_104_104_UPP(benchmark::State& state) {
    const size_t array_size = n2 * n2;
    auto intervals_vec = generateArray(array_size, 42);
    capd::rounding::DoubleRounding l;
    l.roundUp();
    for (auto _ : state) {
        VectorBasic<opt::Interval, array_size> a(intervals_vec);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_104_104_UPP);

static void BM_CTOR_104_104_CAPD(benchmark::State& state) {
    const size_t array_size = n2 * n2;
    auto intervals_vec = generateArray(array_size, 42);
    for (auto _ : state) {
        VectorBasic<capd::filib::Interval<double>, array_size> a(intervals_vec);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_104_104_CAPD);

static void BM_CTOR_104_104_BATCH_MIX(benchmark::State& state) {
    const size_t array_size = n2 * n2;
    auto intervals_vec = generateArray(array_size, 42);
    for (auto _ : state) {
        BatchSwitchVectorMixed<array_size> a(intervals_vec);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_104_104_BATCH_MIX);

static void BM_CTOR_104_104_BATCH_GRP(benchmark::State& state) {
    const size_t array_size = n2 * n2;
    auto intervals_vec = generateArray(array_size, 42);
    for (auto _ : state) {
        BatchSwitchVector_Grouped<array_size> a(intervals_vec);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_104_104_BATCH_GRP);

static void BM_CTOR_104_104_BATCH_GRP_AVX(benchmark::State& state) {
    const size_t array_size = n2 * n2;
    auto intervals_vec = generateArray(array_size, 42);
    for (auto _ : state) {
        BatchSwitchVectorAVX_Grouped<array_size> a(intervals_vec);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_104_104_BATCH_GRP_AVX);

static void BM_CTOR_104_104_BATCH_AVX_G256(benchmark::State& state) {
    const size_t array_size = n2 * n2;
    auto intervals_vec = generateArray(array_size, 42);
    for (auto _ : state) {
        BatchSwitchVectorAVX_G256<array_size> a(intervals_vec);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_104_104_BATCH_AVX_G256);


static void BM_CTOR_104_104_BATCH_GRP_AVX512(benchmark::State& state) {
    const size_t array_size = n2 * n2;
    auto intervals_vec = generateArray(array_size, 42);
    for (auto _ : state) {
        BatchSwitchVectorAVX512_Grouped<array_size> a(intervals_vec);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_104_104_BATCH_GRP_AVX512);


static void BM_CTOR_304_304_UPP(benchmark::State& state) {
    const size_t array_size = n3 * n3;
    auto intervals_vec = generateArray(array_size, 42);
    capd::rounding::DoubleRounding l;
    l.roundUp();
    for (auto _ : state) {
        VectorBasic<opt::Interval, array_size> a(intervals_vec);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_304_304_UPP);

static void BM_CTOR_304_304_CAPD(benchmark::State& state) {
    const size_t array_size = n3 * n3;
    auto intervals_vec = generateArray(array_size, 42);
    for (auto _ : state) {
        VectorBasic<capd::filib::Interval<double>, array_size> a(intervals_vec);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_304_304_CAPD);

static void BM_CTOR_304_304_BATCH_MIX(benchmark::State& state) {
    const size_t array_size = n3 * n3;
    auto intervals_vec = generateArray(array_size, 42);
    for (auto _ : state) {
        BatchSwitchVectorMixed<array_size> a(intervals_vec);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_304_304_BATCH_MIX);

static void BM_CTOR_304_304_BATCH_GRP(benchmark::State& state) {
    const size_t array_size = n3 * n3;
    auto intervals_vec = generateArray(array_size, 42);
    for (auto _ : state) {
        BatchSwitchVector_Grouped<array_size> a(intervals_vec);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_304_304_BATCH_GRP);

static void BM_CTOR_304_304_BATCH_GRP_AVX(benchmark::State& state) {
    const size_t array_size = n3 * n3;
    auto intervals_vec = generateArray(array_size, 42);
    for (auto _ : state) {
        BatchSwitchVectorAVX_Grouped<array_size> a(intervals_vec);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_304_304_BATCH_GRP_AVX);

static void BM_CTOR_304_304_BATCH_AVX_G256(benchmark::State& state) {
    const size_t array_size = n3 * n3;
    auto intervals_vec = generateArray(array_size, 42);
    for (auto _ : state) {
        BatchSwitchVectorAVX_G256<array_size> a(intervals_vec);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_304_304_BATCH_AVX_G256);

static void BM_CTOR_304_304_BATCH_GRP_AVX512(benchmark::State& state) {
    const size_t array_size = n3 * n3;
    auto intervals_vec = generateArray(array_size, 42);
    for (auto _ : state) {
        BatchSwitchVectorAVX512_Grouped<array_size> a(intervals_vec);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_CTOR_304_304_BATCH_GRP_AVX512);


BENCHMARK_MAIN();