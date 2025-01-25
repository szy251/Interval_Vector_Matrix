#include<benchmark/benchmark.h>
#include<Interval.hpp>
#include<VectorBasic.hpp>
#include<BatchSwitchVectorMixed.hpp>
#include<BatchSwitchVector_Grouped.hpp>
#include<BatchSwitchVectorAVX_G256.hpp>
#include<BatchSwitchVectorAVX_Grouped.hpp>
#include<BatchSwitchVectorAVX512_Grouped.hpp>
#include<capd/intervals/Interval.h>

static void BM_ADD_100_100_OPT(benchmark::State& state) {
    const size_t array_size = 10000;  // 100x100 macierz
    opt::Interval intervals[array_size];
    opt::Interval intervals2[array_size];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i] = opt::Interval(i * 1.0, i * 1.0 + 1.0);
        intervals2[i] = opt::Interval(i * 2.0, i * 2.0 + 1.0);
    }
    VectorBasic<opt::Interval, 100*100> vector1(intervals);
    VectorBasic<opt::Interval, 100*100> vector2(intervals2);
    capd::rounding::DoubleRounding l;
    l.roundUp();
    for (auto _ : state) {
        auto a = vector1 + vector2;  // Zmieniono mnożenie na dodawanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ADD_100_100_OPT);

static void BM_ADD_100_100_CAPD(benchmark::State& state) {
    const size_t array_size = 10000;  // 100x100 macierz
    capd::intervals::Interval<double> intervals[array_size];
    capd::intervals::Interval<double> intervals2[array_size];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i] = capd::intervals::Interval<double>(i * 1.0, i * 1.0 + 1.0);
        intervals2[i] = capd::intervals::Interval<double>(i * 2.0, i * 2.0 + 1.0);
    }
    VectorBasic<capd::intervals::Interval<double>, 100*100> vector1(intervals);
    VectorBasic<capd::intervals::Interval<double>, 100*100> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 + vector2;  // Zmieniono mnożenie na dodawanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ADD_100_100_CAPD);

static void BM_ADD_100_100_BATCH(benchmark::State& state) {
    const size_t array_size = 10000;  // 100x100 macierz
    double intervals[array_size*2];
    double intervals2[array_size*2];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i*2] = i * 1.0;
        intervals[i*2+1] = i * 1.0 + 1.0;
        intervals2[i*2] = i * 2.0;
        intervals2[i*2+1] = i * 2.0 + 1.0;
    }
    BatchSwitchVectorMixed<100*100> vector1(intervals);
    BatchSwitchVectorMixed<100*100> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 + vector2;  // Zmieniono mnożenie na dodawanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ADD_100_100_BATCH);

static void BM_ADD_100_100_BATCH_OPT(benchmark::State& state) {
    const size_t array_size = 10000;  // 100x100 macierz
    double intervals[array_size*2];
    double intervals2[array_size*2];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i*2] = i * 1.0;
        intervals[i*2+1] = i * 1.0 + 1.0;
        intervals2[i*2] = i * 2.0;
        intervals2[i*2+1] = i * 2.0 + 1.0;
    }
    BatchSwitchVector_Grouped<100*100> vector1(intervals);
    BatchSwitchVector_Grouped<100*100> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 + vector2;  // Zmieniono mnożenie na dodawanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ADD_100_100_BATCH_OPT);

static void BM_ADD_100_100_BATCH_AVX(benchmark::State& state) {
    const size_t array_size = 10000;  // 100x100 macierz
    double intervals[array_size*2];
    double intervals2[array_size*2];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i*2] = i * 1.0;
        intervals[i*2+1] = i * 1.0 + 1.0;
        intervals2[i*2] = i * 2.0;
        intervals2[i*2+1] = i * 2.0 + 1.0;
    }
    BatchSwitchVectorAVX_G256<100*100> vector1(intervals);
    BatchSwitchVectorAVX_G256<100*100> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 + vector2;  // Zmieniono mnożenie na dodawanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ADD_100_100_BATCH_AVX);

static void BM_ADD_100_100_BATCH_AVX_OPT(benchmark::State& state) {
    const size_t array_size = 10000;  // 100x100 macierz
    double intervals[array_size*2];
    double intervals2[array_size*2];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i*2] = i * 1.0;
        intervals[i*2+1] = i * 1.0 + 1.0;
        intervals2[i*2] = i * 2.0;
        intervals2[i*2+1] = i * 2.0 + 1.0;
    }
    BatchSwitchVectorAVX_Grouped<100*100> vector1(intervals);
    BatchSwitchVectorAVX_Grouped<100*100> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 + vector2;  // Zmieniono mnożenie na dodawanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ADD_100_100_BATCH_AVX_OPT);

static void BM_ADD_100_100_BATCH_AVX512_OPT(benchmark::State& state) {
    const size_t array_size = 10000;  // 100x100 macierz
    double intervals[array_size*2];
    double intervals2[array_size*2];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i*2] = i * 1.0;
        intervals[i*2+1] = i * 1.0 + 1.0;
        intervals2[i*2] = i * 2.0;
        intervals2[i*2+1] = i * 2.0 + 1.0;
    }
    BatchSwitchVectorAVX512_Grouped<100*100> vector1(intervals);
    BatchSwitchVectorAVX512_Grouped<100*100> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 + vector2;  // Zmieniono mnożenie na dodawanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ADD_100_100_BATCH_AVX512_OPT);

static void BM_ADD_300_300_OPT(benchmark::State& state) {
    const size_t array_size = 90000;  // 300x300 macierz
    opt::Interval intervals[array_size];
    opt::Interval intervals2[array_size];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i] = opt::Interval(i * 1.0, i * 1.0 + 1.0);
        intervals2[i] = opt::Interval(i * 2.0, i * 2.0 + 1.0);
    }
    VectorBasic<opt::Interval, 300*300> vector1(intervals);
    VectorBasic<opt::Interval, 300*300> vector2(intervals2);
    capd::rounding::DoubleRounding l;
    l.roundUp();
    for (auto _ : state) {
        auto a = vector1 + vector2;  // Zmieniono mnożenie na dodawanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ADD_300_300_OPT);

static void BM_ADD_300_300_CAPD(benchmark::State& state) {
    const size_t array_size = 90000;  // 300x300 macierz
    capd::intervals::Interval<double> intervals[array_size];
    capd::intervals::Interval<double> intervals2[array_size];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i] = capd::intervals::Interval<double>(i * 1.0, i * 1.0 + 1.0);
        intervals2[i] = capd::intervals::Interval<double>(i * 2.0, i * 2.0 + 1.0);
    }
    VectorBasic<capd::intervals::Interval<double>, 300*300> vector1(intervals);
    VectorBasic<capd::intervals::Interval<double>, 300*300> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 + vector2;  // Zmieniono mnożenie na dodawanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ADD_300_300_CAPD);

static void BM_ADD_300_300_BATCH(benchmark::State& state) {
    const size_t array_size = 90000;  // 300x300 macierz
    double intervals[array_size*2];
    double intervals2[array_size*2];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i*2] = i * 1.0;
        intervals[i*2+1] = i* 1.0 +1.0;
        intervals[i*2] = i * 2.0;
        intervals[i*2+1] = i* 2.0 +1.0;
    }
    BatchSwitchVectorMixed<300*300> vector1(intervals);
    BatchSwitchVectorMixed<300*300> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 + vector2;  // Zmieniono mnożenie na dodawanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ADD_300_300_BATCH);

static void BM_ADD_300_300_BATCH_OPT(benchmark::State& state) {
    const size_t array_size = 90000;  // 300x300 macierz
    double intervals[array_size*2];
    double intervals2[array_size*2];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i*2] = i * 1.0;
        intervals[i*2+1] = i* 1.0 +1.0;
        intervals[i*2] = i * 2.0;
        intervals[i*2+1] = i* 2.0 +1.0;
    }
    BatchSwitchVector_Grouped<300*300> vector1(intervals);
    BatchSwitchVector_Grouped<300*300> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 + vector2;  // Zmieniono mnożenie na dodawanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ADD_300_300_BATCH_OPT);

static void BM_ADD_300_300_BATCH_AVX(benchmark::State& state) {
    const size_t array_size = 90000;  // 300x300 macierz
    double intervals[array_size*2];
    double intervals2[array_size*2];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i*2] = i * 1.0;
        intervals[i*2+1] = i* 1.0 +1.0;
        intervals[i*2] = i * 2.0;
        intervals[i*2+1] = i* 2.0 +1.0;
    }
    BatchSwitchVectorAVX_G256<300*300> vector1(intervals);
    BatchSwitchVectorAVX_G256<300*300> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 + vector2;  // Zmieniono mnożenie na dodawanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ADD_300_300_BATCH_AVX);

static void BM_ADD_300_300_BATCH_AVX_OPT(benchmark::State& state) {
    const size_t array_size = 90000;  // 300x300 macierz
    double intervals[array_size*2];
    double intervals2[array_size*2];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i*2] = i * 1.0;
        intervals[i*2+1] = i* 1.0 +1.0;
        intervals[i*2] = i * 2.0;
        intervals[i*2+1] = i* 2.0 +1.0;
    }
    BatchSwitchVectorAVX_Grouped<300*300> vector1(intervals);
    BatchSwitchVectorAVX_Grouped<300*300> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 + vector2;  // Zmieniono mnożenie na dodawanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ADD_300_300_BATCH_AVX_OPT);

static void BM_ADD_300_300_BATCH_AVX512_OPT(benchmark::State& state) {
    const size_t array_size = 90000;  // 300x300 macierz
    double intervals[array_size*2];
    double intervals2[array_size*2];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i*2] = i * 1.0;
        intervals[i*2+1] = i* 1.0 +1.0;
        intervals[i*2] = i * 2.0;
        intervals[i*2+1] = i* 2.0 +1.0;
    }
    BatchSwitchVectorAVX512_Grouped<300*300> vector1(intervals);
    BatchSwitchVectorAVX512_Grouped<300*300> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 + vector2;  // Zmieniono mnożenie na dodawanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ADD_300_300_BATCH_AVX512_OPT);


static void BM_SUB_100_100_OPT(benchmark::State& state) {
    const size_t array_size = 10000;  // 100x100 macierz
    opt::Interval intervals[array_size];
    opt::Interval intervals2[array_size];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i] = opt::Interval(i * 1.0, i * 1.0 + 1.0);
        intervals2[i] = opt::Interval(i * 2.0, i * 2.0 + 1.0);
    }
    VectorBasic<opt::Interval, 100*100> vector1(intervals);
    VectorBasic<opt::Interval, 100*100> vector2(intervals2);
    capd::rounding::DoubleRounding l;
    l.roundUp();
    for (auto _ : state) {
        auto a = vector1 - vector2;  // Zmieniono dodawanie na odejmowanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_SUB_100_100_OPT);

static void BM_SUB_100_100_CAPD(benchmark::State& state) {
    const size_t array_size = 10000;  // 100x100 macierz
    capd::intervals::Interval<double> intervals[array_size];
    capd::intervals::Interval<double> intervals2[array_size];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i] = capd::intervals::Interval<double>(i * 1.0, i * 1.0 + 1.0);
        intervals2[i] = capd::intervals::Interval<double>(i * 2.0, i * 2.0 + 1.0);
    }
    VectorBasic<capd::intervals::Interval<double>, 100*100> vector1(intervals);
    VectorBasic<capd::intervals::Interval<double>, 100*100> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 - vector2;  // Zmieniono mnożenie na dodawanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_SUB_100_100_CAPD);

static void BM_SUB_100_100_BATCH(benchmark::State& state) {
    const size_t array_size = 10000;  // 100x100 macierz
    double intervals[array_size*2];
    double intervals2[array_size*2];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i*2] = i * 1.0;
        intervals[i*2+1] = i * 1.0 + 1.0;
        intervals2[i*2] = i * 2.0;
        intervals2[i*2+1] = i * 2.0 + 1.0;
    }
    BatchSwitchVectorMixed<100*100> vector1(intervals);
    BatchSwitchVectorMixed<100*100> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 - vector2;  // Zmieniono dodawanie na odejmowanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_SUB_100_100_BATCH);

static void BM_SUB_100_100_BATCH_OPT(benchmark::State& state) {
    const size_t array_size = 10000;  // 100x100 macierz
    double intervals[array_size*2];
    double intervals2[array_size*2];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i*2] = i * 1.0;
        intervals[i*2+1] = i * 1.0 + 1.0;
        intervals2[i*2] = i * 2.0;
        intervals2[i*2+1] = i * 2.0 + 1.0;
    }
    BatchSwitchVector_Grouped<100*100> vector1(intervals);
    BatchSwitchVector_Grouped<100*100> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 - vector2;  // Zmieniono dodawanie na odejmowanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_SUB_100_100_BATCH_OPT);

static void BM_SUB_100_100_BATCH_AVX(benchmark::State& state) {
    const size_t array_size = 10000;  // 100x100 macierz
    double intervals[array_size*2];
    double intervals2[array_size*2];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i*2] = i * 1.0;
        intervals[i*2+1] = i * 1.0 + 1.0;
        intervals2[i*2] = i * 2.0;
        intervals2[i*2+1] = i * 2.0 + 1.0;
    }
    BatchSwitchVectorAVX_G256<100*100> vector1(intervals);
    BatchSwitchVectorAVX_G256<100*100> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 - vector2;  // Zmieniono dodawanie na odejmowanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_SUB_100_100_BATCH_AVX);

static void BM_SUB_100_100_BATCH_AVX_OPT(benchmark::State& state) {
    const size_t array_size = 10000;  // 100x100 macierz
    double intervals[array_size*2];
    double intervals2[array_size*2];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i*2] = i * 1.0;
        intervals[i*2+1] = i * 1.0 + 1.0;
        intervals2[i*2] = i * 2.0;
        intervals2[i*2+1] = i * 2.0 + 1.0;
    }
    BatchSwitchVectorAVX_Grouped<100*100> vector1(intervals);
    BatchSwitchVectorAVX_Grouped<100*100> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 - vector2;  // Zmieniono dodawanie na odejmowanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_SUB_100_100_BATCH_AVX_OPT);

static void BM_SUB_100_100_BATCH_AVX512_OPT(benchmark::State& state) {
    const size_t array_size = 10000;  // 100x100 macierz
    double intervals[array_size*2];
    double intervals2[array_size*2];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i*2] = i * 1.0;
        intervals[i*2+1] = i * 1.0 + 1.0;
        intervals2[i*2] = i * 2.0;
        intervals2[i*2+1] = i * 2.0 + 1.0;
    }
    BatchSwitchVectorAVX512_Grouped<100*100> vector1(intervals);
    BatchSwitchVectorAVX512_Grouped<100*100> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 - vector2;  // Zmieniono dodawanie na odejmowanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_SUB_100_100_BATCH_AVX512_OPT);

static void BM_SUB_300_300_OPT(benchmark::State& state) {
    const size_t array_size = 90000;  // 300x300 macierz
    opt::Interval intervals[array_size];
    opt::Interval intervals2[array_size];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i] = opt::Interval(i * 1.0, i * 1.0 + 1.0);
        intervals2[i] = opt::Interval(i * 2.0, i * 2.0 + 1.0);
    }
    VectorBasic<opt::Interval, 300*300> vector1(intervals);
    VectorBasic<opt::Interval, 300*300> vector2(intervals2);
    capd::rounding::DoubleRounding l;
    l.roundUp();
    for (auto _ : state) {
        auto a = vector1 - vector2;  // Zmieniono dodawanie na odejmowanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_SUB_300_300_OPT);

static void BM_SUB_300_300_CAPD(benchmark::State& state) {
    const size_t array_size = 90000;  // 300x300 macierz
    capd::intervals::Interval<double> intervals[array_size];
    capd::intervals::Interval<double> intervals2[array_size];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i] = capd::intervals::Interval<double>(i * 1.0, i * 1.0 + 1.0);
        intervals2[i] = capd::intervals::Interval<double>(i * 2.0, i * 2.0 + 1.0);
    }
    VectorBasic<capd::intervals::Interval<double>, 300*300> vector1(intervals);
    VectorBasic<capd::intervals::Interval<double>, 300*300> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 - vector2;  // Zmieniono mnożenie na dodawanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_SUB_300_300_CAPD);

static void BM_SUB_300_300_BATCH(benchmark::State& state) {
    const size_t array_size = 90000;  // 300x300 macierz
    double intervals[array_size*2];
    double intervals2[array_size*2];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i*2] = i * 1.0;
        intervals[i*2+1] = i * 1.0 + 1.0;
        intervals2[i*2] = i * 2.0;
        intervals2[i*2+1] = i * 2.0 + 1.0;
    }
    BatchSwitchVectorMixed<300*300> vector1(intervals);
    BatchSwitchVectorMixed<300*300> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 - vector2;  // Zmieniono dodawanie na odejmowanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_SUB_300_300_BATCH);

static void BM_SUB_300_300_BATCH_OPT(benchmark::State& state) {
    const size_t array_size = 90000;  // 300x300 macierz
    double intervals[array_size*2];
    double intervals2[array_size*2];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i*2] = i * 1.0;
        intervals[i*2+1] = i * 1.0 + 1.0;
        intervals2[i*2] = i * 2.0;
        intervals2[i*2+1] = i * 2.0 + 1.0;
    }
    BatchSwitchVector_Grouped<300*300> vector1(intervals);
    BatchSwitchVector_Grouped<300*300> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 - vector2;  // Zmieniono dodawanie na odejmowanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_SUB_300_300_BATCH_OPT);

static void BM_SUB_300_300_BATCH_AVX(benchmark::State& state) {
    const size_t array_size = 90000;  // 300x300 macierz
    double intervals[array_size*2];
    double intervals2[array_size*2];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i*2] = i * 1.0;
        intervals[i*2+1] = i * 1.0 + 1.0;
        intervals2[i*2] = i * 2.0;
        intervals2[i*2+1] = i * 2.0 + 1.0;
    }
    BatchSwitchVectorAVX_G256<300*300> vector1(intervals);
    BatchSwitchVectorAVX_G256<300*300> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 - vector2;  // Zmieniono dodawanie na odejmowanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_SUB_300_300_BATCH_AVX);

static void BM_SUB_300_300_BATCH_AVX_OPT(benchmark::State& state) {
    const size_t array_size = 90000;  // 300x300 macierz
    double intervals[array_size*2];
    double intervals2[array_size*2];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i*2] = i * 1.0;
        intervals[i*2+1] = i * 1.0 + 1.0;
        intervals2[i*2] = i * 2.0;
        intervals2[i*2+1] = i * 2.0 + 1.0;
    }
    BatchSwitchVectorAVX_Grouped<300*300> vector1(intervals);
    BatchSwitchVectorAVX_Grouped<300*300> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 - vector2;  // Zmieniono dodawanie na odejmowanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_SUB_300_300_BATCH_AVX_OPT);

static void BM_SUB_300_300_BATCH_AVX512_OPT(benchmark::State& state) {
    const size_t array_size = 90000;  // 300x300 macierz
    double intervals[array_size*2];
    double intervals2[array_size*2];

    for (size_t i = 0; i < array_size; ++i) {
        intervals[i*2] = i * 1.0;
        intervals[i*2+1] = i * 1.0 + 1.0;
        intervals2[i*2] = i * 2.0;
        intervals2[i*2+1] = i * 2.0 + 1.0;
    }
    BatchSwitchVectorAVX512_Grouped<300*300> vector1(intervals);
    BatchSwitchVectorAVX512_Grouped<300*300> vector2(intervals2);
    for (auto _ : state) {
        auto a = vector1 - vector2;  // Zmieniono dodawanie na odejmowanie
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_SUB_300_300_BATCH_AVX512_OPT);





BENCHMARK_MAIN();


