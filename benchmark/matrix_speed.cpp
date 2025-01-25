#include<benchmark/benchmark.h>
#include<Interval.hpp>
#include<MatrixBasic.hpp>
#include<BatchSwitchMatrixMixed.hpp>
#include<BatchSwitchMatrix_Grouped.hpp>
#include<BatchSwitchMatrixAVX_Grouped.hpp>
#include<BatchSwitchMatrixAVX512_Grouped.hpp>
#include<capd/intervals/Interval.h>
#include<Utilities.hpp>

// static void BM_ADD_10_10_OPT(benchmark::State& state) {
//     const size_t array_size = 100;  // 10x10 macierz
//     auto intervals = generateArray(array_size,42);
//     auto intervals2 = generateArray(array_size,143);
//     MatrixBasic<opt::Interval, 10, 10> matrix1(intervals);
//     MatrixBasic<opt::Interval, 10, 10> matrix2(intervals2);
//     capd::rounding::DoubleRounding l;
//     l.roundUp();
//     for (auto _ : state) {
//         for (size_t i = 0; i < 100; ++i) {
//             auto a = matrix1 + matrix2;  // Zmieniono mnożenie na dodawanie
//             benchmark::DoNotOptimize(a);
//         }
//     }
// }
// BENCHMARK(BM_ADD_10_10_OPT);
// static void BM_ADD_10_10_CAPD(benchmark::State& state) {
//     const size_t array_size = 100;  // 10x10 macierz
//     auto intervals = generateArray(array_size,42);
//     auto intervals2 = generateArray(array_size,143);
//     MatrixBasic<capd::intervals::Interval<double>, 10, 10> matrix1(intervals);
//     MatrixBasic<capd::intervals::Interval<double>, 10, 10> matrix2(intervals2);

//     for (auto _ : state) {
//         for (size_t i = 0; i < 100; ++i) {
//             auto a = matrix1 + matrix2;  // Zmieniono mnożenie na dodawanie
//             benchmark::DoNotOptimize(a);
//         }
//     }
// }
// BENCHMARK(BM_ADD_10_10_CAPD);

// static void BM_ADD_100_100_OPT(benchmark::State& state) {
//     const size_t array_size = 10000;  // 100x100 macierz
//     auto intervals = generateArray(array_size,42);
//     auto intervals2 = generateArray(array_size,143);
//     MatrixBasic<opt::Interval, 100, 100> matrix1(intervals);
//     MatrixBasic<opt::Interval, 100, 100> matrix2(intervals2);
//     capd::rounding::DoubleRounding l;
//     l.roundUp();
//     for (auto _ : state) {
//         auto a = matrix1 + matrix2;  // Zmieniono mnożenie na dodawanie
//         benchmark::DoNotOptimize(a);
//     }
// }
// BENCHMARK(BM_ADD_100_100_OPT);

// static void BM_ADD_100_100_CAPD(benchmark::State& state) {
//     const size_t array_size = 10000;  // 100x100 macierz
//     auto intervals = generateArray(array_size,42);
//     auto intervals2 = generateArray(array_size,143);
//     MatrixBasic<capd::intervals::Interval<double>, 100, 100> matrix1(intervals);
//     MatrixBasic<capd::intervals::Interval<double>, 100, 100> matrix2(intervals2);

//     for (auto _ : state) {
//         auto a = matrix1 + matrix2;  // Zmieniono mnożenie na dodawanie
//         benchmark::DoNotOptimize(a);
//     }
// }
// BENCHMARK(BM_ADD_100_100_CAPD);

// static void BM_ADD_300_300_OPT(benchmark::State& state) {
//     const size_t array_size = 90000;  // 300x300 macierz
//     auto intervals = generateArray(array_size,42);
//     auto intervals2 = generateArray(array_size,143);
//     MatrixBasic<opt::Interval, 300, 300> matrix1(intervals);
//     MatrixBasic<opt::Interval, 300, 300> matrix2(intervals2);
//     capd::rounding::DoubleRounding l;
//     l.roundUp();
//     for (auto _ : state) {
//         auto a = matrix1 + matrix2;  // Zmieniono mnożenie na dodawanie
//         benchmark::DoNotOptimize(a);
//     }
// }
// BENCHMARK(BM_ADD_300_300_OPT);



// static void BM_ADD_300_300_CAPD(benchmark::State& state) {
//     const size_t array_size = 90000;  // 300x300 macierz
//     auto intervals = generateArray(array_size,42);
//     auto intervals2 = generateArray(array_size,143);
//     MatrixBasic<capd::intervals::Interval<double>, 300, 300> matrix1(intervals);
//     MatrixBasic<capd::intervals::Interval<double>, 300, 300> matrix2(intervals2);

//     for (auto _ : state) {
//         auto a = matrix1 + matrix2;  // Zmieniono mnożenie na dodawanie
//         benchmark::DoNotOptimize(a);
//     }
// }
// BENCHMARK(BM_ADD_300_300_CAPD);


// static void BM_SUB_10_10_OPT(benchmark::State& state) {
//     const size_t array_size = 100; // 10x10 macierz
//     auto intervals = generateArray(array_size,42);
//     auto intervals2 = generateArray(array_size,143);
//     MatrixBasic<opt::Interval, 10, 10> matrix1(intervals);
//     MatrixBasic<opt::Interval, 10, 10> matrix2(intervals2);
//     capd::rounding::DoubleRounding l;
//     l.roundUp();
//     for (auto _ : state) {
//         auto a = matrix1 - matrix2; // Odejmowanie macierzy
//         benchmark::DoNotOptimize(a);
//     }
// }
// BENCHMARK(BM_SUB_10_10_OPT);

// static void BM_SUB_10_10_CAPD(benchmark::State& state) {
//     const size_t array_size = 100; // 10x10 macierz
//     auto intervals = generateArray(array_size,42);
//     auto intervals2 = generateArray(array_size,143);
//     MatrixBasic<capd::intervals::Interval<double>, 10, 10> matrix1(intervals);
//     MatrixBasic<capd::intervals::Interval<double>, 10, 10> matrix2(intervals2);

//     for (auto _ : state) {
//         auto a = matrix1 - matrix2; // Odejmowanie macierzy
//         benchmark::DoNotOptimize(a);
//     }
// }
// BENCHMARK(BM_SUB_10_10_CAPD);

// static void BM_SUB_100_100_OPT(benchmark::State& state) {
//     const size_t array_size = 10000; // 100x100 macierz
//     auto intervals = generateArray(array_size,42);
//     auto intervals2 = generateArray(array_size,143);
//     MatrixBasic<opt::Interval, 100, 100> matrix1(intervals);
//     MatrixBasic<opt::Interval, 100, 100> matrix2(intervals2);
//     capd::rounding::DoubleRounding l;
//     l.roundUp();
//     for (auto _ : state) {
//         auto a = matrix1 - matrix2; // Odejmowanie macierzy
//         benchmark::DoNotOptimize(a);
//     }
// }
// BENCHMARK(BM_SUB_100_100_OPT);

// static void BM_SUB_100_100_CAPD(benchmark::State& state) {
//     const size_t array_size = 10000; // 100x100 macierz
//     auto intervals = generateArray(array_size,42);
//     auto intervals2 = generateArray(array_size,143);
//     MatrixBasic<capd::intervals::Interval<double>, 100, 100> matrix1(intervals);
//     MatrixBasic<capd::intervals::Interval<double>, 100, 100> matrix2(intervals2);

//     for (auto _ : state) {
//         auto a = matrix1 - matrix2; // Odejmowanie macierzy
//         benchmark::DoNotOptimize(a);
//     }
// }
// BENCHMARK(BM_SUB_100_100_CAPD);

// static void BM_SUB_300_300_OPT(benchmark::State& state) {
//     const size_t array_size = 90000; // 300x300 macierz
//     auto intervals = generateArray(array_size,42);
//     auto intervals2 = generateArray(array_size,143);
//     MatrixBasic<opt::Interval, 300, 300> matrix1(intervals);
//     MatrixBasic<opt::Interval, 300, 300> matrix2(intervals2);
//     capd::rounding::DoubleRounding l;
//     l.roundUp();
//     for (auto _ : state) {
//         auto a = matrix1 - matrix2; // Odejmowanie macierzy
//         benchmark::DoNotOptimize(a);
//     }
// }
// BENCHMARK(BM_SUB_300_300_OPT);

// static void BM_SUB_300_300_CAPD(benchmark::State& state) {
//     const size_t array_size = 90000; // 300x300 macierz
//     auto intervals = generateArray(array_size,42);
//     auto intervals2 = generateArray(array_size,143);
//     MatrixBasic<capd::intervals::Interval<double>, 300, 300> matrix1(intervals);
//     MatrixBasic<capd::intervals::Interval<double>, 300, 300> matrix2(intervals2);

//     for (auto _ : state) {
//         auto a = matrix1 - matrix2; // Odejmowanie macierzy
//         benchmark::DoNotOptimize(a);
//     }
// }
// BENCHMARK(BM_SUB_300_300_CAPD);



static void BM_MLT_16_16_UPP(benchmark::State& state) {
    const size_t array_size = 16*16; // 10x10 macierz = 100 elementów
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    
    // Macierze 10x10
    MatrixBasic<opt::Interval, 16, 16> matrix1(intervals);
    MatrixBasic<opt::Interval, 16, 16> matrix2(intervals2);
    capd::rounding::DoubleRounding l;
    l.roundUp();

    for (auto _ : state) {
        auto a = matrix1 * matrix2; // Mnożenie macierzy
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_16_16_UPP);

static void BM_MLT_16_16_CAPD(benchmark::State& state) {
    const size_t array_size = 16*16; // 10x10 macierz = 100 elementów
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    // Macierze 10x10
    MatrixBasic<capd::intervals::Interval<double>, 16, 16> matrix1(intervals);
    MatrixBasic<capd::intervals::Interval<double>, 16, 16> matrix2(intervals2);

    for (auto _ : state) {
        auto a = matrix1 * matrix2; // Mnożenie macierzy
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_16_16_CAPD);

static void BM_MLT_16_16_BATCH_MIX(benchmark::State& state) {
    const size_t array_size = 16*16; // 10x10 macierz = 100 elementów
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    
    // Macierze 10x10
    BatchSwitchMatrixMixed<16, 16> matrix1(intervals);
    BatchSwitchMatrixMixed<16, 16> matrix2(intervals2);
    for (auto _ : state) {
        auto a = matrix1 * matrix2; // Mnożenie macierzy
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_16_16_BATCH_MIX);

static void BM_MLT_16_16_BATCH_GRP(benchmark::State& state) {
    const size_t array_size = 16*16; // 10x10 macierz = 100 elementów
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    
    // Macierze 10x10
    BatchSwitchMatrix_Grouped<16, 16> matrix1(intervals);
    BatchSwitchMatrix_Grouped<16, 16> matrix2(intervals2);
    for (auto _ : state) {
        auto a = matrix1 * matrix2; // Mnożenie macierzy
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_16_16_BATCH_GRP);

static void BM_MLT_16_16_BATCH_GRP_AVX(benchmark::State& state) {
    const size_t array_size = 16*16; // 10x10 macierz = 100 elementów
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    
    // Macierze 10x10
    BatchSwitchMatrixAVX_Grouped<16, 16> matrix1(intervals);
    BatchSwitchMatrixAVX_Grouped<16, 16> matrix2(intervals2);
    for (auto _ : state) {
        auto a = matrix1 * matrix2; // Mnożenie macierzy
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_16_16_BATCH_GRP_AVX);

static void BM_MLT_16_16_BATCH_GRP_AVX512(benchmark::State& state) {
    const size_t array_size = 16*16; // 10x10 macierz = 100 elementów
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    
    // Macierze 10x10
    BatchSwitchMatrixAVX512_Grouped<16, 16> matrix1(intervals);
    BatchSwitchMatrixAVX512_Grouped<16, 16> matrix2(intervals2);
    for (auto _ : state) {
        auto a = matrix1 * matrix2; // Mnożenie macierzy
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_16_16_BATCH_GRP_AVX512);


static void BM_MLT_104_104_UPP(benchmark::State& state) {
    const size_t array_size = 104*104;
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    MatrixBasic<opt::Interval,104,104> matrix1(intervals);
    MatrixBasic<opt::Interval,104,104> matrix2(intervals2);
    capd::rounding::DoubleRounding l;
    l.roundUp();
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_104_104_UPP);

static void BM_MLT_104_104_CAPD(benchmark::State& state) {
    const size_t array_size = 104*104;
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    MatrixBasic<capd::intervals::Interval<double>,104,104> matrix1(intervals);
    MatrixBasic<capd::intervals::Interval<double>,104,104> matrix2(intervals2);

    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_104_104_CAPD);

static void BM_MLT_104_104_BATCH_MIX(benchmark::State& state) {
    const size_t array_size = 104*104;
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    BatchSwitchMatrixMixed<104,104> matrix1(intervals);
    BatchSwitchMatrixMixed<104,104> matrix2(intervals2);
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_104_104_BATCH_MIX);

static void BM_MLT_104_104_BATCH_GRP(benchmark::State& state) {
    const size_t array_size = 104*104;
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    BatchSwitchMatrix_Grouped<104,104> matrix1(intervals);
    BatchSwitchMatrix_Grouped<104,104> matrix2(intervals2);
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_104_104_BATCH_GRP);

static void BM_MLT_104_104_BATCH_GRP_AVX(benchmark::State& state) {
    const size_t array_size = 104*104;
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    BatchSwitchMatrixAVX_Grouped<104,104> matrix1(intervals);
    BatchSwitchMatrixAVX_Grouped<104,104> matrix2(intervals2);
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_104_104_BATCH_GRP_AVX);

static void BM_MLT_104_104_BATCH_GRP_AVX512(benchmark::State& state) {
    const size_t array_size = 104*104;
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    BatchSwitchMatrixAVX512_Grouped<104,104> matrix1(intervals);
    BatchSwitchMatrixAVX512_Grouped<104,104> matrix2(intervals2);
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_104_104_BATCH_GRP_AVX512);

static void BM_MLT_304_304_UPP(benchmark::State& state) {
    const size_t array_size = 304*304; 
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    MatrixBasic<opt::Interval,304,304> matrix1(intervals);
    MatrixBasic<opt::Interval,304,304> matrix2(intervals2);
    capd::rounding::DoubleRounding l;

    l.roundUp();
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_304_304_UPP);

static void BM_MLT_304_304_CAPD(benchmark::State& state) {
    const size_t array_size = 304*304;
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    MatrixBasic<capd::intervals::Interval<double>,304,304> matrix1(intervals);
    MatrixBasic<capd::intervals::Interval<double>,304,304> matrix2(intervals2);

    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_304_304_CAPD);


static void BM_MLT_304_304_BATCH_MIX(benchmark::State& state) {
    const size_t array_size = 304*304; 
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    BatchSwitchMatrixMixed<304,304> matrix1(intervals);
    BatchSwitchMatrixMixed<304,304> matrix2(intervals2);
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_304_304_BATCH_MIX);

static void BM_MLT_304_304_BATCH_GRP(benchmark::State& state) {
    const size_t array_size = 304*304; 
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    BatchSwitchMatrix_Grouped<304,304> matrix1(intervals);
    BatchSwitchMatrix_Grouped<304,304> matrix2(intervals2);
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_304_304_BATCH_GRP);

static void BM_MLT_304_304_BATCH_GRP_AVX(benchmark::State& state) {
    const size_t array_size = 304*304; 
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    BatchSwitchMatrixAVX_Grouped<304,304> matrix1(intervals);
    BatchSwitchMatrixAVX_Grouped<304,304> matrix2(intervals2);
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_304_304_BATCH_GRP_AVX);

static void BM_MLT_304_304_BATCH_GRP_AVX512(benchmark::State& state) {
    const size_t array_size = 304*304; 
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    BatchSwitchMatrixAVX512_Grouped<304,304> matrix1(intervals);
    BatchSwitchMatrixAVX512_Grouped<304,304> matrix2(intervals2);
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_304_304_BATCH_GRP_AVX512);


static void BM_MLT_600_600_BATCH_GRP_AVX(benchmark::State& state) {
    const size_t array_size = 600*600; 
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    BatchSwitchMatrixAVX_Grouped<600,600> matrix1(intervals);
    BatchSwitchMatrixAVX_Grouped<600,600> matrix2(intervals2);
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_600_600_BATCH_GRP_AVX);

static void BM_MLT_600_600_BATCH_GRP_AVX512(benchmark::State& state) {
    const size_t array_size = 600*600; 
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    BatchSwitchMatrixAVX512_Grouped<600,600> matrix1(intervals);
    BatchSwitchMatrixAVX512_Grouped<600,600> matrix2(intervals2);
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_600_600_BATCH_GRP_AVX512);

static void BM_MLT_1200_1200_BATCH_GRP_AVX(benchmark::State& state) {
    const size_t array_size = 1200*1200; 
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    BatchSwitchMatrixAVX_Grouped<1200,1200> matrix1(intervals);
    BatchSwitchMatrixAVX_Grouped<1200,1200> matrix2(intervals2);
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_1200_1200_BATCH_GRP_AVX);

static void BM_MLT_1200_1200_BATCH_GRP_AVX512(benchmark::State& state) {
    const size_t array_size = 1200*1200; 
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    BatchSwitchMatrixAVX512_Grouped<1200,1200> matrix1(intervals);
    BatchSwitchMatrixAVX512_Grouped<1200,1200> matrix2(intervals2);
    for (auto _ : state) {
       auto a = matrix1 * matrix2; 
       benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_MLT_1200_1200_BATCH_GRP_AVX512);

// static void BM_Addition_BareInterval(benchmark::State& state) {
//     const size_t array_size = 1000; // Około 100 MB
//     std::vector<opt::Interval> intervals;
//     std::vector<opt::Interval> intervals2;

//     for (size_t i = 0; i < array_size; ++i) {
//         intervals.emplace_back(i * 1.0, i * 1.0 + 1.0);
//         intervals2.emplace_back(i * 2.0, i * 2.0 + 1.0);
//     }

//     for (auto _ : state) {
//         for (size_t i = 0; i < array_size; ++i) {
//             auto result = intervals[i] + intervals2[i];
//             benchmark::DoNotOptimize(result);
//         }
//     }
// }
// BENCHMARK(BM_Addition_BareInterval);
// static void BM_Addition_BareInterval2(benchmark::State& state) {
//     const size_t array_size = 1000; // Około 100 MB
//     std::vector<capd::intervals::Interval<double>> intervals;
//     std::vector<capd::intervals::Interval<double>> intervals2;

//     for (size_t i = 0; i < array_size; ++i) {
//         intervals.emplace_back(i * 1.0, i * 1.0 + 1.0);
//         intervals2.emplace_back(i * 2.0, i * 2.0 + 1.0);
//     }

//     for (auto _ : state) {
//         for (size_t i = 0; i < array_size; ++i) {
//             auto result = intervals[i] + intervals2[i];
//             benchmark::DoNotOptimize(result);
//         }
//     }
// }
// BENCHMARK(BM_Addition_BareInterval2);


BENCHMARK_MAIN();