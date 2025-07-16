#ifndef MATRIX_BATCH_AVX512_GRP_HPP
#define MATRIX_BATCH_AVX512_GRP_HPP

#include <iostream>
#include <capd/rounding/DoubleRounding.h>
#include <capd/filib/Interval.h>
#include <Utilities.hpp>
#include <IntervalProxy.hpp>
#include <MatrixBasic.hpp>
#include <omp.h>
#include <immintrin.h>

namespace details{
    inline void add_min(const __m512d & a, const __m512d& b,const __m512d& c,const __m512d& d,__m512d& result){
    __m512d mlt1 = _mm512_min_pd(_mm512_mul_pd(a,c),_mm512_mul_pd(a,d));
    __m512d mlt2 = _mm512_min_pd(_mm512_mul_pd(b,c),_mm512_mul_pd(b,d));
    result = _mm512_add_pd(result,_mm512_min_pd(mlt1,mlt2));
    }
    inline void add_max(const __m512d& a, const __m512d& b,const __m512d& c,const __m512d& d,__m512d& result){
        __m512d mlt1 = _mm512_max_pd(_mm512_mul_pd(a,c),_mm512_mul_pd(a,d));
        __m512d mlt2 = _mm512_max_pd(_mm512_mul_pd(b,c),_mm512_mul_pd(b,d));
        result = _mm512_add_pd(result,_mm512_max_pd(mlt1,mlt2));
    }
    inline void split_m512d_to_vectors(__m512d vec, __m512d* result) {
    result[0] = _mm512_permutexvar_pd(_mm512_set1_epi64(0), vec); // Powtórz element 0
    result[1] = _mm512_permutexvar_pd(_mm512_set1_epi64(1), vec); // Powtórz element 1
    result[2] = _mm512_permutexvar_pd(_mm512_set1_epi64(2), vec); // Powtórz element 2
    result[3] = _mm512_permutexvar_pd(_mm512_set1_epi64(3), vec); // Powtórz element 3
    result[4] = _mm512_permutexvar_pd(_mm512_set1_epi64(4), vec); // Powtórz element 4
    result[5] = _mm512_permutexvar_pd(_mm512_set1_epi64(5), vec); // Powtórz element 5
    result[6] = _mm512_permutexvar_pd(_mm512_set1_epi64(6), vec); // Powtórz element 6
    result[7] = _mm512_permutexvar_pd(_mm512_set1_epi64(7), vec); // Powtórz element 7
}
}
template<size_t N, size_t M>
class BatchSwitchMatrixAVX512_Grouped
{
private:
    __m512d* lower = nullptr;
    __m512d* upper = nullptr;
    typedef capd::filib::Interval<double> Interval;
    static constexpr size_t vectors_count_row = (M+7)/8;
    static constexpr size_t vectors_count = vectors_count_row*N;
    static constexpr size_t full_vectors = M/8;
    static constexpr size_t rest = M % 8;
    static constexpr bool enable_parallel = (N>=100) &&  (M>=100);

    template<size_t N1,size_t M1>
    friend class BatchSwitchMatrixAVX512_Grouped;

    BatchSwitchMatrixAVX512_Grouped(bool allocateOnly)
    {
        if(allocateOnly){    
            lower = new alignas(64) __m512d[vectors_count];
            upper = new alignas(64) __m512d[vectors_count];
        }
        else{    
            lower = nullptr;
            upper = nullptr;
        }
    }

    struct Index{
        size_t ind;
        size_t poz;
    };
    struct Accessor {
    __m512d* lower, * upper;

    Interval get(Index index) const {
        __m512d vec_lower = lower[index.ind];
        __m512d vec_upper = upper[index.ind];

        double low_val, upp_val;
        switch (index.poz) {
        case 7:
            low_val = _mm256_cvtsd_f64(_mm256_permutex_pd(_mm512_extractf64x4_pd(vec_lower,1),0b00000011));
            upp_val = _mm256_cvtsd_f64(_mm256_permutex_pd(_mm512_extractf64x4_pd(vec_upper,1),0b00000011));
            break;
        case 6:
            low_val = _mm256_cvtsd_f64(_mm256_permutex_pd(_mm512_extractf64x4_pd(vec_lower,1),0b00000010));
            upp_val = _mm256_cvtsd_f64(_mm256_permutex_pd(_mm512_extractf64x4_pd(vec_upper,1),0b00000010));
            break;
        case 5:
            low_val = _mm256_cvtsd_f64(_mm256_permutex_pd(_mm512_extractf64x4_pd(vec_lower,1),0b00000001));
            upp_val = _mm256_cvtsd_f64(_mm256_permutex_pd(_mm512_extractf64x4_pd(vec_upper,1),0b00000001));
            break;
        case 4:
            low_val = _mm256_cvtsd_f64(_mm512_extractf64x4_pd(vec_lower,1));
            upp_val = _mm256_cvtsd_f64(_mm512_extractf64x4_pd(vec_upper,1));
            break;
        case 3:
            low_val = _mm512_cvtsd_f64(_mm512_permutex_pd(vec_lower,0b00000011));
            upp_val = _mm512_cvtsd_f64(_mm512_permutex_pd(vec_upper,0b00000011));
            break;
        case 2:
            low_val = _mm512_cvtsd_f64(_mm512_permutex_pd(vec_lower,0b00000010));
            upp_val = _mm512_cvtsd_f64(_mm512_permutex_pd(vec_upper,0b00000010));
            break;
        case 1:
            low_val = _mm512_cvtsd_f64(_mm512_permute_pd(vec_lower,0b00000001));
            upp_val = _mm512_cvtsd_f64(_mm512_permute_pd(vec_upper,0b00000001));
            break;
        default:
            low_val = _mm512_cvtsd_f64(vec_lower);
            upp_val = _mm512_cvtsd_f64(vec_upper);
            break;
        }

        return Interval(low_val, upp_val);
    }

    void set(Index index, const Interval& interval) {
        __m512d low = _mm512_set1_pd(interval.leftBound());
        __m512d upp = _mm512_set1_pd(interval.rightBound());
        __m512d vec_lower = lower[index.ind];
        __m512d vec_upper = upper[index.ind];

        switch (index.poz) {
        case 7:
            vec_lower = _mm512_mask_blend_pd(0b10000000, vec_lower, low);
            vec_upper = _mm512_mask_blend_pd(0b10000000, vec_upper, upp);
            break;
        case 6:
            vec_lower = _mm512_mask_blend_pd(0b01000000, vec_lower, low);
            vec_upper = _mm512_mask_blend_pd(0b01000000, vec_upper, upp);
            break;
        case 5:
            vec_lower = _mm512_mask_blend_pd(0b00100000, vec_lower, low);
            vec_upper = _mm512_mask_blend_pd(0b00100000, vec_upper, upp);
            break;
        case 4:
            vec_lower = _mm512_mask_blend_pd(0b00010000, vec_lower, low);
            vec_upper = _mm512_mask_blend_pd(0b00010000, vec_upper, upp);
            break;
        case 3:
            vec_lower = _mm512_mask_blend_pd(0b00001000, vec_lower, low);
            vec_upper = _mm512_mask_blend_pd(0b00001000, vec_upper, upp);
            break;
        case 2:
            vec_lower = _mm512_mask_blend_pd(0b00000100, vec_lower, low);
            vec_upper = _mm512_mask_blend_pd(0b00000100, vec_upper, upp);
            break;
        case 1:
            vec_lower = _mm512_mask_blend_pd(0b00000010, vec_lower, low);
            vec_upper = _mm512_mask_blend_pd(0b00000010, vec_upper, upp);
            break;
        default:
            vec_lower = _mm512_mask_blend_pd(0b00000001, vec_lower, low);
            vec_upper = _mm512_mask_blend_pd(0b00000001, vec_upper, upp);
            break;
        }

        lower[index.ind] = vec_lower;
        upper[index.ind] = vec_upper;
    }
};
public:
    static constexpr size_t rows = N;
    static constexpr size_t cols = M;
    BatchSwitchMatrixAVX512_Grouped(){
    if constexpr (enable_parallel){
        lower = new alignas(64) __m512d[vectors_count];
        upper = new alignas(64) __m512d[vectors_count];
        #pragma omp parallel for
        for(size_t i = 0; i < vectors_count; i++){
            lower[i] = _mm512_setzero_pd();
            upper[i] = _mm512_setzero_pd();
        }
    }
    else{
        lower = new alignas(64) __m512d[vectors_count]();
        upper = new alignas(64) __m512d[vectors_count]();
    }
}

    BatchSwitchMatrixAVX512_Grouped(const BatchSwitchMatrixAVX512_Grouped& cpy){
    lower = new alignas(64) __m512d[vectors_count];
    upper = new alignas(64) __m512d[vectors_count];
    
    if constexpr (enable_parallel){
        #pragma omp parallel for
        for(size_t i = 0; i < vectors_count; i++){
            lower[i] = cpy.lower[i];
            upper[i] = cpy.upper[i];
        }
    }
    else{
        for(size_t i = 0; i < vectors_count; i++) {
            lower[i] = cpy.lower[i];
            upper[i] = cpy.upper[i];
        }
    }
}

    BatchSwitchMatrixAVX512_Grouped(BatchSwitchMatrixAVX512_Grouped&& cpy) noexcept {
        lower = cpy.lower;
        upper = cpy.upper;
        cpy.lower = nullptr;
        cpy.upper = nullptr;
    }

    BatchSwitchMatrixAVX512_Grouped(const double(&array)[2 * N * M]){
    lower = new alignas(64) __m512d[vectors_count];
    upper = new alignas(64) __m512d[vectors_count];

    if constexpr (enable_parallel){
        #pragma omp parallel for
        for (size_t row = 0; row < N; ++row) {
            size_t offset = row * M * 2;
            
            for (size_t i = 0; i < full_vectors * 16; i += 16) {
                __m512d a = _mm512_set_pd(
                    array[offset + i + 14],
                    array[offset + i + 12],
                    array[offset + i + 10],
                    array[offset + i + 8],
                    array[offset + i + 6],
                    array[offset + i + 4],
                    array[offset + i + 2],
                    array[offset + i + 0]
                );
                __m512d b = _mm512_set_pd(
                    array[offset + i + 15],
                    array[offset + i + 13],
                    array[offset + i + 11],
                    array[offset + i + 9],
                    array[offset + i + 7],
                    array[offset + i + 5],
                    array[offset + i + 3],
                    array[offset + i + 1]
                );
                size_t vector_index = row * vectors_count_row + i / 16;
                lower[vector_index] = a;
                upper[vector_index] = b;
            }

            if (rest > 0) {
                double temp_lower[8] = {};
                double temp_upper[8] = {};
                for (size_t i = 0; i < rest; ++i) {
                    temp_lower[i] = array[offset + (2 * M - 2 * rest) + 2 * i];
                    temp_upper[i] = array[offset + (2 * M - 2 * rest) + 2 * i + 1];
                }
                size_t vector_index = row * vectors_count_row + full_vectors;
                lower[vector_index] = _mm512_load_pd(temp_lower);
                upper[vector_index] = _mm512_load_pd(temp_upper);
            }
        }
    }
    else{
        for (size_t row = 0; row < N; ++row) {
            size_t offset = row * M * 2;
            
            for (size_t i = 0; i < full_vectors * 16; i += 16) {
                __m512d a = _mm512_set_pd(
                    array[offset + i + 14],
                    array[offset + i + 12],
                    array[offset + i + 10],
                    array[offset + i + 8],
                    array[offset + i + 6],
                    array[offset + i + 4],
                    array[offset + i + 2],
                    array[offset + i + 0]
                );
                __m512d b = _mm512_set_pd(
                    array[offset + i + 15],
                    array[offset + i + 13],
                    array[offset + i + 11],
                    array[offset + i + 9],
                    array[offset + i + 7],
                    array[offset + i + 5],
                    array[offset + i + 3],
                    array[offset + i + 1]
                );
                size_t vector_index = row * vectors_count_row + i / 16;
                lower[vector_index] = a;
                upper[vector_index] = b;
            }

            if (rest > 0) {
                double temp_lower[8] = {};
                double temp_upper[8] = {};
                for (size_t i = 0; i < rest; ++i) {
                    temp_lower[i] = array[offset + (2 * M - 2 * rest) + 2 * i];
                    temp_upper[i] = array[offset + (2 * M - 2 * rest) + 2 * i + 1];
                }
                size_t vector_index = row * vectors_count_row + full_vectors;
                lower[vector_index] = _mm512_load_pd(temp_lower);
                upper[vector_index] = _mm512_load_pd(temp_upper);
            }
        }
    }
}

   BatchSwitchMatrixAVX512_Grouped(const std::vector<double>& array) {
    if (array.size() != 2 * N * M) {
        throw std::invalid_argument("Rozmiar wektora musi wynosić dokładnie 2 * N * M.");
    }
    lower = new alignas(64) __m512d[vectors_count];
    upper = new alignas(64) __m512d[vectors_count];

    if constexpr (enable_parallel){
        #pragma omp parallel for
        for (size_t row = 0; row < N; ++row) {
            size_t offset = row * M * 2;
            
            for (size_t i = 0; i < full_vectors * 16; i += 16) {
                __m512d a = _mm512_set_pd(
                    array[offset + i + 14],
                    array[offset + i + 12],
                    array[offset + i + 10],
                    array[offset + i + 8],
                    array[offset + i + 6],
                    array[offset + i + 4],
                    array[offset + i + 2],
                    array[offset + i + 0]
                );
                __m512d b = _mm512_set_pd(
                    array[offset + i + 15],
                    array[offset + i + 13],
                    array[offset + i + 11],
                    array[offset + i + 9],
                    array[offset + i + 7],
                    array[offset + i + 5],
                    array[offset + i + 3],
                    array[offset + i + 1]
                );
                size_t vector_index = row * vectors_count_row + i / 16;
                lower[vector_index] = a;
                upper[vector_index] = b;
            }

            if (rest > 0) {
                double temp_lower[8] = {};
                double temp_upper[8] = {};
                for (size_t i = 0; i < rest; ++i) {
                    temp_lower[i] = array[offset + 2 * M - 2 * rest + 2 * i];
                    temp_upper[i] = array[offset + 2 * M - 2 * rest + 2 * i + 1];
                }
                size_t vector_index = row * vectors_count_row + full_vectors;
                lower[vector_index] = _mm512_load_pd(temp_lower);
                upper[vector_index] = _mm512_load_pd(temp_upper);
            }
        }
    }
    else{
        for (size_t row = 0; row < N; ++row) {
            size_t offset = row * M * 2;
            
            for (size_t i = 0; i < full_vectors * 16; i += 16) {
                __m512d a = _mm512_set_pd(
                    array[offset + i + 14],
                    array[offset + i + 12],
                    array[offset + i + 10],
                    array[offset + i + 8],
                    array[offset + i + 6],
                    array[offset + i + 4],
                    array[offset + i + 2],
                    array[offset + i + 0]
                );
                __m512d b = _mm512_set_pd(
                    array[offset + i + 15],
                    array[offset + i + 13],
                    array[offset + i + 11],
                    array[offset + i + 9],
                    array[offset + i + 7],
                    array[offset + i + 5],
                    array[offset + i + 3],
                    array[offset + i + 1]
                );
                size_t vector_index = row * vectors_count_row + i / 16;
                lower[vector_index] = a;
                upper[vector_index] = b;
            }

            if (rest > 0) {
                double temp_lower[8] = {};
                double temp_upper[8] = {};
                for (size_t i = 0; i < rest; ++i) {
                    temp_lower[i] = array[offset + 2 * M - 2 * rest + 2 * i];
                    temp_upper[i] = array[offset + 2 * M - 2 * rest + 2 * i + 1];
                }
                size_t vector_index = row * vectors_count_row + full_vectors;
                lower[vector_index] = _mm512_load_pd(temp_lower);
                upper[vector_index] = _mm512_load_pd(temp_upper);
            }
        }
    }
}

    BatchSwitchMatrixAVX512_Grouped& operator=(const BatchSwitchMatrixAVX512_Grouped& fst) {
        if (this != &fst) {
            if (fst.lower == nullptr) {
                delete[] lower;
                lower = nullptr;
                delete[] upper;
                upper = nullptr;
            } else {
                if (lower == nullptr) {
                    lower = new alignas(64) __m512d[vectors_count];
                    upper = new alignas(64) __m512d[vectors_count];
                }
                
                if constexpr (enable_parallel) {
                    #pragma omp parallel for
                    for (size_t i = 0; i < vectors_count; ++i) {
                        lower[i] = fst.lower[i];
                        upper[i] = fst.upper[i];
                    }
                }
                else {
                    for (size_t i = 0; i < vectors_count; ++i) {
                        lower[i] = fst.lower[i];
                        upper[i] = fst.upper[i];
                    } 
                }
            }
        }
        return *this;
    }

    BatchSwitchMatrixAVX512_Grouped& operator=(BatchSwitchMatrixAVX512_Grouped&& fst) noexcept {
        if (this != &fst) {
            delete[] lower;
            delete[] upper;
            lower = fst.lower;
            upper = fst.upper;
            fst.lower = nullptr;
            fst.upper = nullptr;
        }
        return *this;
    }

    template<size_t N1, size_t M1>
    friend BatchSwitchMatrixAVX512_Grouped<M1, N1> reorg(const BatchSwitchMatrixAVX512_Grouped<N1,M1>& fst);


    friend BatchSwitchMatrixAVX512_Grouped operator+(const BatchSwitchMatrixAVX512_Grouped& fst, const BatchSwitchMatrixAVX512_Grouped& scd) {
    BatchSwitchMatrixAVX512_Grouped result(true); 

    if constexpr(enable_parallel)
    {
        #pragma omp parallel
        { 
            capd::rounding::DoubleRounding::roundDown();
            #pragma omp for schedule(static) nowait
            for(size_t i = 0; i < result.vectors_count; i++) {
                result.lower[i] = _mm512_add_pd(fst.lower[i], scd.lower[i]);
            }

            capd::rounding::DoubleRounding::roundUp();
            #pragma omp for schedule(static)
            for(size_t i = 0; i < result.vectors_count; i++) {
                result.upper[i] = _mm512_add_pd(fst.upper[i], scd.upper[i]);
            }
        }
    }
    else
    {
        capd::rounding::DoubleRounding::roundDown();
        for(size_t i = 0; i < result.vectors_count; i++) {
            result.lower[i] = _mm512_add_pd(fst.lower[i], scd.lower[i]);
        }

        capd::rounding::DoubleRounding::roundUp();
        for(size_t i = 0; i < result.vectors_count; i++) {
            result.upper[i] = _mm512_add_pd(fst.upper[i], scd.upper[i]);
        }
    }
    return result;
}

    friend BatchSwitchMatrixAVX512_Grouped operator-(const BatchSwitchMatrixAVX512_Grouped& fst, const BatchSwitchMatrixAVX512_Grouped& scd) {
    BatchSwitchMatrixAVX512_Grouped result(true);
    if constexpr(enable_parallel)
    {
        #pragma omp parallel
        { 
            capd::rounding::DoubleRounding::roundDown();
            #pragma omp for schedule(static) nowait
            for(size_t i = 0; i < result.vectors_count; i++) {
                // Dolna granica wyniku: fst.lower - scd.upper
                result.lower[i] = _mm512_sub_pd(fst.lower[i], scd.upper[i]);
            }

            capd::rounding::DoubleRounding::roundUp();
            #pragma omp for schedule(static)
            for(size_t i = 0; i < result.vectors_count; i++) {
                // Górna granica wyniku: fst.upper - scd.lower
                result.upper[i] = _mm512_sub_pd(fst.upper[i], scd.lower[i]);
            }
        }
    }
    else
    {
        capd::rounding::DoubleRounding::roundDown();
        for(size_t i = 0; i < result.vectors_count; i++) {
            result.lower[i] = _mm512_sub_pd(fst.lower[i], scd.upper[i]);
        }

        capd::rounding::DoubleRounding::roundUp();
        for(size_t i = 0; i < result.vectors_count; i++) {
            result.upper[i] = _mm512_sub_pd(fst.upper[i], scd.lower[i]);
        }
    }
    return result;
}

    //mnożenie z reorganizacją


    // template <size_t P>
    // BatchSwitchMatrixAVX512_Grouped<N, P> operator*(const BatchSwitchMatrixAVX512_Grouped<M, P>& fst) {
    //     BatchSwitchMatrixAVX512_Grouped<N, P> result(true);
    //     auto pom = reorg(fst);
    //     alignas(64) double sum[8];

    //     __m512d * one_lower= fst.lower;
    //     __m512d * two_lower= fst.lower + vectors_count/8;
    //     __m512d * three_lower= fst.lower + vectors_count/4;
    //     __m512d * four_lower= fst.lower + vectors_count*3/8;
    //     __m512d * five_lower= fst.lower + vectors_count/2;
    //     __m512d * six_lower= fst.lower + vectors_count*5/8;
    //     __m512d * seven_lower= fst.lower + vectors_count*3/4;
    //     __m512d * eight_lower= fst.lower + vectors_count*7/8;

    //     __m512d * one_upper= fst.upper;
    //     __m512d * two_upper= fst.upper + vectors_count/4;
    //     __m512d * three_upper= fst.upper + vectors_count/2;
    //     __m512d * four_upper= fst.upper + vectors_count*3/4;
    //     __m512d * five_upper= fst.upper + vectors_count/2;
    //     __m512d * six_upper= fst.upper + vectors_count*5/8;
    //     __m512d * seven_upper= fst.upper + vectors_count*3/4;
    //     __m512d * eight_upper= fst.upper + vectors_count*7/8;
    //     // Round down pass
    //     //(i,k), (j,k)
    //     capd::rounding::DoubleRounding::roundDown();
    //     for (size_t i = 0; i < N; ++i) {
    //         for (size_t j = 0; j < result.vectors_count_row; ++j) {
    //             __m512d sum1 = _mm512_setzero_pd(), sum2 = _mm512_setzero_pd(),sum3 = _mm512_setzero_pd(),sum4 = _mm512_setzero_pd(),
    //                     sum5 = _mm512_setzero_pd(), sum6 = _mm512_setzero_pd(),sum7 = _mm512_setzero_pd(),sum8 = _mm512_setzero_pd();
    //             for (size_t k = 0; k < vectors_count_row; ++k) {
    //                 __m512d low = lower[i*vectors_count_row+k];
    //                 __m512d up = upper[i*vectors_count_row+k];
    //                 details::add_min(low,up,one_lower[j*vectors_count_row+k],one_upper[j*vectors_count_row+k],sum1);
    //                 details::add_min(low,up,two_lower[j*vectors_count_row+k],two_upper[j*vectors_count_row+k],sum2);
    //                 details::add_min(low,up,three_lower[j*vectors_count_row+k],three_upper[j*vectors_count_row+k],sum3);
    //                 details::add_min(low,up,four_lower[j*vectors_count_row+k],four_upper[j*vectors_count_row+k],sum4);

    //                 details::add_min(low,up,five_lower[j*vectors_count_row+k],five_upper[j*vectors_count_row+k],sum5);
    //                 details::add_min(low,up,six_lower[j*vectors_count_row+k],six_upper[j*vectors_count_row+k],sum6);
    //                 details::add_min(low,up,seven_lower[j*vectors_count_row+k],seven_upper[j*vectors_count_row+k],sum7);
    //                 details::add_min(low,up,eight_lower[j*vectors_count_row+k],eight_upper[j*vectors_count_row+k],sum8);
    //             }
    //             sum[0]  = _mm512_reduce_add_pd(sum1);
    //             sum[1]  = _mm512_reduce_add_pd(sum2);
    //             sum[2]  = _mm512_reduce_add_pd(sum3);
    //             sum[3]  = _mm512_reduce_add_pd(sum4);
    //             sum[4]  = _mm512_reduce_add_pd(sum5);
    //             sum[5]  = _mm512_reduce_add_pd(sum6);
    //             sum[6]  = _mm512_reduce_add_pd(sum7);
    //             sum[7]  = _mm512_reduce_add_pd(sum8);

    //             result.lower[i * result.vectors_count_row + j] = _mm512_load_pd(&sum[0]);
    //         }
            
    //     }

    //     // Round up pass
    //     capd::rounding::DoubleRounding::roundUp();
    //     for (size_t i = 0; i < N; ++i) {
    //         for (size_t j = 0; j < result.vectors_count_row; ++j) {
    //             __m512d sum1 = _mm512_setzero_pd(), sum2 = _mm512_setzero_pd(),sum3 = _mm512_setzero_pd(),sum4 = _mm512_setzero_pd(),
    //                     sum5 = _mm512_setzero_pd(), sum6 = _mm512_setzero_pd(),sum7 = _mm512_setzero_pd(),sum8 = _mm512_setzero_pd();
    //             for (size_t k = 0; k < vectors_count_row; ++k) {
    //                 __m512d low = lower[i*vectors_count_row+k];
    //                 __m512d up = upper[i*vectors_count_row+k];
    //                 details::add_max(low,up,one_lower[j*vectors_count_row+k],one_upper[j*vectors_count_row+k],sum1);
    //                 details::add_max(low,up,two_lower[j*vectors_count_row+k],two_upper[j*vectors_count_row+k],sum2);
    //                 details::add_max(low,up,three_lower[j*vectors_count_row+k],three_upper[j*vectors_count_row+k],sum3);
    //                 details::add_max(low,up,four_lower[j*vectors_count_row+k],four_upper[j*vectors_count_row+k],sum4);

    //                 details::add_max(low,up,five_lower[j*vectors_count_row+k],five_upper[j*vectors_count_row+k],sum5);
    //                 details::add_max(low,up,six_lower[j*vectors_count_row+k],six_upper[j*vectors_count_row+k],sum6);
    //                 details::add_max(low,up,seven_lower[j*vectors_count_row+k],seven_upper[j*vectors_count_row+k],sum7);
    //                 details::add_max(low,up,eight_lower[j*vectors_count_row+k],eight_upper[j*vectors_count_row+k],sum8);
    //             }
    //             sum[0]  = _mm512_reduce_add_pd(sum1);
    //             sum[1]  = _mm512_reduce_add_pd(sum2);
    //             sum[2]  = _mm512_reduce_add_pd(sum3);
    //             sum[3]  = _mm512_reduce_add_pd(sum4);
    //             sum[4]  = _mm512_reduce_add_pd(sum5);
    //             sum[5]  = _mm512_reduce_add_pd(sum6);
    //             sum[6]  = _mm512_reduce_add_pd(sum7);
    //             sum[7]  = _mm512_reduce_add_pd(sum8);
    //             result.upper[i * result.vectors_count_row + j] = _mm512_load_pd(&sum[0]);
    //         }
            
    //     }


    //     return result;
    // }


    //standard
    template <size_t P>
BatchSwitchMatrixAVX512_Grouped<N, P> operator*(const BatchSwitchMatrixAVX512_Grouped<M, P>& fst) {
    BatchSwitchMatrixAVX512_Grouped<N, P> result{};
    static constexpr bool enable_parallel_mlt = (N >= 30) && (M >= 30) && (P >=30);
    if constexpr(enable_parallel_mlt){
        #pragma omp parallel
        {
            capd::rounding::DoubleRounding::roundDown();
            #pragma omp for
            for (size_t i = 0; i < N; ++i) {
                alignas(64) __m512d lows[8];
                alignas(64) __m512d upps[8];
                for (size_t k = 0; k < full_vectors; ++k) {
                    __m512d low = lower[i*vectors_count_row+k];
                    __m512d up = upper[i*vectors_count_row+k];
                    details::split_m512d_to_vectors(low,lows);
                    details::split_m512d_to_vectors(up,upps);
                    for (size_t j = 0; j < result.vectors_count_row; ++j) {
                        __m512d& wyn = result.lower[i * result.vectors_count_row + j];
                        details::add_min(lows[0],upps[0],fst.lower[k*8 * result.vectors_count_row + j],fst.upper[k*8 * result.vectors_count_row + j],wyn);
                        details::add_min(lows[1],upps[1],fst.lower[(k*8 +1)* result.vectors_count_row + j],fst.upper[(k*8 +1) * result.vectors_count_row + j],wyn);
                        details::add_min(lows[2],upps[2],fst.lower[(k*8 +2) * result.vectors_count_row + j],fst.upper[(k*8 +2) * result.vectors_count_row + j],wyn);
                        details::add_min(lows[3],upps[3],fst.lower[(k*8 +3) * result.vectors_count_row + j],fst.upper[(k*8 +3) * result.vectors_count_row + j],wyn);
                        details::add_min(lows[4],upps[4],fst.lower[(k*8 +4) * result.vectors_count_row + j],fst.upper[(k*8 +4) * result.vectors_count_row + j],wyn);
                        details::add_min(lows[5],upps[5],fst.lower[(k*8 +5) * result.vectors_count_row + j],fst.upper[(k*8 +5) * result.vectors_count_row + j],wyn);
                        details::add_min(lows[6],upps[6],fst.lower[(k*8 +6) * result.vectors_count_row + j],fst.upper[(k*8 +6) * result.vectors_count_row + j],wyn);
                        details::add_min(lows[7],upps[7],fst.lower[(k*8 +7) * result.vectors_count_row + j],fst.upper[(k*8 +7) * result.vectors_count_row + j],wyn);
                    }
                }
                if constexpr(rest >= 1){
                    __m512d low = lower[i*vectors_count_row+full_vectors];
                    __m512d up = upper[i*vectors_count_row+full_vectors];
                    details::split_m512d_to_vectors(low,lows);
                    details::split_m512d_to_vectors(up,upps);
                    for (size_t j = 0; j < result.vectors_count_row; ++j) {
                        __m512d& wyn = result.lower[i * result.vectors_count_row + j];
                        if constexpr(rest >= 1)
                            details::add_min(lows[0],upps[0],fst.lower[(full_vectors*8 +0)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +0)* result.vectors_count_row + j],wyn);
                        if constexpr(rest >= 2)
                            details::add_min(lows[1],upps[1],fst.lower[(full_vectors*8 +1)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +1)* result.vectors_count_row + j],wyn);
                        if constexpr(rest >= 3)
                            details::add_min(lows[2],upps[2],fst.lower[(full_vectors*8 +2)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +2)* result.vectors_count_row + j],wyn);
                        if constexpr(rest >= 4)
                            details::add_min(lows[3],upps[3],fst.lower[(full_vectors*8 +3)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +3)* result.vectors_count_row + j],wyn);
                        if constexpr(rest >= 5)
                            details::add_min(lows[4],upps[4],fst.lower[(full_vectors*8 +4)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +4)* result.vectors_count_row + j],wyn);
                        if constexpr(rest >= 6)
                            details::add_min(lows[5],upps[5],fst.lower[(full_vectors*8 +5)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +5)* result.vectors_count_row + j],wyn);
                        if constexpr(rest >= 7)
                            details::add_min(lows[6],upps[6],fst.lower[(full_vectors*8 +6)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +6)* result.vectors_count_row + j],wyn);
                    }
                }
            }

            //#pragma omp barrier

            capd::rounding::DoubleRounding::roundUp();
            #pragma omp for
            for (size_t i = 0; i < N; ++i) {
                alignas(64) __m512d lows[8];
                alignas(64) __m512d upps[8];
                for (size_t k = 0; k < full_vectors; ++k) {
                    __m512d low = lower[i*vectors_count_row+k];
                    __m512d up = upper[i*vectors_count_row+k];
                    details::split_m512d_to_vectors(low,lows);
                    details::split_m512d_to_vectors(up,upps);
                    for (size_t j = 0; j < result.vectors_count_row; ++j) {
                        __m512d& wyn = result.upper[i * result.vectors_count_row + j];
                        details::add_max(lows[0],upps[0],fst.lower[k*8 * result.vectors_count_row + j],fst.upper[k*8 * result.vectors_count_row + j],wyn);
                        details::add_max(lows[1],upps[1],fst.lower[(k*8 +1)* result.vectors_count_row + j],fst.upper[(k*8 +1) * result.vectors_count_row + j],wyn);
                        details::add_max(lows[2],upps[2],fst.lower[(k*8 +2) * result.vectors_count_row + j],fst.upper[(k*8 +2) * result.vectors_count_row + j],wyn);
                        details::add_max(lows[3],upps[3],fst.lower[(k*8 +3) * result.vectors_count_row + j],fst.upper[(k*8 +3) * result.vectors_count_row + j],wyn);
                        details::add_max(lows[4],upps[4],fst.lower[(k*8 +4) * result.vectors_count_row + j],fst.upper[(k*8 +4) * result.vectors_count_row + j],wyn);
                        details::add_max(lows[5],upps[5],fst.lower[(k*8 +5) * result.vectors_count_row + j],fst.upper[(k*8 +5) * result.vectors_count_row + j],wyn);
                        details::add_max(lows[6],upps[6],fst.lower[(k*8 +6) * result.vectors_count_row + j],fst.upper[(k*8 +6) * result.vectors_count_row + j],wyn);
                        details::add_max(lows[7],upps[7],fst.lower[(k*8 +7) * result.vectors_count_row + j],fst.upper[(k*8 +7) * result.vectors_count_row + j],wyn);
                    }
                }
                if constexpr(rest >= 1){
                    __m512d low = lower[i*vectors_count_row+full_vectors];
                    __m512d up = upper[i*vectors_count_row+full_vectors];
                    details::split_m512d_to_vectors(low,lows);
                    details::split_m512d_to_vectors(up,upps);
                    for (size_t j = 0; j < result.vectors_count_row; ++j) {
                        __m512d& wyn = result.upper[i * result.vectors_count_row + j];
                        if constexpr(rest >= 1)
                            details::add_max(lows[0],upps[0],fst.lower[(full_vectors*8 +0)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +0)* result.vectors_count_row + j],wyn);
                        if constexpr(rest >= 2)
                            details::add_max(lows[1],upps[1],fst.lower[(full_vectors*8 +1)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +1)* result.vectors_count_row + j],wyn);
                        if constexpr(rest >= 3)
                            details::add_max(lows[2],upps[2],fst.lower[(full_vectors*8 +2)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +2)* result.vectors_count_row + j],wyn);
                        if constexpr(rest >= 4)
                            details::add_max(lows[3],upps[3],fst.lower[(full_vectors*8 +3)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +3)* result.vectors_count_row + j],wyn);
                        if constexpr(rest >= 5)
                            details::add_max(lows[4],upps[4],fst.lower[(full_vectors*8 +4)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +4)* result.vectors_count_row + j],wyn);
                        if constexpr(rest >= 6)
                            details::add_max(lows[5],upps[5],fst.lower[(full_vectors*8 +5)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +5)* result.vectors_count_row + j],wyn);
                        if constexpr(rest >= 7)
                            details::add_max(lows[6],upps[6],fst.lower[(full_vectors*8 +6)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +6)* result.vectors_count_row + j],wyn);
                    }
                }
            }
        }
    }
    else{
        alignas(64) __m512d lows[8];
        alignas(64) __m512d upps[8];
        capd::rounding::DoubleRounding::roundDown();
        for (size_t i = 0; i < N; ++i) {
            for (size_t k = 0; k < full_vectors; ++k) {
                __m512d low = lower[i*vectors_count_row+k];
                __m512d up = upper[i*vectors_count_row+k];
                details::split_m512d_to_vectors(low,lows);
                details::split_m512d_to_vectors(up,upps);
                for (size_t j = 0; j < result.vectors_count_row; ++j) {
                    __m512d& wyn = result.lower[i * result.vectors_count_row + j];
                    details::add_min(lows[0],upps[0],fst.lower[k*8 * result.vectors_count_row + j],fst.upper[k*8 * result.vectors_count_row + j],wyn);
                    details::add_min(lows[1],upps[1],fst.lower[(k*8 +1)* result.vectors_count_row + j],fst.upper[(k*8 +1) * result.vectors_count_row + j],wyn);
                    details::add_min(lows[2],upps[2],fst.lower[(k*8 +2) * result.vectors_count_row + j],fst.upper[(k*8 +2) * result.vectors_count_row + j],wyn);
                    details::add_min(lows[3],upps[3],fst.lower[(k*8 +3) * result.vectors_count_row + j],fst.upper[(k*8 +3) * result.vectors_count_row + j],wyn);
                    details::add_min(lows[4],upps[4],fst.lower[(k*8 +4) * result.vectors_count_row + j],fst.upper[(k*8 +4) * result.vectors_count_row + j],wyn);
                    details::add_min(lows[5],upps[5],fst.lower[(k*8 +5) * result.vectors_count_row + j],fst.upper[(k*8 +5) * result.vectors_count_row + j],wyn);
                    details::add_min(lows[6],upps[6],fst.lower[(k*8 +6) * result.vectors_count_row + j],fst.upper[(k*8 +6) * result.vectors_count_row + j],wyn);
                    details::add_min(lows[7],upps[7],fst.lower[(k*8 +7) * result.vectors_count_row + j],fst.upper[(k*8 +7) * result.vectors_count_row + j],wyn);
                }
            }
            if constexpr(rest >= 1){
                __m512d low = lower[i*vectors_count_row+full_vectors];
                __m512d up = upper[i*vectors_count_row+full_vectors];
                details::split_m512d_to_vectors(low,lows);
                details::split_m512d_to_vectors(up,upps);
                for (size_t j = 0; j < result.vectors_count_row; ++j) {
                    __m512d& wyn = result.lower[i * result.vectors_count_row + j];
                    if constexpr(rest >= 1)
                        details::add_min(lows[0],upps[0],fst.lower[(full_vectors*8 +0)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +0)* result.vectors_count_row + j],wyn);
                    if constexpr(rest >= 2)
                        details::add_min(lows[1],upps[1],fst.lower[(full_vectors*8 +1)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +1)* result.vectors_count_row + j],wyn);
                    if constexpr(rest >= 3)
                        details::add_min(lows[2],upps[2],fst.lower[(full_vectors*8 +2)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +2)* result.vectors_count_row + j],wyn);
                    if constexpr(rest >= 4)
                        details::add_min(lows[3],upps[3],fst.lower[(full_vectors*8 +3)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +3)* result.vectors_count_row + j],wyn);
                    if constexpr(rest >= 5)
                        details::add_min(lows[4],upps[4],fst.lower[(full_vectors*8 +4)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +4)* result.vectors_count_row + j],wyn);
                    if constexpr(rest >= 6)
                        details::add_min(lows[5],upps[5],fst.lower[(full_vectors*8 +5)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +5)* result.vectors_count_row + j],wyn);
                    if constexpr(rest >= 7)
                        details::add_min(lows[6],upps[6],fst.lower[(full_vectors*8 +6)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +6)* result.vectors_count_row + j],wyn);
                }
            }
        }

        capd::rounding::DoubleRounding::roundUp();
        for (size_t i = 0; i < N; ++i) {
            for (size_t k = 0; k < full_vectors; ++k) {
                __m512d low = lower[i*vectors_count_row+k];
                __m512d up = upper[i*vectors_count_row+k];
                details::split_m512d_to_vectors(low,lows);
                details::split_m512d_to_vectors(up,upps);
                for (size_t j = 0; j < result.vectors_count_row; ++j) {
                    __m512d& wyn = result.upper[i * result.vectors_count_row + j];
                    details::add_max(lows[0],upps[0],fst.lower[k*8 * result.vectors_count_row + j],fst.upper[k*8 * result.vectors_count_row + j],wyn);
                    details::add_max(lows[1],upps[1],fst.lower[(k*8 +1)* result.vectors_count_row + j],fst.upper[(k*8 +1) * result.vectors_count_row + j],wyn);
                    details::add_max(lows[2],upps[2],fst.lower[(k*8 +2) * result.vectors_count_row + j],fst.upper[(k*8 +2) * result.vectors_count_row + j],wyn);
                    details::add_max(lows[3],upps[3],fst.lower[(k*8 +3) * result.vectors_count_row + j],fst.upper[(k*8 +3) * result.vectors_count_row + j],wyn);
                    details::add_max(lows[4],upps[4],fst.lower[(k*8 +4) * result.vectors_count_row + j],fst.upper[(k*8 +4) * result.vectors_count_row + j],wyn);
                    details::add_max(lows[5],upps[5],fst.lower[(k*8 +5) * result.vectors_count_row + j],fst.upper[(k*8 +5) * result.vectors_count_row + j],wyn);
                    details::add_max(lows[6],upps[6],fst.lower[(k*8 +6) * result.vectors_count_row + j],fst.upper[(k*8 +6) * result.vectors_count_row + j],wyn);
                    details::add_max(lows[7],upps[7],fst.lower[(k*8 +7) * result.vectors_count_row + j],fst.upper[(k*8 +7) * result.vectors_count_row + j],wyn);
                }
            }
            if constexpr(rest >= 1){
                __m512d low = lower[i*vectors_count_row+full_vectors];
                __m512d up = upper[i*vectors_count_row+full_vectors];
                details::split_m512d_to_vectors(low,lows);
                details::split_m512d_to_vectors(up,upps);
                for (size_t j = 0; j < result.vectors_count_row; ++j) {
                    __m512d& wyn = result.upper[i * result.vectors_count_row + j];
                    if constexpr(rest >= 1)
                        details::add_max(lows[0],upps[0],fst.lower[(full_vectors*8 +0)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +0)* result.vectors_count_row + j],wyn);
                    if constexpr(rest >= 2)
                        details::add_max(lows[1],upps[1],fst.lower[(full_vectors*8 +1)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +1)* result.vectors_count_row + j],wyn);
                    if constexpr(rest >= 3)
                        details::add_max(lows[2],upps[2],fst.lower[(full_vectors*8 +2)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +2)* result.vectors_count_row + j],wyn);
                    if constexpr(rest >= 4)
                        details::add_max(lows[3],upps[3],fst.lower[(full_vectors*8 +3)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +3)* result.vectors_count_row + j],wyn);
                    if constexpr(rest >= 5)
                        details::add_max(lows[4],upps[4],fst.lower[(full_vectors*8 +4)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +4)* result.vectors_count_row + j],wyn);
                    if constexpr(rest >= 6)
                        details::add_max(lows[5],upps[5],fst.lower[(full_vectors*8 +5)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +5)* result.vectors_count_row + j],wyn);
                    if constexpr(rest >= 7)
                        details::add_max(lows[6],upps[6],fst.lower[(full_vectors*8 +6)* result.vectors_count_row + j],fst.upper[(full_vectors*8 +6)* result.vectors_count_row + j],wyn);
                }
            }
        }
    }
    return result;
}

    BatchSwitchMatrixAVX512_Grouped & operator+=(const BatchSwitchMatrixAVX512_Grouped& fst){
        *this = *this+fst;
        return *this;
    }
    BatchSwitchMatrixAVX512_Grouped & operator-=(const BatchSwitchMatrixAVX512_Grouped& fst){
        *this = *this-fst;
        return *this;
    }

  friend BatchSwitchMatrixAVX512_Grouped operator+(const BatchSwitchMatrixAVX512_Grouped &fst, const Interval & scd){
    BatchSwitchMatrixAVX512_Grouped result(true); 
    __m512d scalar_low = _mm512_set1_pd(scd.leftBound());
    __m512d scalar_high = _mm512_set1_pd(scd.rightBound());

    if constexpr (enable_parallel)
    {
        #pragma omp parallel
        {
            capd::rounding::DoubleRounding::roundDown();
            #pragma omp for schedule(static) nowait
            for(size_t i = 0; i < result.vectors_count; i++){
                result.lower[i] = _mm512_add_pd(fst.lower[i], scalar_low);
            }

            capd::rounding::DoubleRounding::roundUp();
            #pragma omp for schedule(static)
            for(size_t i = 0; i < result.vectors_count; i++){
                result.upper[i] = _mm512_add_pd(fst.upper[i], scalar_high);
            }
        }
    }
    else
    {
        capd::rounding::DoubleRounding::roundDown();
        for(size_t i = 0; i < result.vectors_count; i++){
            result.lower[i] = _mm512_add_pd(fst.lower[i], scalar_low);
        }

        capd::rounding::DoubleRounding::roundUp();
        for(size_t i = 0; i < result.vectors_count; i++){
            result.upper[i] = _mm512_add_pd(fst.upper[i], scalar_high);
        }
    }
    return result;
}

friend BatchSwitchMatrixAVX512_Grouped operator-(const BatchSwitchMatrixAVX512_Grouped &fst, const Interval & scd){
    BatchSwitchMatrixAVX512_Grouped result(true);
  
    __m512d scalar_for_lower = _mm512_set1_pd(scd.rightBound());
    __m512d scalar_for_upper = _mm512_set1_pd(scd.leftBound()); 

    if constexpr (enable_parallel)
    {
        #pragma omp parallel
        {
            capd::rounding::DoubleRounding::roundDown();
            #pragma omp for schedule(static) nowait
            for(size_t i = 0; i < result.vectors_count; i++){
                result.lower[i] = _mm512_sub_pd(fst.lower[i], scalar_for_lower);
            }

            capd::rounding::DoubleRounding::roundUp();
            #pragma omp for schedule(static)
            for(size_t i = 0; i < result.vectors_count; i++){
                result.upper[i] = _mm512_sub_pd(fst.upper[i], scalar_for_upper);
            }
        }
    }
    else
    {
        capd::rounding::DoubleRounding::roundDown();
        for(size_t i = 0; i < result.vectors_count; i++){
            result.lower[i] = _mm512_sub_pd(fst.lower[i], scalar_for_lower);
        }

        capd::rounding::DoubleRounding::roundUp();
        for(size_t i = 0; i < result.vectors_count; i++){
            result.upper[i] = _mm512_sub_pd(fst.upper[i], scalar_for_upper);
        }
    }
    return result;
}
    friend BatchSwitchMatrixAVX512_Grouped operator*(const BatchSwitchMatrixAVX512_Grouped &fst, const Interval & scd){
    char type_scalar = type(scd.leftBound(), scd.rightBound());

    if constexpr(enable_parallel)
    {
        switch(type_scalar)
        {
            case 0:
                return BatchSwitchMatrixAVX512_Grouped();
            case 1: {
                __m512d scalar = _mm512_set1_pd(scd.leftBound());
                BatchSwitchMatrixAVX512_Grouped result(true);
                #pragma omp parallel
                {
                    capd::rounding::DoubleRounding::roundDown();
                    #pragma omp for schedule(static) nowait
                    for(size_t i = 0; i < result.vectors_count; i++){
                        result.lower[i] = _mm512_mul_pd(fst.lower[i], scalar);
                    }
                    capd::rounding::DoubleRounding::roundUp();
                    #pragma omp for schedule(static)
                    for(size_t i = 0; i < result.vectors_count; i++){
                        result.upper[i] = _mm512_mul_pd(fst.upper[i], scalar);
                    }
                }
                return result;
            }
            case 2: {
                __m512d scalar = _mm512_set1_pd(scd.leftBound());
                BatchSwitchMatrixAVX512_Grouped result(true);
                #pragma omp parallel
                {
                    capd::rounding::DoubleRounding::roundDown();
                    #pragma omp for schedule(static) nowait
                    for(size_t i = 0; i < result.vectors_count; i++){
                        result.lower[i] = _mm512_mul_pd(fst.upper[i], scalar);
                    }
                    capd::rounding::DoubleRounding::roundUp();
                    #pragma omp for schedule(static)
                    for(size_t i = 0; i < result.vectors_count; i++){
                        result.upper[i] = _mm512_mul_pd(fst.lower[i], scalar);
                    }
                }
                return result;
            }
            case 3: {
                __m512d scalar = _mm512_set1_pd(scd.rightBound());
                __m512d zero = _mm512_setzero_pd();
                BatchSwitchMatrixAVX512_Grouped result(true);
                #pragma omp parallel
                {
                    capd::rounding::DoubleRounding::roundDown();
                    #pragma omp for schedule(static) nowait
                    for(size_t i = 0; i < result.vectors_count; i++){
                        result.lower[i] = _mm512_min_pd(_mm512_mul_pd(fst.lower[i], scalar), zero);
                    }
                    capd::rounding::DoubleRounding::roundUp();
                    #pragma omp for schedule(static)
                    for(size_t i = 0; i < result.vectors_count; i++){
                        result.upper[i] = _mm512_max_pd(_mm512_mul_pd(fst.upper[i], scalar), zero);
                    }
                }
                return result;
            }
            case 4: {
                __m512d scalar1 = _mm512_set1_pd(scd.leftBound());
                __m512d scalar2 = _mm512_set1_pd(scd.rightBound());
                BatchSwitchMatrixAVX512_Grouped result(true);
                #pragma omp parallel
                {
                    capd::rounding::DoubleRounding::roundDown();
                    #pragma omp for schedule(static) nowait
                    for(size_t i = 0; i < result.vectors_count; i++){
                        result.lower[i] = _mm512_min_pd(_mm512_mul_pd(fst.lower[i], scalar1), _mm512_mul_pd(fst.lower[i], scalar2));
                    }
                    capd::rounding::DoubleRounding::roundUp();
                    #pragma omp for schedule(static)
                    for(size_t i = 0; i < result.vectors_count; i++){
                        result.upper[i] = _mm512_max_pd(_mm512_mul_pd(fst.upper[i], scalar1), _mm512_mul_pd(fst.upper[i], scalar2));
                    }
                }
                return result;
            }
            case 5: {
                __m512d scalar = _mm512_set1_pd(scd.leftBound());
                __m512d zero = _mm512_setzero_pd();
                BatchSwitchMatrixAVX512_Grouped result(true);
                #pragma omp parallel
                {
                    capd::rounding::DoubleRounding::roundDown();
                    #pragma omp for schedule(static) nowait
                    for(size_t i = 0; i < result.vectors_count; i++){
                        result.lower[i] = _mm512_min_pd(_mm512_mul_pd(fst.upper[i], scalar), zero);
                    }
                    capd::rounding::DoubleRounding::roundUp();
                    #pragma omp for schedule(static)
                    for(size_t i = 0; i < result.vectors_count; i++){
                        result.upper[i] = _mm512_max_pd(_mm512_mul_pd(fst.lower[i], scalar), zero);
                    }
                }
                return result;
            }
            case 6: {
                __m512d scalar1 = _mm512_set1_pd(scd.leftBound());
                __m512d scalar2 = _mm512_set1_pd(scd.rightBound());
                BatchSwitchMatrixAVX512_Grouped result(true);
                #pragma omp parallel 
                {
                    capd::rounding::DoubleRounding::roundDown();
                    #pragma omp for schedule(static) nowait
                    for(size_t i = 0; i < result.vectors_count; i++){
                        result.lower[i] = _mm512_min_pd(_mm512_mul_pd(fst.upper[i], scalar1), _mm512_mul_pd(fst.upper[i], scalar2));
                    }
                    capd::rounding::DoubleRounding::roundUp();
                    #pragma omp for schedule(static)
                    for(size_t i = 0; i < result.vectors_count; i++){
                        result.upper[i] = _mm512_max_pd(_mm512_mul_pd(fst.lower[i], scalar1), _mm512_mul_pd(fst.lower[i], scalar2));
                    }
                }
                return result;
            }
            case 7: {
                __m512d scalar1 = _mm512_set1_pd(scd.leftBound());
                __m512d scalar2 = _mm512_set1_pd(scd.rightBound());
                BatchSwitchMatrixAVX512_Grouped result(true);
                #pragma omp parallel
                {
                    capd::rounding::DoubleRounding::roundDown();
                    #pragma omp for schedule(static) nowait
                    for(size_t i = 0; i < result.vectors_count; i++){
                        result.lower[i] = _mm512_min_pd(_mm512_mul_pd(fst.upper[i], scalar1), _mm512_mul_pd(fst.lower[i], scalar2));
                    }
                    capd::rounding::DoubleRounding::roundUp();
                    #pragma omp for schedule(static)
                    for(size_t i = 0; i < result.vectors_count; i++){
                        result.upper[i] = _mm512_max_pd(_mm512_mul_pd(fst.lower[i], scalar1), _mm512_mul_pd(fst.upper[i], scalar2));
                    }
                }
                return result;
            }
        }
    }
    else 
    {
        switch(type_scalar)
        {
            case 0:
                return BatchSwitchMatrixAVX512_Grouped();
            case 1: {
                __m512d scalar = _mm512_set1_pd(scd.leftBound());
                BatchSwitchMatrixAVX512_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i < result.vectors_count; i++){
                    result.lower[i] = _mm512_mul_pd(fst.lower[i],scalar);
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i < result.vectors_count; i++){
                    result.upper[i] = _mm512_mul_pd(fst.upper[i],scalar);
                }
                return result;
            }
            case 2: {
                __m512d scalar = _mm512_set1_pd(scd.leftBound());
                BatchSwitchMatrixAVX512_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i < result.vectors_count; i++){
                    result.lower[i] = _mm512_mul_pd(fst.upper[i],scalar);
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i < result.vectors_count; i++){
                    result.upper[i] = _mm512_mul_pd(fst.lower[i],scalar);
                }
                return result;
            }
            case 3: {
                __m512d scalar = _mm512_set1_pd(scd.rightBound());
                __m512d zero = _mm512_setzero_pd();
                BatchSwitchMatrixAVX512_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i < result.vectors_count; i++){
                    result.lower[i] = _mm512_min_pd(_mm512_mul_pd(fst.lower[i],scalar),zero);
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i < result.vectors_count; i++){
                    result.upper[i] = _mm512_max_pd(_mm512_mul_pd(fst.upper[i],scalar),zero);
                }
                return result;
            }
            case 4: {
                __m512d scalar1 = _mm512_set1_pd(scd.leftBound());
                __m512d scalar2 = _mm512_set1_pd(scd.rightBound());
                BatchSwitchMatrixAVX512_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i < result.vectors_count; i++){
                    result.lower[i] = _mm512_min_pd(_mm512_mul_pd(fst.lower[i],scalar1),_mm512_mul_pd(fst.lower[i],scalar2));
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i < result.vectors_count; i++){
                    result.upper[i] = _mm512_max_pd(_mm512_mul_pd(fst.upper[i],scalar1),_mm512_mul_pd(fst.upper[i],scalar2));
                }
                return result;
            }
            case 5: {
                __m512d scalar = _mm512_set1_pd(scd.leftBound());
                __m512d zero = _mm512_setzero_pd();
                BatchSwitchMatrixAVX512_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i < result.vectors_count; i++){
                    result.lower[i] = _mm512_min_pd(_mm512_mul_pd(fst.upper[i],scalar),zero);
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i < result.vectors_count; i++){
                     result.upper[i] = _mm512_max_pd(_mm512_mul_pd(fst.lower[i],scalar),zero);
                }
                return result;
            }
            case 6: {
                __m512d scalar1 = _mm512_set1_pd(scd.leftBound());
                __m512d scalar2 = _mm512_set1_pd(scd.rightBound());
                BatchSwitchMatrixAVX512_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i < result.vectors_count; i++){
                    result.lower[i] = _mm512_min_pd(_mm512_mul_pd(fst.upper[i],scalar1),_mm512_mul_pd(fst.upper[i],scalar2));
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i < result.vectors_count; i++){
                    result.upper[i] = _mm512_max_pd(_mm512_mul_pd(fst.lower[i],scalar1),_mm512_mul_pd(fst.lower[i],scalar2));
                }
                return result;
            }
            case 7: {
                __m512d scalar1 = _mm512_set1_pd(scd.leftBound());
                __m512d scalar2 = _mm512_set1_pd(scd.rightBound());
                BatchSwitchMatrixAVX512_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i < result.vectors_count; i++){
                    result.lower[i] = _mm512_min_pd(_mm512_mul_pd(fst.upper[i],scalar1),_mm512_mul_pd(fst.lower[i],scalar2));
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i < result.vectors_count; i++){
                    result.upper[i] = _mm512_max_pd(_mm512_mul_pd(fst.lower[i],scalar1),_mm512_mul_pd(fst.upper[i],scalar2));
                }
                return result;
            }
        }
    }
    return BatchSwitchMatrixAVX512_Grouped(); 
}
    friend BatchSwitchMatrixAVX512_Grouped operator/(const BatchSwitchMatrixAVX512_Grouped &fst, const Interval & scd){
    char type_scalar = type(scd.leftBound(), scd.rightBound());

    if constexpr(enable_parallel)
    {
        switch (type_scalar)
        {
            case 0:
            case 3:
            case 5:
            case 7:
                throw std::invalid_argument("Scalar has 0 in it");
            case 1: {
                BatchSwitchMatrixAVX512_Grouped result(true);
                __m512d scalar = _mm512_set1_pd(scd.leftBound());
                #pragma omp parallel 
                {
                    capd::rounding::DoubleRounding::roundDown();
                    #pragma omp for schedule(static) nowait
                    for(size_t i = 0; i < result.vectors_count; i++){
                        result.lower[i] = _mm512_div_pd(fst.lower[i], scalar);
                    }
                    capd::rounding::DoubleRounding::roundUp();
                    #pragma omp for schedule(static)
                    for(size_t i = 0; i < result.vectors_count; i++){
                        result.upper[i] = _mm512_div_pd(fst.upper[i], scalar);
                    }
                }
                return result;
            }
            case 2: {
                __m512d scalar = _mm512_set1_pd(scd.leftBound());
                BatchSwitchMatrixAVX512_Grouped result(true);
                #pragma omp parallel 
                {
                    capd::rounding::DoubleRounding::roundDown();
                    #pragma omp for schedule(static) nowait
                    for(size_t i = 0; i < result.vectors_count; i++){
                        result.lower[i] = _mm512_div_pd(fst.upper[i], scalar);
                    }
                    capd::rounding::DoubleRounding::roundUp();
                    #pragma omp for schedule(static)
                    for(size_t i = 0; i < result.vectors_count; i++){
                        result.upper[i] = _mm512_div_pd(fst.lower[i], scalar);
                    }
                }
                return result;
            }
            case 4: {
                __m512d scalar1 = _mm512_set1_pd(scd.leftBound());
                __m512d scalar2 = _mm512_set1_pd(scd.rightBound());
                BatchSwitchMatrixAVX512_Grouped result(true);
                #pragma omp parallel 
                {
                    capd::rounding::DoubleRounding::roundDown();
                    #pragma omp for schedule(static) nowait
                    for(size_t i = 0; i < result.vectors_count; i++){
                        result.lower[i] = _mm512_min_pd(_mm512_div_pd(fst.lower[i], scalar1), _mm512_div_pd(fst.lower[i], scalar2));
                    }
                    capd::rounding::DoubleRounding::roundUp();
                    #pragma omp for schedule(static)
                    for(size_t i = 0; i < result.vectors_count; i++){
                        result.upper[i] = _mm512_max_pd(_mm512_div_pd(fst.upper[i], scalar1), _mm512_div_pd(fst.upper[i], scalar2));
                    }
                }
                return result;
            }
            case 6: {
                __m512d scalar1 = _mm512_set1_pd(scd.leftBound());
                __m512d scalar2 = _mm512_set1_pd(scd.rightBound());
                BatchSwitchMatrixAVX512_Grouped result(true);
                #pragma omp parallel 
                {
                    capd::rounding::DoubleRounding::roundDown();
                    #pragma omp for schedule(static) nowait
                    for(size_t i = 0; i < result.vectors_count; i++){
                        result.lower[i] = _mm512_min_pd(_mm512_div_pd(fst.upper[i], scalar1), _mm512_div_pd(fst.upper[i], scalar2));
                    }
                    capd::rounding::DoubleRounding::roundUp();
                    #pragma omp for schedule(static)
                    for(size_t i = 0; i < result.vectors_count; i++){
                        result.upper[i] = _mm512_min_pd(_mm512_div_pd(fst.lower[i], scalar1), _mm512_div_pd(fst.lower[i], scalar2));
                    }
                }
                return result;
            }
        }
    }
    else
    {
        switch (type_scalar)
        {
            case 0:
            case 3:
            case 5:
            case 7:
                throw std::invalid_argument("Scalar has 0 in it");
            case 1: {
                __m512d scalar = _mm512_set1_pd(scd.leftBound());
                BatchSwitchMatrixAVX512_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i < result.vectors_count; i++){
                    result.lower[i] = _mm512_div_pd(fst.lower[i],scalar);
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i < result.vectors_count; i++){
                    result.upper[i] = _mm512_div_pd(fst.upper[i],scalar);
                }
                return result;
            }
            case 2: {
                __m512d scalar = _mm512_set1_pd(scd.leftBound());
                BatchSwitchMatrixAVX512_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i < result.vectors_count; i++){
                    result.lower[i] = _mm512_div_pd(fst.upper[i],scalar);
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i < result.vectors_count; i++){
                    result.upper[i] = _mm512_div_pd(fst.lower[i],scalar);
                }
                return result;
            }
            case 4: {
                __m512d scalar1 = _mm512_set1_pd(scd.leftBound());
                __m512d scalar2 = _mm512_set1_pd(scd.rightBound());
                BatchSwitchMatrixAVX512_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i < result.vectors_count; i++){
                    result.lower[i] = _mm512_min_pd(_mm512_div_pd(fst.lower[i],scalar1),_mm512_div_pd(fst.lower[i],scalar2));
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i < result.vectors_count; i++){
                    result.upper[i] = _mm512_max_pd(_mm512_div_pd(fst.upper[i],scalar1),_mm512_div_pd(fst.upper[i],scalar2));
                }
                return result;
            }
            case 6: {
                __m512d scalar1 = _mm512_set1_pd(scd.leftBound());
                __m512d scalar2 = _mm512_set1_pd(scd.rightBound());
                BatchSwitchMatrixAVX512_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i < result.vectors_count; i++){
                    result.lower[i] = _mm512_min_pd(_mm512_div_pd(fst.upper[i],scalar1),_mm512_div_pd(fst.upper[i],scalar2));
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i < result.vectors_count; i++){
                    result.upper[i] = _mm512_min_pd(_mm512_div_pd(fst.lower[i],scalar1),_mm512_div_pd(fst.lower[i],scalar2));
                }
                return result;
            }
        }
    }
    return BatchSwitchMatrixAVX512_Grouped(); 
}
    BatchSwitchMatrixAVX512_Grouped & operator+=(const Interval& fst){
        *this = *this+fst;
        return *this;
    }
    BatchSwitchMatrixAVX512_Grouped & operator-=(const Interval& fst){
        *this = *this-fst;
        return *this;
    }
    BatchSwitchMatrixAVX512_Grouped & operator*=(const Interval& fst){
        *this = *this * fst;
        return *this;
    }
    BatchSwitchMatrixAVX512_Grouped & operator/=(const Interval& fst){
        *this = *this / fst;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, const BatchSwitchMatrixAVX512_Grouped& matrix){
        for (size_t row = 0; row < N; ++row) {
        os << "Row " << row << ": ";
        for (size_t vec = 0; vec < matrix.vectors_count_row; ++vec) {
            
            size_t index = row * matrix.vectors_count_row + vec;

            
            alignas(64) double elements[8];
            _mm512_store_pd(elements, matrix.upper[index]);

            os << "[";
            for (size_t i = 0; i < 8; ++i) {
                os << elements[i];
                if (i < 7) os << ", ";
            }
            os << "]";
            if (vec < matrix.vectors_count_row - 1) os << " ";
        }
        os << "\n";
    }
    return os;
    }

    IntervalProxy<Accessor,Index> operator()(size_t j, size_t k){
        Index ind(j*vectors_count_row+k/8,k%8);
        Accessor acc = {lower,upper};
        return IntervalProxy<Accessor,Index>(acc,ind);
    }
    IntervalProxy<Accessor,Index> operator()(size_t j, size_t k) const{
        Index ind(j*vectors_count_row+k/8,k%8);
        Accessor acc = {lower,upper};
        return IntervalProxy<Accessor,Index>(acc,ind);
    }

    bool operator==(const MatrixBasic<Interval,N,M> & fst){
        for(size_t i = 0; i <N; i++){
            for(size_t j = 0; j < M; j++){
                Interval interval = (*this)(i,j);
                if(interval != fst(i,j)) {
                    return false;
                }
            }
        }
        return true;
    }
   
    ~BatchSwitchMatrixAVX512_Grouped(){
        delete[] lower;
        delete[] upper;
    }
};

namespace details{
    inline void transpose8x8(const __m512d& row0, const __m512d& row1, const __m512d& row2, const __m512d& row3,
                         const __m512d& row4, const __m512d& row5, const __m512d& row6, const __m512d& row7,
                         __m512d& out0, __m512d& out1, __m512d& out2, __m512d& out3,
                         __m512d& out4, __m512d& out5, __m512d& out6, __m512d& out7) {
    __m512d t0 = _mm512_unpacklo_pd(row0, row1); // [row0[0], row1[0], row0[2], row1[2], row0[4], row1[4], row0[6], row1[6]]
    __m512d t1 = _mm512_unpackhi_pd(row0, row1); // [row0[1], row1[1], row0[3], row1[3], row0[5], row1[5], row0[7], row1[7]]
    __m512d t2 = _mm512_unpacklo_pd(row2, row3);
    __m512d t3 = _mm512_unpackhi_pd(row2, row3);
    __m512d t4 = _mm512_unpacklo_pd(row4, row5);
    __m512d t5 = _mm512_unpackhi_pd(row4, row5);
    __m512d t6 = _mm512_unpacklo_pd(row6, row7);
    __m512d t7 = _mm512_unpackhi_pd(row6, row7);


    __m512d m0 = _mm512_permutex2var_pd(t0,_mm512_set_epi64(11,10,3,2,9,8,1,0),t2);
    __m512d m1 = _mm512_permutex2var_pd(t0,_mm512_set_epi64(15,14,7,6,13,12,5,4),t2);
    __m512d m2 = _mm512_permutex2var_pd(t1,_mm512_set_epi64(11,10,3,2,9,8,1,0),t3);
    __m512d m3 = _mm512_permutex2var_pd(t1,_mm512_set_epi64(15,14,7,6,13,12,5,4),t3);
    __m512d m4 = _mm512_permutex2var_pd(t4,_mm512_set_epi64(11,10,3,2,9,8,1,0),t6);
    __m512d m5 = _mm512_permutex2var_pd(t4,_mm512_set_epi64(15,14,7,6,13,12,5,4),t6);
    __m512d m6 = _mm512_permutex2var_pd(t5,_mm512_set_epi64(11,10,3,2,9,8,1,0),t7);
    __m512d m7 = _mm512_permutex2var_pd(t5,_mm512_set_epi64(15,14,7,6,13,12,5,4),t7);

    out0 = _mm512_shuffle_f64x2(m0,m4,0b01000100);
    out1 = _mm512_shuffle_f64x2(m2,m6,0b01000100);
    out2 = _mm512_shuffle_f64x2(m0,m4,0b11101110);
    out3 = _mm512_shuffle_f64x2(m2,m6,0b11101110);
    out4 = _mm512_shuffle_f64x2(m1,m5,0b01000100);
    out5 = _mm512_shuffle_f64x2(m3,m7,0b01000100);
    out6 = _mm512_shuffle_f64x2(m1,m5,0b11101110);
    out7 = _mm512_shuffle_f64x2(m3,m7,0b11101110);
}

}
template<size_t N, size_t M>
BatchSwitchMatrixAVX512_Grouped<M, N> reorg(const BatchSwitchMatrixAVX512_Grouped<N,M>& fst){
    BatchSwitchMatrixAVX512_Grouped<M,N> result(true);
    size_t full1 = fst.full_vectors;
    size_t full2 = result.full_vectors;
    size_t offset = result.vectors_count / 8;
    for (size_t j = 0; j < N; j += 8) {
        for (size_t i = 0; i < full1; i++) {
            details::transpose8x8(
                fst.upper[full1 * j + i],
                fst.upper[full1 * (j + 1) + i],
                fst.upper[full1 * (j + 2) + i],
                fst.upper[full1 * (j + 3) + i],
                fst.upper[full1 * (j + 4)+ i],
                fst.upper[full1 * (j + 5) + i],
                fst.upper[full1 * (j + 6) + i],
                fst.upper[full1 * (j + 7) + i],
                result.upper[full2 * i + j/8],
                result.upper[full2 * i + offset + j/8],
                result.upper[full2 * i + offset*2 + j/8],
                result.upper[full2 * i + offset*3 + j/8],
                result.upper[full2 * i + offset*4 + j/8],
                result.upper[full2 * i + offset*5 + j/8],
                result.upper[full2 * i + offset*6 + j/8],
                result.upper[full2 * i + offset*7 + j/8]
            );
            details::transpose8x8(
                fst.lower[full1 * j + i],
                fst.lower[full1 * (j + 1) + i],
                fst.lower[full1 * (j + 2) + i],
                fst.lower[full1 * (j + 3) + i],
                fst.lower[full1 * (j + 4)+ i],
                fst.lower[full1 * (j + 5) + i],
                fst.lower[full1 * (j + 6) + i],
                fst.lower[full1 * (j + 7) + i],
                result.lower[full2 * i + j/8],
                result.lower[full2 * i + offset + j/8],
                result.lower[full2 * i + offset*2 + j/8],
                result.lower[full2 * i + offset*3 + j/8],
                result.lower[full2 * i + offset*4 + j/8],
                result.lower[full2 * i + offset*5 + j/8],
                result.lower[full2 * i + offset*6 + j/8],
                result.lower[full2 * i + offset*7 + j/8]
            );
        }
    }
    return result;   
}
#endif