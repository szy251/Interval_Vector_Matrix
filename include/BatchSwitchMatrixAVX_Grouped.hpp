#ifndef MATRIX_BATCH_AVX_GRP_HPP
#define MATRIX_BATCH_AVX_GRP_HPP

#include <iostream>
#include <capd/rounding/DoubleRounding.h>
#include <capd/filib/Interval.h>
#include <Utilities.hpp>
#include <IntervalProxy.hpp>
#include <MatrixBasic.hpp>
#include <immintrin.h>
#include <omp.h>

namespace details{
    inline void add_min(const __m256d& a, const __m256d& b,const __m256d& c,const __m256d& d,__m256d& result){
    __m256d mlt1 = _mm256_min_pd(_mm256_mul_pd(a,c),_mm256_mul_pd(a,d));
    __m256d mlt2 = _mm256_min_pd(_mm256_mul_pd(b,c),_mm256_mul_pd(b,d));
    result = _mm256_add_pd(result, _mm256_min_pd(mlt1,mlt2));
    }
    inline void add_max(const __m256d& a, const __m256d& b,const __m256d& c,const __m256d& d,__m256d& result){
        __m256d mlt1 = _mm256_max_pd(_mm256_mul_pd(a,c),_mm256_mul_pd(a,d));
        __m256d mlt2 = _mm256_max_pd(_mm256_mul_pd(b,c),_mm256_mul_pd(b,d));
        result = _mm256_add_pd(result, _mm256_max_pd(mlt1,mlt2));
    }
    inline void split_m256d_to_vectors(__m256d vec, __m256d* result) {
        result[0] = _mm256_permute4x64_pd(vec, 0b00000000); // Powtórz element 0
        result[1] = _mm256_permute4x64_pd(vec, 0b01010101); // Powtórz element 1
        result[2] = _mm256_permute4x64_pd(vec, 0b10101010); // Powtórz element 2
        result[3] = _mm256_permute4x64_pd(vec, 0b11111111); // Powtórz element 3
    }
}
template<size_t N, size_t M>
class BatchSwitchMatrixAVX_Grouped
{
private:
    __m256d* lower = nullptr;
    __m256d* upper = nullptr;
    typedef capd::filib::Interval<double> Interval;
    static constexpr size_t vectors_count_row = (M+3)/4;
    static constexpr size_t vectors_count = vectors_count_row*N;
    static constexpr size_t full_vectors = M/4;
    static constexpr size_t rest = M % 4;
    static constexpr bool enable_parallel = (N>=100) &&  (M>=100);

    template<size_t N1,size_t M1>
    friend class BatchSwitchMatrixAVX_Grouped;

    BatchSwitchMatrixAVX_Grouped(bool allocateOnly)
    {
        if(allocateOnly){    
            lower = new alignas(32) __m256d[vectors_count];
            upper = new alignas(32) __m256d[vectors_count];
        }
        else{    
            lower = nullptr;
            upper = nullptr;
        }
    }

    struct Index{
        size_t ind;
        uint8_t poz;
    };
    struct Accessor {
        __m256d* lower,*upper;
        Interval get(Index index) const {
            __m256d vec_lower = lower[index.ind];
            __m256d vec_upper = upper[index.ind];
            switch (index.poz)
            {
            case 3:{
                __m128d low = _mm256_extractf128_pd(vec_lower,1);
                double low_val = _mm_cvtsd_f64(_mm_shuffle_pd(low,low,1));
                __m128d upp = _mm256_extractf128_pd(vec_upper,1);
                double upp_val = _mm_cvtsd_f64(_mm_shuffle_pd(upp,upp,1));
                return Interval(low_val,upp_val);
            }
            case 2:{
                __m128d low = _mm256_extractf128_pd(vec_lower,1);
                double low_val = _mm_cvtsd_f64(low);
                __m128d upp = _mm256_extractf128_pd(vec_upper,1);
                double upp_val = _mm_cvtsd_f64(upp);
                return Interval(low_val,upp_val);
            }
            case 1:{
                __m128d low = _mm256_castpd256_pd128(vec_lower);
                double low_val = _mm_cvtsd_f64(_mm_shuffle_pd(low,low,1));
                __m128d upp = _mm256_castpd256_pd128(vec_upper);
                double upp_val = _mm_cvtsd_f64(_mm_shuffle_pd(upp,upp,1));
                return Interval(low_val,upp_val);
            }
            default:
                 __m128d low = _mm256_castpd256_pd128(vec_lower);
                double low_val = _mm_cvtsd_f64(low);
                __m128d upp = _mm256_castpd256_pd128(vec_upper);
                double upp_val = _mm_cvtsd_f64(upp);
                return Interval(low_val,upp_val);
            }
        }
        void set(Index index, const Interval& interval) {
            __m256d low = _mm256_set1_pd(interval.leftBound());
            __m256d upp = _mm256_set1_pd(interval.rightBound());
            __m256d vec_lower = lower[index.ind];
            __m256d vec_upper = upper[index.ind];
            switch (index.poz)
            {
            case  3:{
                vec_lower = _mm256_blend_pd(vec_lower,low,0b1000);
                vec_upper = _mm256_blend_pd(vec_upper,upp,0b1000);
                break;
            }
            case  2:{
                vec_lower = _mm256_blend_pd(vec_lower,low,0b0100);
                vec_upper = _mm256_blend_pd(vec_upper,upp,0b0100);
                break;
            }
            case  1:{
                vec_lower = _mm256_blend_pd(vec_lower,low,0b0010);
                vec_upper = _mm256_blend_pd(vec_upper,upp,0b0010);
                break;
            }
            default :{
                vec_lower = _mm256_blend_pd(vec_lower,low,0b0001);
                vec_upper = _mm256_blend_pd(vec_upper,upp,0b0001);
                break;
            }
            }
            lower[index.ind] = vec_lower;
            upper[index.ind] = vec_upper;
        }
    };
public:
    static constexpr size_t rows = N;
    static constexpr size_t cols = M;
    BatchSwitchMatrixAVX_Grouped(){
        if constexpr (enable_parallel){
            lower = new alignas(32) __m256d[vectors_count];
            upper = new alignas(32) __m256d[vectors_count];
            #pragma omp parallel for
            for(size_t i = 0; i < vectors_count; i++){
                lower[i] = _mm256_setzero_pd();
                upper[i] = _mm256_setzero_pd();
            }
        }
        else{
            lower = new alignas(32) __m256d[vectors_count]();
            upper = new alignas(32) __m256d[vectors_count]();
        }
    }

    BatchSwitchMatrixAVX_Grouped(const BatchSwitchMatrixAVX_Grouped& cpy){
        lower = new alignas(32) __m256d[vectors_count];
        upper = new alignas(32) __m256d[vectors_count];
        
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

    BatchSwitchMatrixAVX_Grouped(BatchSwitchMatrixAVX_Grouped&& cpy) noexcept {
        lower = cpy.lower;
        upper = cpy.upper;
        cpy.lower = nullptr;
        cpy.upper = nullptr;
    }

    BatchSwitchMatrixAVX_Grouped(const double(&array)[2 * N * M]){
        lower = new alignas(32) __m256d[vectors_count];
        upper = new alignas(32) __m256d[vectors_count];

        if constexpr (enable_parallel){
            #pragma omp parallel for
            for (size_t row = 0; row < N; ++row) {
                size_t offset = row * M * 2; 
                
                for (size_t i = 0; i < full_vectors * 8; i += 8) {
                    __m256d a = _mm256_set_pd(
                        array[offset + i + 6],
                        array[offset + i + 4],
                        array[offset + i + 2],
                        array[offset + i + 0]
                    );
                    __m256d b = _mm256_set_pd(
                        array[offset + i + 7],
                        array[offset + i + 5],
                        array[offset + i + 3],
                        array[offset + i + 1]
                    );
                    size_t vector_index = row * vectors_count_row + i / 8;
                    lower[vector_index] = a;
                    upper[vector_index] = b;
                }

                if (rest > 0) {
                    double temp_lower[4] = {};
                    double temp_upper[4] = {};
                    for (size_t i = 0; i < rest; ++i) {
                        temp_lower[i] = array[offset + 2 * M - 2 * rest + 2 * i];
                        temp_upper[i] = array[offset + 2 * M - 2 * rest + 2 * i + 1];
                    }
                    size_t vector_index = row * vectors_count_row + full_vectors;
                    lower[vector_index] = _mm256_load_pd(temp_lower);
                    upper[vector_index] = _mm256_load_pd(temp_upper);
                }
            }
        }
        else{
            for (size_t row = 0; row < N; ++row) {
                size_t offset = row * M * 2; 
                
                for (size_t i = 0; i < full_vectors * 8; i += 8) {
                    __m256d a = _mm256_set_pd(
                        array[offset + i + 6],
                        array[offset + i + 4],
                        array[offset + i + 2],
                        array[offset + i + 0]
                    );
                    __m256d b = _mm256_set_pd(
                        array[offset + i + 7],
                        array[offset + i + 5],
                        array[offset + i + 3],
                        array[offset + i + 1]
                    );
                    size_t vector_index = row * vectors_count_row + i / 8;
                    lower[vector_index] = a;
                    upper[vector_index] = b;
                }

                if (rest > 0) {
                    double temp_lower[4] = {};
                    double temp_upper[4] = {};
                    for (size_t i = 0; i < rest; ++i) {
                        temp_lower[i] = array[offset + 2 * M - 2 * rest + 2 * i];
                        temp_upper[i] = array[offset + 2 * M - 2 * rest + 2 * i + 1];
                    }
                    size_t vector_index = row * vectors_count_row + full_vectors;
                    lower[vector_index] = _mm256_load_pd(temp_lower);
                    upper[vector_index] = _mm256_load_pd(temp_upper);
                }
            }
        }
    }

    BatchSwitchMatrixAVX_Grouped(const std::vector<double>& array) {
        if (array.size() != 2 * N * M) {
            throw std::invalid_argument("Rozmiar wektora musi wynosić dokładnie 2 * N * M.");
        }
        lower = new alignas(32) __m256d[vectors_count];
        upper = new alignas(32) __m256d[vectors_count];

        if constexpr (enable_parallel){
            #pragma omp parallel for
            for (size_t row = 0; row < N; ++row) {
                size_t offset = row * M * 2; 
                
                for (size_t i = 0; i < full_vectors * 8; i += 8) {
                    __m256d a = _mm256_set_pd(
                        array[offset + i + 6],
                        array[offset + i + 4],
                        array[offset + i + 2],
                        array[offset + i + 0]
                    );
                    __m256d b = _mm256_set_pd(
                        array[offset + i + 7],
                        array[offset + i + 5],
                        array[offset + i + 3],
                        array[offset + i + 1]
                    );
                    size_t vector_index = row * vectors_count_row + i / 8;
                    lower[vector_index] = a;
                    upper[vector_index] = b;
                }

                if (rest > 0) {
                    double temp_lower[4] = {};
                    double temp_upper[4] = {};
                    for (size_t i = 0; i < rest; ++i) {
                        temp_lower[i] = array[offset + 2 * M - 2 * rest + 2 * i];
                        temp_upper[i] = array[offset + 2 * M - 2 * rest + 2 * i + 1];
                    }
                    size_t vector_index = row * vectors_count_row + full_vectors;
                    lower[vector_index] = _mm256_load_pd(temp_lower);
                    upper[vector_index] = _mm256_load_pd(temp_upper);
                }
            }
        }
        else{
            for (size_t row = 0; row < N; ++row) {
                size_t offset = row * M * 2; 
                
                for (size_t i = 0; i < full_vectors * 8; i += 8) {
                    __m256d a = _mm256_set_pd(
                        array[offset + i + 6],
                        array[offset + i + 4],
                        array[offset + i + 2],
                        array[offset + i + 0]
                    );
                    __m256d b = _mm256_set_pd(
                        array[offset + i + 7],
                        array[offset + i + 5],
                        array[offset + i + 3],
                        array[offset + i + 1]
                    );
                    size_t vector_index = row * vectors_count_row + i / 8;
                    lower[vector_index] = a;
                    upper[vector_index] = b;
                }

                if (rest > 0) {
                    double temp_lower[4] = {};
                    double temp_upper[4] = {};
                    for (size_t i = 0; i < rest; ++i) {
                        temp_lower[i] = array[offset + 2 * M - 2 * rest + 2 * i];
                        temp_upper[i] = array[offset + 2 * M - 2 * rest + 2 * i + 1];
                    }
                    size_t vector_index = row * vectors_count_row + full_vectors;
                    lower[vector_index] = _mm256_load_pd(temp_lower);
                    upper[vector_index] = _mm256_load_pd(temp_upper);
                }
            }
        }
    }

    BatchSwitchMatrixAVX_Grouped& operator=(const BatchSwitchMatrixAVX_Grouped& fst) {
        if (this != &fst) {
            if (!fst.lower) {
                delete[] lower;
                lower = nullptr;
                delete[] upper;
                upper = nullptr;
            } else {
                if (!lower) {
                    lower = new alignas(32) __m256d[vectors_count];
                    upper = new alignas(32) __m256d[vectors_count];
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

    BatchSwitchMatrixAVX_Grouped& operator=(BatchSwitchMatrixAVX_Grouped&& fst) noexcept {
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
    friend BatchSwitchMatrixAVX_Grouped<N1, M1> reorg(const BatchSwitchMatrixAVX_Grouped<N1,M1>& fst);

    friend BatchSwitchMatrixAVX_Grouped operator+(const BatchSwitchMatrixAVX_Grouped& fst, const BatchSwitchMatrixAVX_Grouped& scd){
        BatchSwitchMatrixAVX_Grouped result(true);
        if constexpr(enable_parallel)
        {
            #pragma omp parallel
            { 
                capd::rounding::DoubleRounding::roundDown();
                #pragma omp for schedule(static) nowait
                for(size_t i = 0; i < result.vectors_count; i++) {
                    result.lower[i] = _mm256_add_pd(fst.lower[i], scd.lower[i]);
                }

                capd::rounding::DoubleRounding::roundUp();
                #pragma omp for schedule(static)
                for(size_t i = 0; i < result.vectors_count; i++) {
                    result.upper[i] = _mm256_add_pd(fst.upper[i], scd.upper[i]);
                }
            }
        }
        else
        {
            capd::rounding::DoubleRounding::roundDown();

            for(size_t i = 0; i < result.vectors_count; i++) {
                result.lower[i] = _mm256_add_pd(fst.lower[i], scd.lower[i]);
            }
    
            capd::rounding::DoubleRounding::roundUp();
             for(size_t i = 0; i < result.vectors_count; i++) {
                result.upper[i] = _mm256_add_pd(fst.upper[i], scd.upper[i]);
            }
        }
        

        return result;
    }

    friend BatchSwitchMatrixAVX_Grouped operator-(const BatchSwitchMatrixAVX_Grouped& fst, const BatchSwitchMatrixAVX_Grouped& scd){
        BatchSwitchMatrixAVX_Grouped result(true);
        if constexpr(enable_parallel)
        {
            #pragma omp parallel
            { 
                capd::rounding::DoubleRounding::roundDown();
                #pragma omp for schedule(static) nowait
                for(size_t i = 0; i < result.vectors_count; i++) {
                    result.lower[i] = _mm256_sub_pd(fst.lower[i], scd.lower[i]);
                }

                capd::rounding::DoubleRounding::roundUp();
                #pragma omp for schedule(static)
                for(size_t i = 0; i < result.vectors_count; i++) {
                    result.upper[i] = _mm256_sub_pd(fst.upper[i], scd.upper[i]);
                }
            }
        }
        else
        {
            capd::rounding::DoubleRounding::roundDown();

            for(size_t i = 0; i < result.vectors_count; i++) {
                result.lower[i] = _mm256_sub_pd(fst.lower[i], scd.lower[i]);
            }
    
            capd::rounding::DoubleRounding::roundUp();
             for(size_t i = 0; i < result.vectors_count; i++) {
                result.upper[i] = _mm256_sub_pd(fst.upper[i], scd.upper[i]);
            }
        }
        return result;
    }


    //mnożenie z rerganizacją danych


    // template <size_t P>
    // BatchSwitchMatrixAVX_Grouped<N, P> operator*(const BatchSwitchMatrixAVX_Grouped<M, P>& fst) {
    //     BatchSwitchMatrixAVX_Grouped<N, P> result(true);
    //     auto pom = reorg(fst);

    //     __m256d * one_lower= pom.lower;
    //     __m256d * two_lower= pom.lower + vectors_count/4;
    //     __m256d * three_lower= pom.lower + vectors_count/2;
    //     __m256d * four_lower= pom.lower + vectors_count*3/4;
    //     __m256d * one_upper= pom.upper;
    //     __m256d * two_upper= pom.upper + vectors_count/4;
    //     __m256d * three_upper= pom.upper + vectors_count/2;
    //     __m256d * four_upper= pom.upper + vectors_count*3/4;
    //     // Round down pass
    //     //(i,k), (j,k)
    //     capd::rounding::DoubleRounding::roundDown();
    //     for (size_t i = 0; i < N; ++i) {
    //         for (size_t j = 0; j < result.vectors_count_row; ++j) {
    //             __m256d sum1 = _mm256_setzero_pd(), sum2 = _mm256_setzero_pd(),sum3 = _mm256_setzero_pd(),sum4 = _mm256_setzero_pd();
    //             for (size_t k = 0; k < vectors_count_row; ++k) {
    //                 __m256d low = lower[i*vectors_count_row+k];
    //                 __m256d up = upper[i*vectors_count_row+k];
    //                 details::add_min(low,up,one_lower[j*vectors_count_row+k],one_upper[j*vectors_count_row+k],sum1);
    //                 details::add_min(low,up,two_lower[j*vectors_count_row+k],two_upper[j*vectors_count_row+k],sum2);
    //                 details::add_min(low,up,three_lower[j*vectors_count_row+k],three_upper[j*vectors_count_row+k],sum3);
    //                 details::add_min(low,up,four_lower[j*vectors_count_row+k],four_upper[j*vectors_count_row+k],sum4);
    //             }
    //             sum1 = _mm256_hadd_pd(sum1,sum1);
    //             sum2 = _mm256_hadd_pd(sum2,sum2);
    //             sum3 = _mm256_hadd_pd(sum3,sum4);
    //             sum4 = _mm256_hadd_pd(sum4,sum4);
    //             sum1 = _mm256_permute4x64_pd(sum1, 0b11011000);
    //             sum2 = _mm256_permute4x64_pd(sum2, 0b11011000);
    //             sum3 = _mm256_permute4x64_pd(sum3, 0b11011000);
    //             sum4 = _mm256_permute4x64_pd(sum4, 0b11011000);
    //             sum1 = _mm256_hadd_pd(sum1,sum2);
    //             sum2 = _mm256_hadd_pd(sum3,sum4);
    //             result.lower[i * result.vectors_count_row + j] = _mm256_permute2f128_pd(sum1, sum2, 0b00110000);
    //         }
            
    //     }

    //     // Round up pass
    //     capd::rounding::DoubleRounding::roundUp();
    //     for (size_t i = 0; i < N; ++i) {
    //         for (size_t j = 0; j < result.vectors_count_row; ++j) {
    //             __m256d sum1 = _mm256_setzero_pd(), sum2 = _mm256_setzero_pd(),sum3 = _mm256_setzero_pd(),sum4 = _mm256_setzero_pd();
    //             for (size_t k = 0; k < vectors_count_row; ++k) {
    //                 __m256d low = lower[i*vectors_count_row+k];
    //                 __m256d up = upper[i*vectors_count_row+k];
    //                 details::add_max(low,up,one_lower[j*vectors_count_row+k],one_upper[j*vectors_count_row+k],sum1);
    //                 details::add_max(low,up,two_lower[j*vectors_count_row+k],two_upper[j*vectors_count_row+k],sum2);
    //                 details::add_max(low,up,three_lower[j*vectors_count_row+k],three_upper[j*vectors_count_row+k],sum3);
    //                 details::add_max(low,up,four_lower[j*vectors_count_row+k],four_upper[j*vectors_count_row+k],sum4);
    //             }
    //             sum1 = _mm256_hadd_pd(sum1,sum1);
    //             sum2 = _mm256_hadd_pd(sum2,sum2);
    //             sum3 = _mm256_hadd_pd(sum3,sum4);
    //             sum4 = _mm256_hadd_pd(sum4,sum4);
    //             sum1 = _mm256_permute4x64_pd(sum1, 0b11011000);
    //             sum2 = _mm256_permute4x64_pd(sum2, 0b11011000);
    //             sum3 = _mm256_permute4x64_pd(sum3, 0b11011000);
    //             sum4 = _mm256_permute4x64_pd(sum4, 0b11011000);
    //             sum1 = _mm256_hadd_pd(sum1,sum2);
    //             sum2 = _mm256_hadd_pd(sum3,sum4);
    //             result.upper[i * result.vectors_count_row + j] = _mm256_permute2f128_pd(sum1, sum2, 0b00110000);
    //         }
            
    //     }


    //     return result;
    // }


    //standard 

    template <size_t P>
    BatchSwitchMatrixAVX_Grouped<N, P> operator*(const BatchSwitchMatrixAVX_Grouped<M, P>& fst) {
        BatchSwitchMatrixAVX_Grouped<N, P> result{};

        // Warunek aktywacji równoległości 
        static constexpr bool enable_parallel_mlt = (N >= 30) && (M >= 30) && (P >=30);


        if constexpr(enable_parallel_mlt){
            #pragma omp parallel
            {
                capd::rounding::DoubleRounding::roundDown();

                #pragma omp for schedule(static) nowait
                for (size_t i = 0; i < N; ++i) {
                    alignas(32) __m256d lows[4];
                    alignas(32) __m256d upps[4];
                    
                    // Główna część - pełne wektory
                    for (size_t k = 0; k < full_vectors; ++k) {
                        __m256d low = lower[i*vectors_count_row + k];
                        __m256d up = upper[i*vectors_count_row + k];
                        details::split_m256d_to_vectors(low, lows);
                        details::split_m256d_to_vectors(up, upps);
                        
                        for (size_t j = 0; j < result.vectors_count_row; ++j) {
                            __m256d& wyn = result.lower[i * result.vectors_count_row + j];
                            details::add_min(lows[0], upps[0], fst.lower[(k*4 + 0) * result.vectors_count_row + j], fst.upper[(k*4 + 0) * result.vectors_count_row + j], wyn);
                            details::add_min(lows[1], upps[1], fst.lower[(k*4 + 1) * result.vectors_count_row + j], fst.upper[(k*4 + 1) * result.vectors_count_row + j], wyn);
                            details::add_min(lows[2], upps[2], fst.lower[(k*4 + 2) * result.vectors_count_row + j], fst.upper[(k*4 + 2) * result.vectors_count_row + j], wyn);
                            details::add_min(lows[3], upps[3], fst.lower[(k*4 + 3) * result.vectors_count_row + j], fst.upper[(k*4 + 3) * result.vectors_count_row + j], wyn);
                        }
                    }
                    
                    // Reszta (0-3 elementy)
                    if constexpr (rest > 0) {
                        __m256d low = lower[i*vectors_count_row + full_vectors];
                        __m256d up = upper[i*vectors_count_row + full_vectors];
                        details::split_m256d_to_vectors(low, lows);
                        details::split_m256d_to_vectors(up, upps);
                        
                        for (size_t j = 0; j < result.vectors_count_row; ++j) {
                            __m256d& wyn = result.lower[i * result.vectors_count_row + j];
                            if constexpr (rest >= 1) details::add_min(lows[0], upps[0], fst.lower[(full_vectors*4 + 0) * result.vectors_count_row + j], fst.upper[(full_vectors*4 + 0) * result.vectors_count_row + j], wyn);
                            if constexpr (rest >= 2) details::add_min(lows[1], upps[1], fst.lower[(full_vectors*4 + 1) * result.vectors_count_row + j], fst.upper[(full_vectors*4 + 1) * result.vectors_count_row + j], wyn);
                            if constexpr (rest >= 3) details::add_min(lows[2], upps[2], fst.lower[(full_vectors*4 + 2) * result.vectors_count_row + j], fst.upper[(full_vectors*4 + 2) * result.vectors_count_row + j], wyn);
                        }
                    }
                }

                //#pragma omp barrier

                capd::rounding::DoubleRounding::roundUp();
                
                #pragma omp for schedule(static)
                for (size_t i = 0; i < N; ++i) {
                    alignas(32) __m256d lows[4];
                    alignas(32) __m256d upps[4];
                    
                    for (size_t k = 0; k < full_vectors; ++k) {
                        __m256d low = lower[i*vectors_count_row + k];
                        __m256d up = upper[i*vectors_count_row + k];
                        details::split_m256d_to_vectors(low, lows);
                        details::split_m256d_to_vectors(up, upps);
                        
                        for (size_t j = 0; j < result.vectors_count_row; ++j) {
                            __m256d& wyn = result.upper[i * result.vectors_count_row + j];
                            details::add_max(lows[0], upps[0], fst.lower[(k*4 + 0) * result.vectors_count_row + j], fst.upper[(k*4 + 0) * result.vectors_count_row + j], wyn);
                            details::add_max(lows[1], upps[1], fst.lower[(k*4 + 1) * result.vectors_count_row + j], fst.upper[(k*4 + 1) * result.vectors_count_row + j], wyn);
                            details::add_max(lows[2], upps[2], fst.lower[(k*4 + 2) * result.vectors_count_row + j], fst.upper[(k*4 + 2) * result.vectors_count_row + j], wyn);
                            details::add_max(lows[3], upps[3], fst.lower[(k*4 + 3) * result.vectors_count_row + j], fst.upper[(k*4 + 3) * result.vectors_count_row + j], wyn);
                        }
                    }
                    
                    if constexpr (rest > 0) {
                        __m256d low = lower[i*vectors_count_row + full_vectors];
                        __m256d up = upper[i*vectors_count_row + full_vectors];
                        details::split_m256d_to_vectors(low, lows);
                        details::split_m256d_to_vectors(up, upps);
                        
                        for (size_t j = 0; j < result.vectors_count_row; ++j) {
                            __m256d& wyn = result.upper[i * result.vectors_count_row + j];
                            if constexpr (rest >= 1) details::add_max(lows[0], upps[0], fst.lower[(full_vectors*4 + 0) * result.vectors_count_row + j], fst.upper[(full_vectors*4 + 0) * result.vectors_count_row + j], wyn);
                            if constexpr (rest >= 2) details::add_max(lows[1], upps[1], fst.lower[(full_vectors*4 + 1) * result.vectors_count_row + j], fst.upper[(full_vectors*4 + 1) * result.vectors_count_row + j], wyn);
                            if constexpr (rest >= 3) details::add_max(lows[2], upps[2], fst.lower[(full_vectors*4 + 2) * result.vectors_count_row + j], fst.upper[(full_vectors*4 + 2) * result.vectors_count_row + j], wyn);
                        }
                    }
                }
            }
        }
        else{
            

            capd::rounding::DoubleRounding::roundDown();
            alignas(32) __m256d lows[4];
            alignas(32) __m256d upps[4];
            for (size_t i = 0; i < N; ++i) {
                
                for (size_t k = 0; k < full_vectors; ++k) {
                    __m256d low = lower[i*vectors_count_row + k];
                    __m256d up = upper[i*vectors_count_row + k];
                    details::split_m256d_to_vectors(low, lows);
                    details::split_m256d_to_vectors(up, upps);
                    
                    for (size_t j = 0; j < result.vectors_count_row; ++j) {
                        __m256d& wyn = result.lower[i * result.vectors_count_row + j];
                        details::add_min(lows[0], upps[0], fst.lower[(k*4 + 0) * result.vectors_count_row + j], fst.upper[(k*4 + 0) * result.vectors_count_row + j], wyn);
                        details::add_min(lows[1], upps[1], fst.lower[(k*4 + 1) * result.vectors_count_row + j], fst.upper[(k*4 + 1) * result.vectors_count_row + j], wyn);
                        details::add_min(lows[2], upps[2], fst.lower[(k*4 + 2) * result.vectors_count_row + j], fst.upper[(k*4 + 2) * result.vectors_count_row + j], wyn);
                        details::add_min(lows[3], upps[3], fst.lower[(k*4 + 3) * result.vectors_count_row + j], fst.upper[(k*4 + 3) * result.vectors_count_row + j], wyn);
                    }
                }
                
                // Reszta (0-3 elementy)
                if constexpr (rest > 0) {
                    __m256d low = lower[i*vectors_count_row + full_vectors];
                    __m256d up = upper[i*vectors_count_row + full_vectors];
                    details::split_m256d_to_vectors(low, lows);
                    details::split_m256d_to_vectors(up, upps);
                    
                    for (size_t j = 0; j < result.vectors_count_row; ++j) {
                        __m256d& wyn = result.lower[i * result.vectors_count_row + j];
                        if constexpr (rest >= 1) details::add_min(lows[0], upps[0], fst.lower[(full_vectors*4 + 0) * result.vectors_count_row + j], fst.upper[(full_vectors*4 + 0) * result.vectors_count_row + j], wyn);
                        if constexpr (rest >= 2) details::add_min(lows[1], upps[1], fst.lower[(full_vectors*4 + 1) * result.vectors_count_row + j], fst.upper[(full_vectors*4 + 1) * result.vectors_count_row + j], wyn);
                        if constexpr (rest >= 3) details::add_min(lows[2], upps[2], fst.lower[(full_vectors*4 + 2) * result.vectors_count_row + j], fst.upper[(full_vectors*4 + 2) * result.vectors_count_row + j], wyn);
                    }
                }
            }


            capd::rounding::DoubleRounding::roundUp();
            
            for (size_t i = 0; i < N; ++i) {
                
                for (size_t k = 0; k < full_vectors; ++k) {
                    __m256d low = lower[i*vectors_count_row + k];
                    __m256d up = upper[i*vectors_count_row + k];
                    details::split_m256d_to_vectors(low, lows);
                    details::split_m256d_to_vectors(up, upps);
                    
                    for (size_t j = 0; j < result.vectors_count_row; ++j) {
                        __m256d& wyn = result.upper[i * result.vectors_count_row + j];
                        details::add_max(lows[0], upps[0], fst.lower[(k*4 + 0) * result.vectors_count_row + j], fst.upper[(k*4 + 0) * result.vectors_count_row + j], wyn);
                        details::add_max(lows[1], upps[1], fst.lower[(k*4 + 1) * result.vectors_count_row + j], fst.upper[(k*4 + 1) * result.vectors_count_row + j], wyn);
                        details::add_max(lows[2], upps[2], fst.lower[(k*4 + 2) * result.vectors_count_row + j], fst.upper[(k*4 + 2) * result.vectors_count_row + j], wyn);
                        details::add_max(lows[3], upps[3], fst.lower[(k*4 + 3) * result.vectors_count_row + j], fst.upper[(k*4 + 3) * result.vectors_count_row + j], wyn);
                    }
                }
                
                if constexpr (rest > 0) {
                    __m256d low = lower[i*vectors_count_row + full_vectors];
                    __m256d up = upper[i*vectors_count_row + full_vectors];
                    details::split_m256d_to_vectors(low, lows);
                    details::split_m256d_to_vectors(up, upps);
                    
                    for (size_t j = 0; j < result.vectors_count_row; ++j) {
                        __m256d& wyn = result.upper[i * result.vectors_count_row + j];
                        if constexpr (rest >= 1) details::add_max(lows[0], upps[0], fst.lower[(full_vectors*4 + 0) * result.vectors_count_row + j], fst.upper[(full_vectors*4 + 0) * result.vectors_count_row + j], wyn);
                        if constexpr (rest >= 2) details::add_max(lows[1], upps[1], fst.lower[(full_vectors*4 + 1) * result.vectors_count_row + j], fst.upper[(full_vectors*4 + 1) * result.vectors_count_row + j], wyn);
                        if constexpr (rest >= 3) details::add_max(lows[2], upps[2], fst.lower[(full_vectors*4 + 2) * result.vectors_count_row + j], fst.upper[(full_vectors*4 + 2) * result.vectors_count_row + j], wyn);
                    }
                }
            }
        }
        return result;
    }
    BatchSwitchMatrixAVX_Grouped & operator+=(const BatchSwitchMatrixAVX_Grouped& fst){
        *this = *this+fst;
        return *this;
    }
    BatchSwitchMatrixAVX_Grouped & operator-=(const BatchSwitchMatrixAVX_Grouped& fst){
        *this = *this-fst;
        return *this;
    }

    friend BatchSwitchMatrixAVX_Grouped operator+(const BatchSwitchMatrixAVX_Grouped &fst, const Interval & scd){
        BatchSwitchMatrixAVX_Grouped result(true);
        __m256d scalar = _mm256_set1_pd(scd.leftBound());
        __m256d scalar2 = _mm256_set1_pd(scd.rightBound());
        if constexpr (enable_parallel)
        {
            #pragma omp parallel
            {
                capd::rounding::DoubleRounding::roundDown();
                #pragma omp for schedule(static) nowait
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.lower[i] = _mm256_add_pd(fst.lower[i],scalar);
                }
                capd::rounding::DoubleRounding::roundUp();
                #pragma omp for schedule(static)
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.upper[i] = _mm256_add_pd(fst.upper[i],scalar2);
                }
            }
        }
        else
        {
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<result.vectors_count;i++){
                result.lower[i] = _mm256_add_pd(fst.lower[i],scalar);
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 0; i<result.vectors_count;i++){
                result.upper[i] = _mm256_add_pd(fst.upper[i],scalar2);
            }
        }
        return result;
    }
    friend BatchSwitchMatrixAVX_Grouped operator-(const BatchSwitchMatrixAVX_Grouped &fst, const Interval & scd){
        BatchSwitchMatrixAVX_Grouped result(true);
        __m256d scalar = _mm256_set1_pd(scd.rightBound());
        __m256d scalar2 = _mm256_set1_pd(scd.leftBound());
        if constexpr (enable_parallel)
        {
            #pragma omp parallel
            {
                capd::rounding::DoubleRounding::roundDown();
                #pragma omp for schedule(static) nowait
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.lower[i] = _mm256_sub_pd(fst.lower[i],scalar);
                }
                capd::rounding::DoubleRounding::roundUp();
                #pragma omp for schedule(static)
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.upper[i] = _mm256_sub_pd(fst.upper[i],scalar2);
                }
            }
        }
        else
        {
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<result.vectors_count;i++){
                result.lower[i] = _mm256_sub_pd(fst.lower[i],scalar);
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 0; i<result.vectors_count;i++){
                result.upper[i] = _mm256_sub_pd(fst.upper[i],scalar2);
            }
        }
        return result;
    }
    friend BatchSwitchMatrixAVX_Grouped operator*(const BatchSwitchMatrixAVX_Grouped &fst, const Interval & scd){
        char type_scalar = type(scd.leftBound(),scd.rightBound());
        if constexpr(enable_parallel)
        {
            switch(type_scalar)
            {
                case 0:
                    return BatchSwitchMatrixAVX_Grouped ();
                case 1:{
                    __m256d scalar = _mm256_set1_pd(scd.leftBound());
                    BatchSwitchMatrixAVX_Grouped result(true);
                    #pragma omp parallel
                    {
                        capd::rounding::DoubleRounding::roundDown();
                        #pragma omp for schedule(static) nowait
                        for(size_t i = 0; i<result.vectors_count;i++){
                            result.lower[i] = _mm256_mul_pd(fst.lower[i],scalar);
                        }
                        capd::rounding::DoubleRounding::roundUp();
                        #pragma omp for schedule(static)
                        for(size_t i = 0; i<result.vectors_count;i++){
                            result.upper[i] = _mm256_mul_pd(fst.upper[i],scalar);
                        }
                    }
                    return result;
                }
                case 2:{
                    __m256d scalar = _mm256_set1_pd(scd.leftBound());
                    BatchSwitchMatrixAVX_Grouped result(true);
                    #pragma omp parallel
                    {
                        capd::rounding::DoubleRounding::roundDown();
                        #pragma omp for schedule(static) nowait
                        for(size_t i = 0; i<result.vectors_count;i++){
                            result.lower[i] = _mm256_mul_pd(fst.upper[i],scalar);
                        }
                        capd::rounding::DoubleRounding::roundUp();
                        #pragma omp for schedule(static)
                        for(size_t i = 0; i<result.vectors_count;i++){
                        result.upper[i] = _mm256_mul_pd(fst.lower[i],scalar);
                        }
                    }
                    return result;
                }
                case 3:{
                    __m256d scalar = _mm256_set1_pd(scd.rightBound());
                    __m256d zero = _mm256_setzero_pd();
                    BatchSwitchMatrixAVX_Grouped result(true);
                    #pragma omp parallel
                    {
                        capd::rounding::DoubleRounding::roundDown();
                        #pragma omp for schedule(static) nowait
                        for(size_t i = 0; i<result.vectors_count;i++){
                            result.lower[i] = _mm256_min_pd(_mm256_mul_pd(fst.lower[i],scalar),zero);
                        }
                        capd::rounding::DoubleRounding::roundUp();
                        #pragma omp for schedule(static)
                        for(size_t i = 0; i<result.vectors_count;i++){
                            result.upper[i] = _mm256_max_pd(_mm256_mul_pd(fst.upper[i],scalar),zero);
                        }
                    }
                    return result;
                }
                case 4:{
                    __m256d scalar1 = _mm256_set1_pd(scd.leftBound());
                    __m256d scalar2 = _mm256_set1_pd(scd.rightBound());

                    BatchSwitchMatrixAVX_Grouped result(true);
                    #pragma omp parallel
                    {
                        capd::rounding::DoubleRounding::roundDown();
                        #pragma omp for schedule(static) nowait
                        for(size_t i = 0; i<result.vectors_count;i++){
                            result.lower[i] = _mm256_min_pd(_mm256_mul_pd(fst.lower[i],scalar1),_mm256_mul_pd(fst.lower[i],scalar2));
                        }
                        capd::rounding::DoubleRounding::roundUp();
                        #pragma omp for schedule(static)
                        for(size_t i = 0; i<result.vectors_count;i++){
                            result.upper[i] = _mm256_max_pd(_mm256_mul_pd(fst.upper[i],scalar1),_mm256_mul_pd(fst.upper[i],scalar2));
                        }
                    }
                    return result;
                }
                case 5:{
                    __m256d scalar = _mm256_set1_pd(scd.leftBound());
                    __m256d zero = _mm256_setzero_pd();

                    BatchSwitchMatrixAVX_Grouped result(true);

                    #pragma omp parallel
                    {
                        capd::rounding::DoubleRounding::roundDown();
                        #pragma omp for schedule(static) nowait
                        for(size_t i = 0; i<result.vectors_count;i++){
                            result.lower[i] = _mm256_min_pd(_mm256_mul_pd(fst.upper[i],scalar),zero);
                        }
                        capd::rounding::DoubleRounding::roundUp();
                        #pragma omp for schedule(static)
                        for(size_t i = 0; i<result.vectors_count;i++){
                            result.upper[i] = _mm256_max_pd(_mm256_mul_pd(fst.lower[i],scalar),zero);
                        }
                    }
                    return result;
                }
                case 6:{
                    __m256d scalar1 = _mm256_set1_pd(scd.leftBound());
                    __m256d scalar2 = _mm256_set1_pd(scd.rightBound());

                    BatchSwitchMatrixAVX_Grouped result(true);
                    #pragma omp parallel 
                    {
                        capd::rounding::DoubleRounding::roundDown();
                        #pragma omp for schedule(static) nowait
                        for(size_t i = 0; i<result.vectors_count;i++){
                            result.lower[i] = _mm256_min_pd(_mm256_mul_pd(fst.upper[i],scalar1),_mm256_mul_pd(fst.upper[i],scalar2));
                        }
                        capd::rounding::DoubleRounding::roundUp();
                        #pragma omp for schedule(static)
                        for(size_t i = 0; i<result.vectors_count;i++){
                            result.upper[i] = _mm256_max_pd(_mm256_mul_pd(fst.lower[i],scalar1),_mm256_mul_pd(fst.lower[i],scalar2));
                        }
                    }
                    return result;
                }
                case 7:{
                    __m256d scalar1 = _mm256_set1_pd(scd.leftBound());
                    __m256d scalar2 = _mm256_set1_pd(scd.rightBound());

                    BatchSwitchMatrixAVX_Grouped result(true);
                    #pragma omp parallel
                    {
                        capd::rounding::DoubleRounding::roundDown();
                        #pragma omp for schedule(static) nowait
                        for(size_t i = 0; i<result.vectors_count;i++){
                            result.lower[i] = _mm256_min_pd(_mm256_mul_pd(fst.upper[i],scalar1),_mm256_mul_pd(fst.lower[i],scalar2));
                        }
                        capd::rounding::DoubleRounding::roundUp();
                        #pragma omp for schedule(static)
                        for(size_t i = 0; i<result.vectors_count;i++){
                            result.upper[i] = _mm256_max_pd(_mm256_mul_pd(fst.lower[i],scalar1),_mm256_mul_pd(fst.upper[i],scalar2));
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
                    return BatchSwitchMatrixAVX_Grouped ();
                case 1:{
                    __m256d scalar = _mm256_set1_pd(scd.leftBound());
                    BatchSwitchMatrixAVX_Grouped result(true);
                    capd::rounding::DoubleRounding::roundDown();
                    for(size_t i = 0; i<result.vectors_count;i++){
                        result.lower[i] = _mm256_mul_pd(fst.lower[i],scalar);
                    }
                    capd::rounding::DoubleRounding::roundUp();
                    for(size_t i = 0; i<result.vectors_count;i++){
                        result.upper[i] = _mm256_mul_pd(fst.upper[i],scalar);
                    }
                    return result;
                }
                case 2:{
                    __m256d scalar = _mm256_set1_pd(scd.leftBound());
                    BatchSwitchMatrixAVX_Grouped result(true);
                    capd::rounding::DoubleRounding::roundDown();
                    for(size_t i = 0; i<result.vectors_count;i++){
                        result.lower[i] = _mm256_mul_pd(fst.upper[i],scalar);
                    }
                    capd::rounding::DoubleRounding::roundUp();
                    for(size_t i = 0; i<result.vectors_count;i++){
                    result.upper[i] = _mm256_mul_pd(fst.lower[i],scalar);
                    }
                    return result;
                }
                case 3:{
                    __m256d scalar = _mm256_set1_pd(scd.rightBound());
                    __m256d zero = _mm256_setzero_pd();
                    BatchSwitchMatrixAVX_Grouped result(true);
                    capd::rounding::DoubleRounding::roundDown();
                    for(size_t i = 0; i<result.vectors_count;i++){
                        result.lower[i] = _mm256_min_pd(_mm256_mul_pd(fst.lower[i],scalar),zero);
                    }
                    capd::rounding::DoubleRounding::roundUp();
                    for(size_t i = 0; i<result.vectors_count;i++){
                        result.upper[i] = _mm256_max_pd(_mm256_mul_pd(fst.upper[i],scalar),zero);
                    }
                    return result;
                }
                case 4:{
                    __m256d scalar1 = _mm256_set1_pd(scd.leftBound());
                    __m256d scalar2 = _mm256_set1_pd(scd.rightBound());

                    BatchSwitchMatrixAVX_Grouped result(true);

                    capd::rounding::DoubleRounding::roundDown();
                    for(size_t i = 0; i<result.vectors_count;i++){
                        result.lower[i] = _mm256_min_pd(_mm256_mul_pd(fst.lower[i],scalar1),_mm256_mul_pd(fst.lower[i],scalar2));
                    }
                    capd::rounding::DoubleRounding::roundUp();
                    for(size_t i = 0; i<result.vectors_count;i++){
                        result.upper[i] = _mm256_max_pd(_mm256_mul_pd(fst.upper[i],scalar1),_mm256_mul_pd(fst.upper[i],scalar2));
                    }
                    return result;
                }
                case 5:{
                    __m256d scalar = _mm256_set1_pd(scd.leftBound());
                    __m256d zero = _mm256_setzero_pd();

                    BatchSwitchMatrixAVX_Grouped result(true);

                    capd::rounding::DoubleRounding::roundDown();
                    for(size_t i = 0; i<result.vectors_count;i++){
                        result.lower[i] = _mm256_min_pd(_mm256_mul_pd(fst.upper[i],scalar),zero);
                    }
                    capd::rounding::DoubleRounding::roundUp();
                    for(size_t i = 0; i<result.vectors_count;i++){
                        result.upper[i] = _mm256_max_pd(_mm256_mul_pd(fst.lower[i],scalar),zero);
                    }
                    return result;
                }
                case 6:{
                    __m256d scalar1 = _mm256_set1_pd(scd.leftBound());
                    __m256d scalar2 = _mm256_set1_pd(scd.rightBound());

                    BatchSwitchMatrixAVX_Grouped result(true);

                    capd::rounding::DoubleRounding::roundDown();
                    for(size_t i = 0; i<result.vectors_count;i++){
                        result.lower[i] = _mm256_min_pd(_mm256_mul_pd(fst.upper[i],scalar1),_mm256_mul_pd(fst.upper[i],scalar2));
                    }
                    capd::rounding::DoubleRounding::roundUp();
                    for(size_t i = 0; i<result.vectors_count;i++){
                        result.upper[i] = _mm256_max_pd(_mm256_mul_pd(fst.lower[i],scalar1),_mm256_mul_pd(fst.lower[i],scalar2));
                    }
                    return result;
                }
                case 7:{
                    __m256d scalar1 = _mm256_set1_pd(scd.leftBound());
                    __m256d scalar2 = _mm256_set1_pd(scd.rightBound());

                    BatchSwitchMatrixAVX_Grouped result(true);

                    capd::rounding::DoubleRounding::roundDown();
                    for(size_t i = 0; i<result.vectors_count;i++){
                        result.lower[i] = _mm256_min_pd(_mm256_mul_pd(fst.upper[i],scalar1),_mm256_mul_pd(fst.lower[i],scalar2));
                    }
                    capd::rounding::DoubleRounding::roundUp();
                    for(size_t i = 0; i<result.vectors_count;i++){
                        result.upper[i] = _mm256_max_pd(_mm256_mul_pd(fst.lower[i],scalar1),_mm256_mul_pd(fst.upper[i],scalar2));
                    }
                    return result;
                }
            }
        }
        return BatchSwitchMatrixAVX_Grouped ();
    }
    friend BatchSwitchMatrixAVX_Grouped operator/(const BatchSwitchMatrixAVX_Grouped &fst, const Interval & scd){
        char type_scalar = type(scd.leftBound(),scd.rightBound());
        if constexpr(enable_parallel)
        {
            switch (type_scalar)
            {
            case 0:
            case 3:
            case 5:
            case 7:
                throw std::invalid_argument("Scalar has 0 in it");
            case 1:{
                BatchSwitchMatrixAVX_Grouped result(true);
                __m256d scalar = _mm256_set1_pd(scd.leftBound());
                #pragma omp parallel 
                {
                    capd::rounding::DoubleRounding::roundDown();
                    #pragma omp for schedule(static) nowait
                    for(size_t i = 0; i<result.vectors_count;i++){
                        result.lower[i] = _mm256_div_pd(fst.lower[i],scalar);
                    }
                    capd::rounding::DoubleRounding::roundUp();
                    #pragma omp for schedule(static)
                    for(size_t i = 0; i<result.vectors_count;i++){
                        result.upper[i] = _mm256_div_pd(fst.upper[i],scalar);
                    }
                }
                return result;
            }
            case 2:{
                __m256d scalar = _mm256_set1_pd(scd.leftBound());
                BatchSwitchMatrixAVX_Grouped result(true);
                #pragma omp parallel 
                {
                    capd::rounding::DoubleRounding::roundDown();
                    #pragma omp for schedule(static) nowait
                    for(size_t i = 0; i<result.vectors_count;i++){
                        result.lower[i] = _mm256_div_pd(fst.upper[i],scalar);
                    }
                    capd::rounding::DoubleRounding::roundUp();
                    #pragma omp for schedule(static)
                    for(size_t i = 0; i<result.vectors_count;i++){
                        result.upper[i] = _mm256_div_pd(fst.lower[i],scalar);
                    }
                }
                return result;
            }
            case 4:{
                __m256d scalar1 = _mm256_set1_pd(scd.leftBound());
                __m256d scalar2 = _mm256_set1_pd(scd.rightBound());

                BatchSwitchMatrixAVX_Grouped result(true);

                #pragma omp parallel 
                {
                    capd::rounding::DoubleRounding::roundDown();
                    #pragma omp for schedule(static) nowait
                    for(size_t i = 0; i<result.vectors_count;i++){
                        result.lower[i] = _mm256_min_pd(_mm256_div_pd(fst.lower[i],scalar1),_mm256_div_pd(fst.lower[i],scalar2));
                    }
                    capd::rounding::DoubleRounding::roundUp();
                    #pragma omp for schedule(static)
                    for(size_t i = 0; i<result.vectors_count;i++){
                        result.upper[i] = _mm256_max_pd(_mm256_div_pd(fst.upper[i],scalar1),_mm256_div_pd(fst.upper[i],scalar2));
                    }
                }
                return result;
            }
            case 6:{
                __m256d scalar1 = _mm256_set1_pd(scd.leftBound());
                __m256d scalar2 = _mm256_set1_pd(scd.rightBound());

                BatchSwitchMatrixAVX_Grouped result(true);

                #pragma omp parallel 
                {
                    capd::rounding::DoubleRounding::roundDown();
                    #pragma omp for schedule(static) nowait
                    for(size_t i = 0; i<result.vectors_count;i++){
                        result.lower[i] = _mm256_min_pd(_mm256_div_pd(fst.upper[i],scalar1),_mm256_div_pd(fst.upper[i],scalar2));
                    }
                    capd::rounding::DoubleRounding::roundUp();
                    #pragma omp for schedule(static)
                    for(size_t i = 0; i<result.vectors_count;i++){
                        result.upper[i] = _mm256_min_pd(_mm256_div_pd(fst.lower[i],scalar1),_mm256_div_pd(fst.lower[i],scalar2));
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
            case 1:{
                __m256d scalar = _mm256_set1_pd(scd.leftBound());
                BatchSwitchMatrixAVX_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.lower[i] = _mm256_div_pd(fst.lower[i],scalar);
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.upper[i] = _mm256_div_pd(fst.upper[i],scalar);
                }
                return result;
            }
            case 2:{
                __m256d scalar = _mm256_set1_pd(scd.leftBound());
                BatchSwitchMatrixAVX_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.lower[i] = _mm256_div_pd(fst.upper[i],scalar);
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.upper[i] = _mm256_div_pd(fst.lower[i],scalar);
                }
                return result;
            }
            case 4:{
                __m256d scalar1 = _mm256_set1_pd(scd.leftBound());
                __m256d scalar2 = _mm256_set1_pd(scd.rightBound());

                BatchSwitchMatrixAVX_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.lower[i] = _mm256_min_pd(_mm256_div_pd(fst.lower[i],scalar1),_mm256_div_pd(fst.lower[i],scalar2));
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.upper[i] = _mm256_max_pd(_mm256_div_pd(fst.upper[i],scalar1),_mm256_div_pd(fst.upper[i],scalar2));
                }
                return result;
            }
            case 6:{
                __m256d scalar1 = _mm256_set1_pd(scd.leftBound());
                __m256d scalar2 = _mm256_set1_pd(scd.rightBound());

                BatchSwitchMatrixAVX_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.lower[i] = _mm256_min_pd(_mm256_div_pd(fst.upper[i],scalar1),_mm256_div_pd(fst.upper[i],scalar2));
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.upper[i] = _mm256_min_pd(_mm256_div_pd(fst.lower[i],scalar1),_mm256_div_pd(fst.lower[i],scalar2));
                }
                return result;
            }
            }
        }
        return BatchSwitchMatrixAVX_Grouped ();
    }
    BatchSwitchMatrixAVX_Grouped & operator+=(const Interval& fst){
        *this = *this+fst;
        return *this;
    }
    BatchSwitchMatrixAVX_Grouped & operator-=(const Interval& fst){
        *this = *this-fst;
        return *this;
    }
    BatchSwitchMatrixAVX_Grouped & operator*=(const Interval& fst){
        *this = *this * fst;
        return *this;
    }
    BatchSwitchMatrixAVX_Grouped & operator/=(const Interval& fst){
        *this = *this / fst;
        return *this;
    }

    bool operator==(const MatrixBasic<Interval,N,M> & fst){
        for(size_t i = 0; i <N; i++){
            for(size_t j = 0; j < M; j++){
                Interval interval = (*this)(i,j);
                if(interval != fst(i,j)) {
                    std::cout << i << " " << j << std::endl;
                    return false;
                }
            }
        }
        return true;
    }

    friend std::ostream& operator<<(std::ostream& os, const BatchSwitchMatrixAVX_Grouped& matrix){
        for (size_t row = 0; row < N; ++row) {
        os << "Row " << row << ": ";
        for (size_t vec = 0; vec < matrix.vectors_count_row; ++vec) {
            
            size_t index = row * matrix.vectors_count_row + vec;

            alignas(32) double elements[4];
            _mm256_store_pd(elements, matrix.lower[index]);

            os << "[";
            for (size_t i = 0; i < 4; ++i) {
                os << elements[i];
                if (i < 3) os << ", ";
            }
            os << "]";
            if (vec < matrix.vectors_count_row - 1) os << " ";
        }
        os << "\n";
    }
    return os;
    }

    IntervalProxy<Accessor,Index> operator()(size_t j, size_t k){
        Index ind(j*vectors_count_row+(k/4),k%4);
        Accessor acc = {lower,upper};
        return IntervalProxy<Accessor,Index>(acc,ind);
    }
    IntervalProxy<Accessor,Index> operator()(size_t j, size_t k) const{
        Index ind(j*vectors_count_row+(k/4),k%4);
        Accessor acc = {lower,upper};
        return IntervalProxy<Accessor,Index>(acc,ind);
    }
   
    ~BatchSwitchMatrixAVX_Grouped(){
        if(lower) delete[] lower;
        if(upper) delete[] upper;
    }
};
namespace details{
    inline void transpose4x4(const __m256d& row0, const __m256d& row1, const __m256d& row2, const __m256d& row3,
                  __m256d& out0, __m256d& out1, __m256d& out2, __m256d& out3) {
        // Interleave low and high elements of row0 and row1
        __m256d t0 = _mm256_unpacklo_pd(row0, row1); // [row0[0], row1[0], row0[2], row1[2]]
        __m256d t1 = _mm256_unpackhi_pd(row0, row1); // [row0[1], row1[1], row0[3], row1[3]]
        __m256d t2 = _mm256_unpacklo_pd(row2, row3); // [row2[0], row3[0], row2[2], row3[2]]
        __m256d t3 = _mm256_unpackhi_pd(row2, row3); // [row2[1], row3[1], row2[3], row3[3]]

        // Shuffle and combine rows to get the final transposed vectors
        out0 = _mm256_permute2f128_pd(t0, t2, 0x20); // [row0[0], row1[0], row2[0], row3[0]]
        out1 = _mm256_permute2f128_pd(t1, t3, 0x20); // [row0[1], row1[1], row2[1], row3[1]]
        out2 = _mm256_permute2f128_pd(t0, t2, 0x31); // [row0[2], row1[2], row2[2], row3[2]]
        out3 = _mm256_permute2f128_pd(t1, t3, 0x31); // [row0[3], row1[3], row2[3], row3[3]]
    }

    inline void transpose3x4(const __m256d& row0, const __m256d& row1, const __m256d& row2,
                  __m256d& out0, __m256d& out1, __m256d& out2, __m256d& out3) {
        __m256d zero = _mm256_setzero_pd();
        // Interleave low and high elements of row0 and row1
        __m256d t0 = _mm256_unpacklo_pd(row0, row1); // [row0[0], row1[0], row0[2], row1[2]]
        __m256d t1 = _mm256_unpackhi_pd(row0, row1); // [row0[1], row1[1], row0[3], row1[3]]
        __m256d t2 = _mm256_unpacklo_pd(row2, zero); // [row2[0], 0, row2[2], 0]
        __m256d t3 = _mm256_unpackhi_pd(row2, zero); // [row2[1], 0, row2[3], 0]

        // Shuffle and combine rows to get the final transposed vectors
        out0 = _mm256_permute2f128_pd(t0, t2, 0x20); // [row0[0], row1[0], row2[0], row3[0]]
        out1 = _mm256_permute2f128_pd(t1, t3, 0x20); // [row0[1], row1[1], row2[1], row3[1]]
        out2 = _mm256_permute2f128_pd(t0, t2, 0x31); // [row0[2], row1[2], row2[2], row3[2]]
        out3 = _mm256_permute2f128_pd(t1, t3, 0x31); // [row0[3], row1[3], row2[3], row3[3]]
    }

    inline void transpose2x4(const __m256d& row0, const __m256d& row1,
                  __m256d& out0, __m256d& out1, __m256d& out2, __m256d& out3) {
        __m256d zero = _mm256_setzero_pd();
        // Interleave low and high elements of row0 and row1
        __m256d t0 = _mm256_unpacklo_pd(row0, row1); // [row0[0], row1[0], row0[2], row1[2]]
        __m256d t1 = _mm256_unpackhi_pd(row0, row1); // [row0[1], row1[1], row0[3], row1[3]]

        // Shuffle and combine rows to get the final transposed vectors
        out0 = _mm256_permute2f128_pd(t0, zero, 0x20); // [row0[0], row1[0], row2[0], row3[0]]
        out1 = _mm256_permute2f128_pd(t1, zero, 0x20); // [row0[1], row1[1], row2[1], row3[1]]
        out2 = _mm256_permute2f128_pd(t0, zero, 0x31); // [row0[2], row1[2], row2[2], row3[2]]
        out3 = _mm256_permute2f128_pd(t1, zero, 0x31); // [row0[3], row1[3], row2[3], row3[3]]
    }

    inline void transpose1x4(const __m256d& row0,
                  __m256d& out0, __m256d& out1, __m256d& out2, __m256d& out3) {
        __m256d zero = _mm256_setzero_pd();
        // Interleave low and high elements of row0 and row1
        __m256d t0 = _mm256_unpacklo_pd(row0, zero); // [row0[0], 0, row0[2], 0]
        __m256d t1 = _mm256_unpackhi_pd(row0, zero); // [row0[1], 0, row0[3], 0]

        // Shuffle and combine rows to get the final transposed vectors
        out0 = _mm256_permute2f128_pd(t0, zero, 0x20); // [row0[0], row1[0], row2[0], row3[0]]
        out1 = _mm256_permute2f128_pd(t1, zero, 0x20); // [row0[1], row1[1], row2[1], row3[1]]
        out2 = _mm256_permute2f128_pd(t0, zero, 0x31); // [row0[2], row1[2], row2[2], row3[2]]
        out3 = _mm256_permute2f128_pd(t1, zero, 0x31); // [row0[3], row1[3], row2[3], row3[3]]
    }

    inline void transpose4x3(const __m256d& col0, const __m256d& col1, const __m256d& col2, const __m256d& col3,
                         __m256d& out0, __m256d& out1, __m256d& out2) {
    // Interleave low and high elements of col0 and col1
    __m256d t0 = _mm256_unpacklo_pd(col0, col1); // [col0[0], col1[0], col0[2], col1[2]]
    __m256d t1 = _mm256_unpackhi_pd(col0, col1); // [col0[1], col1[1], col0[3], col1[3]]
    __m256d t2 = _mm256_unpacklo_pd(col2, col3); // [col2[0], col3[0], col2[2], col3[2]]
    __m256d t3 = _mm256_unpackhi_pd(col2, col3); // [col2[1], col3[1], col2[3], col3[3]]

    // Combine rows to produce the transposed output
    out0 = _mm256_permute2f128_pd(t0, t2, 0x20); // [col0[0], col1[0], col2[0], col3[0]]
    out1 = _mm256_permute2f128_pd(t1, t3, 0x20); // [col0[1], col1[1], col2[1], col3[1]]
    out2 = _mm256_permute2f128_pd(t0, t2, 0x31); // [col0[2], col1[2], col2[2], col3[2]]
}

inline void transpose4x2(const __m256d& row0, const __m256d& row1,
                         const __m256d& row2, const __m256d& row3,
                         __m256d& out0, __m256d& out1) {
    // Łączenie par elementów z wierszy
    __m256d t0 = _mm256_unpacklo_pd(row0, row1); // [row0[0], row1[0], row0[2], row1[2]]
    __m256d t1 = _mm256_unpackhi_pd(row0, row1); // [row0[1], row1[1], row0[3], row1[3]]
    __m256d t2 = _mm256_unpacklo_pd(row2, row3); // [row2[0], row3[0], row2[2], row3[2]]
    __m256d t3 = _mm256_unpackhi_pd(row2, row3); // [row2[1], row3[1], row2[3], row3[3]]

    // Łączenie wyników, aby uzyskać 2 wiersze po transpozycji
    out0 = _mm256_permute2f128_pd(t0, t2, 0x20); // [row0[0], row1[0], row2[0], row3[0]]
    out1 = _mm256_permute2f128_pd(t1, t3, 0x20); // [row0[1], row1[1], row2[1], row3[1]]
}

inline void transpose4x1(const __m256d& row0, const __m256d& row1,
                         const __m256d& row2, const __m256d& row3,
                         __m256d& out0) {
    // Łączenie par elementów z wierszy
    __m256d t0 = _mm256_unpacklo_pd(row0, row1); // [row0[0], row1[0], row0[2], row1[2]]
    __m256d t2 = _mm256_unpacklo_pd(row2, row3); // [row2[0], row3[0], row2[2], row3[2]]

    // Łączenie wyników
    out0 = _mm256_permute2f128_pd(t0, t2, 0x20); // [row0[0], row1[0], row2[0], row3[0]]
}

}
template<size_t N, size_t M>
BatchSwitchMatrixAVX_Grouped<N,M> reorg(const BatchSwitchMatrixAVX_Grouped<N,M>& fst){
    BatchSwitchMatrixAVX_Grouped<N,M> result(true);
    size_t full1 = fst.full_vectors;
    size_t full2 = result.full_vectors;
    size_t offset = result.vectors_count / 4;
    for (size_t j = 0; j < N; j += 4) {
        for (size_t i = 0; i < full1; i++) {
            details::transpose4x4(
                fst.upper[full1 * j + i],
                fst.upper[full1 * (j + 1) + i],
                fst.upper[full1 * (j + 2) + i],
                fst.upper[full1 * (j + 3) + i],
                result.upper[full2 * i + j/4],
                result.upper[full2 * i + offset + j/4],
                result.upper[full2 * i + offset*2 + j/4],
                result.upper[full2 * i + offset*3 + j/4]
            );
            details::transpose4x4(
                fst.lower[full1 * j + i],
                fst.lower[full1 * (j + 1) + i],
                fst.lower[full1 * (j + 2) + i],
                fst.lower[full1 * (j + 3) + i],
                result.lower[full2 * i + j/4],
                result.lower[full2 * i + offset + j/4],
                result.lower[full2 * i + offset*2 + j/4],
                result.lower[full2 * i + offset*3 + j/4]
            );
        }
    }
    return result;   
}


#endif