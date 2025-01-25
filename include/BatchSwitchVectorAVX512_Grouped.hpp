#ifndef VECTOR_BATCH_AVX512_GRP_HPP
#define VECTOR_BATCH_AVX512_GRP_HPP

#include <iostream>
#include <capd/intervals/Interval.hpp>
#include <omp.h>
#include <immintrin.h>
#include <Utilities.hpp>
#include <IntervalProxy.hpp>

template<size_t N>
class BatchSwitchVectorAVX512_Grouped
{
private:
    __m512d* lower = nullptr;
    __m512d* upper = nullptr;
    typedef capd::intervals::Interval<double> Interval;
    static constexpr size_t vectors_count = (N + 7) / 8; // Number of 512-bit vectors
    static constexpr size_t full_vectors = N / 8;        // Full vectors
    static constexpr size_t rest = N % 8;               // Remaining elements

    BatchSwitchVectorAVX512_Grouped(bool allocateOnly)
    {
        if (allocateOnly) {
            lower = new alignas(64) __m512d[vectors_count];
            upper = new alignas(64) __m512d[vectors_count];
        } else {
            lower = nullptr;
            upper = nullptr;
        }
    }
    struct Index{
        size_t ind;
        uint8_t poz;
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
            upp_val = _mm512_cvtsd_f64(_mm512_permutex_pd(vec_lower,0b00000011));
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

    BatchSwitchVectorAVX512_Grouped() {
        lower = new alignas(64) __m512d[vectors_count]();
        upper = new alignas(64) __m512d[vectors_count]();
    }

    BatchSwitchVectorAVX512_Grouped(const BatchSwitchVectorAVX512_Grouped& cpy) {
        lower = new alignas(64) __m512d[vectors_count];
        upper = new alignas(64) __m512d[vectors_count];
        for (size_t i = 0; i < vectors_count; i++) {
            lower[i] = cpy.lower[i];
            upper[i] = cpy.upper[i];
        }
    }

    BatchSwitchVectorAVX512_Grouped(BatchSwitchVectorAVX512_Grouped&& cpy) noexcept {
        lower = cpy.lower;
        upper = cpy.upper;
        cpy.lower = nullptr;
        cpy.upper = nullptr;
    }

    BatchSwitchVectorAVX512_Grouped& operator=(const BatchSwitchVectorAVX512_Grouped& fst) {
        if (this != &fst) {
            delete[] lower;
            delete[] upper;
            if (fst.lower) {
                lower = new __m512d[vectors_count];
                upper = new __m512d[vectors_count];
                for (size_t i = 0; i < vectors_count; ++i) {
                    lower[i] = fst.lower[i];
                    upper[i] = fst.upper[i];
                }
            } else {
                lower = nullptr;
                upper = nullptr;
            }
        }
        return *this;
    }

    BatchSwitchVectorAVX512_Grouped& operator=(BatchSwitchVectorAVX512_Grouped&& fst) noexcept {
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

    BatchSwitchVectorAVX512_Grouped(const double(&array)[2 * N]) {
        lower = new alignas(64) __m512d[vectors_count];
        upper = new alignas(64) __m512d[vectors_count];

        for (size_t i = 0; i < full_vectors * 16; i += 16) {
            __m512d a = _mm512_set_pd(array[i + 14], array[i + 12], array[i + 10], array[i + 8], array[i + 6], array[i + 4], array[i + 2], array[i]);
            __m512d b = _mm512_set_pd(array[i + 15], array[i + 13], array[i + 11], array[i + 9], array[i + 7], array[i + 5], array[i + 3], array[i + 1]);
            lower[i / 16] = a;
            upper[i / 16] = b;
        }

        if (rest > 0) {
            double temp_lower[8] = {};
            double temp_upper[8] = {};
            for (size_t i = 0; i < rest; ++i) {
                temp_lower[i] = array[2 * N - 2 * rest + 2 * i];
                temp_upper[i] = array[2 * N - 2 * rest + 2 * i + 1];
            }
            lower[vectors_count - 1] = _mm512_load_pd(temp_lower);
            upper[vectors_count - 1] = _mm512_load_pd(temp_upper);
        }
    }

    BatchSwitchVectorAVX512_Grouped(const std::vector<double>& array) {
        if (array.size() != 2 * N) {
            throw std::invalid_argument("Rozmiar wektora musi wynosić dokładnie 2 * N.");
        }
        lower = new alignas(64) __m512d[vectors_count];
        upper = new alignas(64) __m512d[vectors_count];

        for (size_t i = 0; i < full_vectors * 16; i += 16) {
            __m512d a = _mm512_set_pd(array[i + 14], array[i + 12], array[i + 10], array[i + 8], array[i + 6], array[i + 4], array[i + 2], array[i]);
            __m512d b = _mm512_set_pd(array[i + 15], array[i + 13], array[i + 11], array[i + 9], array[i + 7], array[i + 5], array[i + 3], array[i + 1]);
            lower[i / 16] = a;
            upper[i / 16] = b;
        }

        if (rest > 0) {
            double temp_lower[8] = {};
            double temp_upper[8] = {};
            for (size_t i = 0; i < rest; ++i) {
                temp_lower[i] = array[2 * N - 2 * rest + 2 * i];
                temp_upper[i] = array[2 * N - 2 * rest + 2 * i + 1];
            }
            lower[vectors_count - 1] = _mm512_load_pd(temp_lower);
            upper[vectors_count - 1] = _mm512_load_pd(temp_upper);
        }
    }

    friend BatchSwitchVectorAVX512_Grouped operator+(const BatchSwitchVectorAVX512_Grouped& fst, const BatchSwitchVectorAVX512_Grouped& scd) {
        BatchSwitchVectorAVX512_Grouped result(true);
        capd::rounding::DoubleRounding::roundDown();
        for (size_t i = 0; i < result.vectors_count; i++) {
            result.lower[i] = _mm512_add_pd(fst.lower[i], scd.lower[i]);
        }

        capd::rounding::DoubleRounding::roundUp();
        for (size_t i = 0; i < result.vectors_count; i++) {
            result.upper[i] = _mm512_add_pd(fst.upper[i], scd.upper[i]);
        }
        return result;
    }

    friend BatchSwitchVectorAVX512_Grouped operator-(const BatchSwitchVectorAVX512_Grouped& fst, const BatchSwitchVectorAVX512_Grouped& scd) {
        BatchSwitchVectorAVX512_Grouped result(true);
        capd::rounding::DoubleRounding::roundDown();
        for (size_t i = 0; i < result.vectors_count; i++) {
            result.lower[i] = _mm512_sub_pd(fst.lower[i], scd.upper[i]);
        }

        capd::rounding::DoubleRounding::roundUp();
        for (size_t i = 0; i < result.vectors_count; i++) {
            result.upper[i] = _mm512_sub_pd(fst.upper[i], scd.lower[i]);
        }
        return result;
    }


    BatchSwitchVectorAVX512_Grouped & operator+=(const BatchSwitchVectorAVX512_Grouped& fst){
        *this = *this+fst;
        return *this;
    }
    BatchSwitchVectorAVX512_Grouped & operator-=(const BatchSwitchVectorAVX512_Grouped& fst){
        *this = *this-fst;
        return *this;
    }

    friend BatchSwitchVectorAVX512_Grouped operator+(const BatchSwitchVectorAVX512_Grouped &fst, const Interval & scd){
        BatchSwitchVectorAVX512_Grouped result(true);
        __m512d scalar = _mm512_set1_pd(scd.leftBound());
        capd::rounding::DoubleRounding::roundDown();
        for(size_t i = 0; i<result.vectors_count;i++){
            result.lower[i] = _mm512_add_pd(fst.lower[i],scalar);
        }
        scalar = _mm512_set1_pd(scd.rightBound());
        capd::rounding::DoubleRounding::roundUp();
         for(size_t i = 0; i<result.vectors_count;i++){
            result.upper[i] = _mm512_add_pd(fst.upper[i],scalar);
        }
        return result;
    }
    friend BatchSwitchVectorAVX512_Grouped operator-(const BatchSwitchVectorAVX512_Grouped &fst, const Interval & scd){
        BatchSwitchVectorAVX512_Grouped result(true);
        __m512d scalar = _mm512_set1_pd(scd.rightBound());
        capd::rounding::DoubleRounding::roundDown();
        for(size_t i = 0; i<result.vectors_count;i++){
            result.lower[i] = _mm512_sub_pd(fst.lower[i],scalar);
        }
        scalar = _mm512_set1_pd(scd.leftBound());
        capd::rounding::DoubleRounding::roundUp();
         for(size_t i = 0; i<result.vectors_count;i++){
            result.upper[i] = _mm512_sub_pd(fst.upper[i],scalar);
        }
        return result;
    }
    friend BatchSwitchVectorAVX512_Grouped operator*(const BatchSwitchVectorAVX512_Grouped &fst, const Interval & scd){
        char type_scalar = type(scd.leftBound(),scd.rightBound());
        switch(type_scalar)
        {
            case 0:
                return BatchSwitchVectorAVX512_Grouped ();
            case 1:{
                __m512d scalar = _mm512_set1_pd(scd.leftBound());
                BatchSwitchVectorAVX512_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                 for(size_t i = 0; i<result.vectors_count;i++){
                    result.lower[i] = _mm512_mul_pd(fst.lower[i],scalar);
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.upper[i] = _mm512_mul_pd(fst.upper[i],scalar);
                }
                return result;
            }
            case 2:{
                __m512d scalar = _mm512_set1_pd(scd.leftBound());
                BatchSwitchVectorAVX512_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.lower[i] = _mm512_mul_pd(fst.upper[i],scalar);
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<result.vectors_count;i++){
                result.upper[i] = _mm512_mul_pd(fst.lower[i],scalar);
                }
                return result;
            }
            case 3:{
                __m512d scalar = _mm512_set1_pd(scd.rightBound());
                __m512d zero = _mm512_setzero_pd();
                BatchSwitchVectorAVX512_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.lower[i] = _mm512_min_pd(_mm512_mul_pd(fst.lower[i],scalar),zero);
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.upper[i] = _mm512_max_pd(_mm512_mul_pd(fst.upper[i],scalar),zero);
                }
                return result;
            }
            case 4:{
                __m512d scalar1 = _mm512_set1_pd(scd.leftBound());
                __m512d scalar2 = _mm512_set1_pd(scd.rightBound());

                BatchSwitchVectorAVX512_Grouped result(true);

                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.lower[i] = _mm512_min_pd(_mm512_mul_pd(fst.lower[i],scalar1),_mm512_mul_pd(fst.lower[i],scalar2));
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.upper[i] = _mm512_max_pd(_mm512_mul_pd(fst.upper[i],scalar1),_mm512_mul_pd(fst.upper[i],scalar2));
                }
                return result;
            }
            case 5:{
                __m512d scalar = _mm512_set1_pd(scd.leftBound());
                __m512d zero = _mm512_setzero_pd();

                BatchSwitchVectorAVX512_Grouped result(true);

                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.lower[i] = _mm512_min_pd(_mm512_mul_pd(fst.upper[i],scalar),zero);
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<result.vectors_count;i++){
                     result.upper[i] = _mm512_max_pd(_mm512_mul_pd(fst.lower[i],scalar),zero);
                }
                return result;
            }
            case 6:{
                __m512d scalar1 = _mm512_set1_pd(scd.leftBound());
                __m512d scalar2 = _mm512_set1_pd(scd.rightBound());

                BatchSwitchVectorAVX512_Grouped result(true);

                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.lower[i] = _mm512_min_pd(_mm512_mul_pd(fst.upper[i],scalar1),_mm512_mul_pd(fst.upper[i],scalar2));
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.upper[i] = _mm512_max_pd(_mm512_mul_pd(fst.lower[i],scalar1),_mm512_mul_pd(fst.lower[i],scalar2));
                }
                return result;
            }
            case 7:{
                __m512d scalar1 = _mm512_set1_pd(scd.leftBound());
                __m512d scalar2 = _mm512_set1_pd(scd.rightBound());

                BatchSwitchVectorAVX512_Grouped result(true);

                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.lower[i] = _mm512_min_pd(_mm512_mul_pd(fst.upper[i],scalar1),_mm512_mul_pd(fst.lower[i],scalar2));
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<result.vectors_count;i++){
                    result.upper[i] = _mm512_max_pd(_mm512_mul_pd(fst.lower[i],scalar1),_mm512_mul_pd(fst.upper[i],scalar2));
                }
                return result;
            }
        }
        return BatchSwitchVectorAVX512_Grouped ();
    }
    friend BatchSwitchVectorAVX512_Grouped operator/(const BatchSwitchVectorAVX512_Grouped &fst, const Interval & scd){
        char type_scalar = type(scd.leftBound(),scd.rightBound());
        switch (type_scalar)
        {
        case 0:
        case 3:
        case 5:
        case 7:
            throw std::invalid_argument("Scalar has 0 in it");
        case 1:{
            __m512d scalar = _mm512_set1_pd(scd.leftBound());
            BatchSwitchVectorAVX512_Grouped result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<result.vectors_count;i++){
                result.lower[i] = _mm512_div_pd(fst.lower[i],scalar);
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 0; i<result.vectors_count;i++){
                result.upper[i] = _mm512_div_pd(fst.upper[i],scalar);
            }
            return result;
        }
        case 2:{
            __m512d scalar = _mm512_set1_pd(scd.leftBound());
            BatchSwitchVectorAVX512_Grouped result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<result.vectors_count;i++){
                result.lower[i] = _mm512_div_pd(fst.upper[i],scalar);
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 0; i<result.vectors_count;i++){
                result.upper[i] = _mm512_div_pd(fst.lower[i],scalar);
            }
            return result;
        }
        case 4:{
            __m512d scalar1 = _mm512_set1_pd(scd.leftBound());
            __m512d scalar2 = _mm512_set1_pd(scd.rightBound());

            BatchSwitchVectorAVX512_Grouped result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<result.vectors_count;i++){
                result.lower[i] = _mm512_min_pd(_mm512_div_pd(fst.lower[i],scalar1),_mm512_div_pd(fst.lower[i],scalar2));
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 0; i<result.vectors_count;i++){
                result.upper[i] = _mm512_max_pd(_mm512_div_pd(fst.upper[i],scalar1),_mm512_div_pd(fst.upper[i],scalar2));
            }
            return result;
        }
        case 6:{
            __m512d scalar1 = _mm512_set1_pd(scd.leftBound());
            __m512d scalar2 = _mm512_set1_pd(scd.rightBound());

            BatchSwitchVectorAVX512_Grouped result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<result.vectors_count;i++){
                result.lower[i] = _mm512_min_pd(_mm512_div_pd(fst.upper[i],scalar1),_mm512_div_pd(fst.upper[i],scalar2));
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 0; i<result.vectors_count;i++){
                result.upper[i] = _mm512_min_pd(_mm512_div_pd(fst.lower[i],scalar1),_mm512_div_pd(fst.lower[i],scalar2));
            }
            return result;
        }
        }
        return BatchSwitchVectorAVX512_Grouped ();
    }
    BatchSwitchVectorAVX512_Grouped & operator+=(const Interval& fst){
        *this = *this+fst;
        return *this;
    }
    BatchSwitchVectorAVX512_Grouped & operator-=(const Interval& fst){
        *this = *this-fst;
        return *this;
    }
    BatchSwitchVectorAVX512_Grouped & operator*=(const Interval& fst){
        *this = *this * fst;
        return *this;
    }
    BatchSwitchVectorAVX512_Grouped & operator/=(const Interval& fst){
        *this = *this / fst;
        return *this;
    }

    ~BatchSwitchVectorAVX512_Grouped() {
        delete[] lower;
        delete[] upper;
    }
};

#endif
