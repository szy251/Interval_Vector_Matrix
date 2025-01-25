#ifndef VECTOR_BATCH_AVX_GRP_HPP
#define VECTOR_BATCH_AVX_GRP_HPP

#include <iostream>
#include <capd/intervals/Interval.hpp>
#include <Utilities.hpp>
#include <IntervalProxy.hpp>
#include <immintrin.h>

template<size_t N>
class BatchSwitchVectorAVX_Grouped
{
private:
    __m256d* lower = nullptr;
    __m256d* upper = nullptr;
    typedef capd::intervals::Interval<double> Interval;
    size_t vectors_count = (N+3)/4;
    size_t full_vectors = N/4;
    size_t rest = N % 4;

     BatchSwitchVectorAVX_Grouped(bool allocateOnly)
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
        __m256d* lower,upper;
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
    BatchSwitchVectorAVX_Grouped(){
        lower = new alignas(32) __m256d[vectors_count]();
        upper = new alignas(32) __m256d[vectors_count]();
    }

    BatchSwitchVectorAVX_Grouped(const BatchSwitchVectorAVX_Grouped& cpy){
        lower = new alignas(32) __m256d[vectors_count];
        upper = new alignas(32) __m256d[vectors_count];
        for(size_t i = 0; i < vectors_count; i++) {
            lower[i] = cpy.lower[i];
            upper[i] = cpy.upper[i];
        }
    }

    BatchSwitchVectorAVX_Grouped(BatchSwitchVectorAVX_Grouped&& cpy) noexcept {
        lower = cpy.lower;
        upper = cpy.upper;
        cpy.lower = nullptr;
        cpy.upper = nullptr;
    }

    BatchSwitchVectorAVX_Grouped(const double(&array)[2 * N]){
        lower = new alignas(32) __m256d[vectors_count];
        upper = new alignas(32) __m256d[vectors_count];

        for(size_t i = 0; i < full_vectors * 8; i += 8){
            __m256d a = _mm256_set_pd(array[i+6], array[i+4], array[i+2], array[i]);
            __m256d b = _mm256_set_pd(array[i+7], array[i+5], array[i+3], array[i+1]);
            lower[i/8] = a;
            upper[i/8] = b;
        }

        if (rest > 0) {
            double temp_lower[4] = {};
            double temp_upper[4] = {};
            for (size_t i = 0; i < rest; ++i) {
                temp_lower[i] = array[2 * N - 2 * rest + 2 * i];
                temp_upper[i] = array[2 * N - 2 * rest + 2 * i + 1];
            }
            lower[vectors_count - 1] = _mm256_load_pd(temp_lower);
            upper[vectors_count - 1] = _mm256_load_pd(temp_upper);
        }
    }

    BatchSwitchVectorAVX_Grouped(const std::vector<double>& array) {
        if (array.size() != 2 * N) {
            throw std::invalid_argument("Rozmiar wektora musi wynosić dokładnie 2 * N.");
        }
        lower = new alignas(32) __m256d[vectors_count];
        upper = new alignas(32) __m256d[vectors_count];

        for(size_t i = 0; i < full_vectors * 8; i += 8){
            __m256d a = _mm256_set_pd(array[i+6], array[i+4], array[i+2], array[i]);
            __m256d b = _mm256_set_pd(array[i+7], array[i+5], array[i+3], array[i+1]);
            lower[i/8] = a;
            upper[i/8] = b;
        }

        if (rest > 0) {
            double temp_lower[4] = {};
            double temp_upper[4] = {};
            for (size_t i = 0; i < rest; ++i) {
                temp_lower[i] = array[2 * N - 2 * rest + 2 * i];
                temp_upper[i] = array[2 * N - 2 * rest + 2 * i + 1];
            }
            lower[vectors_count - 1] = _mm256_load_pd(temp_lower);
            upper[vectors_count - 1] = _mm256_load_pd(temp_upper);
        }
    }

    BatchSwitchVectorAVX_Grouped& operator=(const BatchSwitchVectorAVX_Grouped& fst) {
        if (this != &fst) {
            delete[] lower;
            delete[] upper;
            if (fst.lower) {
                lower = new alignas(32) __m256d[vectors_count];
                upper = new alignas(32) __m256d[vectors_count];
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

    BatchSwitchVectorAVX_Grouped& operator=(BatchSwitchVectorAVX_Grouped&& fst) noexcept {
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

   

    friend BatchSwitchVectorAVX_Grouped operator+(const BatchSwitchVectorAVX_Grouped& fst, const BatchSwitchVectorAVX_Grouped& scd){
        BatchSwitchVectorAVX_Grouped result(true);
        capd::rounding::DoubleRounding::roundDown();

        for(size_t i = 0; i < result.vectors_count; i++) {
            result.lower[i] = _mm256_add_pd(fst.lower[i], scd.lower[i]);
        }

        capd::rounding::DoubleRounding::roundUp();
         for(size_t i = 0; i < result.vectors_count; i++) {
            result.upper[i] = _mm256_add_pd(fst.upper[i], scd.upper[i]);
        }

        return result;
    }

    friend BatchSwitchVectorAVX_Grouped operator-(const BatchSwitchVectorAVX_Grouped& fst, const BatchSwitchVectorAVX_Grouped& scd){
        BatchSwitchVectorAVX_Grouped result(true);
        capd::rounding::DoubleRounding::roundDown();

        for(size_t i = 0; i < result.vectors_count; i++) {
            result.lower[i] = _mm256_sub_pd(fst.lower[i], scd.upper[i]);
        }



        capd::rounding::DoubleRounding::roundUp();
        for(size_t i = 0; i < result.vectors_count; i++) {
            result.upper[i] = _mm256_sub_pd(fst.upper[i], scd.lower[i]);
        }
        return result;
    }

    BatchSwitchVectorAVX_Grouped & operator+=(const BatchSwitchVectorAVX_Grouped& fst){
        *this = *this+fst;
        return *this;
    }
    BatchSwitchVectorAVX_Grouped & operator-=(const BatchSwitchVectorAVX_Grouped& fst){
        *this = *this-fst;
        return *this;
    }

    friend BatchSwitchVectorAVX_Grouped operator+(const BatchSwitchVectorAVX_Grouped &fst, const Interval & scd){
        BatchSwitchVectorAVX_Grouped result(true);
        __m256d scalar = _mm256_set1_pd(scd.leftBound());
        capd::rounding::DoubleRounding::roundDown();
        for(size_t i = 0; i<result.vectors_count;i++){
            result.lower[i] = _mm256_add_pd(fst.lower[i],scalar);
        }
        scalar = _mm256_set1_pd(scd.rightBound());
        capd::rounding::DoubleRounding::roundUp();
         for(size_t i = 0; i<result.vectors_count;i++){
            result.upper[i] = _mm256_add_pd(fst.upper[i],scalar);
        }
        return result;
    }
    friend BatchSwitchVectorAVX_Grouped operator-(const BatchSwitchVectorAVX_Grouped &fst, const Interval & scd){
        BatchSwitchVectorAVX_Grouped result(true);
        __m256d scalar = _mm256_set1_pd(scd.rightBound());
        capd::rounding::DoubleRounding::roundDown();
        for(size_t i = 0; i<result.vectors_count;i++){
            result.lower[i] = _mm256_sub_pd(fst.lower[i],scalar);
        }
        scalar = _mm256_set1_pd(scd.leftBound());
        capd::rounding::DoubleRounding::roundUp();
         for(size_t i = 0; i<result.vectors_count;i++){
            result.upper[i] = _mm256_sub_pd(fst.upper[i],scalar);
        }
        return result;
    }
    friend BatchSwitchVectorAVX_Grouped operator*(const BatchSwitchVectorAVX_Grouped &fst, const Interval & scd){
        char type_scalar = type(scd.leftBound(),scd.rightBound());
        switch(type_scalar)
        {
            case 0:
                return BatchSwitchVectorAVX_Grouped ();
            case 1:{
                __m256d scalar = _mm256_set1_pd(scd.leftBound());
                BatchSwitchVectorAVX_Grouped result(true);
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
                BatchSwitchVectorAVX_Grouped result(true);
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
                BatchSwitchVectorAVX_Grouped result(true);
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

                BatchSwitchVectorAVX_Grouped result(true);

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

                BatchSwitchVectorAVX_Grouped result(true);

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

                BatchSwitchVectorAVX_Grouped result(true);

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

                BatchSwitchVectorAVX_Grouped result(true);

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
        return BatchSwitchVectorAVX_Grouped ();
    }
    friend BatchSwitchVectorAVX_Grouped operator/(const BatchSwitchVectorAVX_Grouped &fst, const Interval & scd){
        char type_scalar = type(scd.leftBound(),scd.rightBound());
        switch (type_scalar)
        {
        case 0:
        case 3:
        case 5:
        case 7:
            throw std::invalid_argument("Scalar has 0 in it");
        case 1:{
            __m256d scalar = _mm256_set1_pd(scd.leftBound());
            BatchSwitchVectorAVX_Grouped result(true);
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
            BatchSwitchVectorAVX_Grouped result(true);
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

            BatchSwitchVectorAVX_Grouped result(true);
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

            BatchSwitchVectorAVX_Grouped result(true);
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
        return BatchSwitchVectorAVX_Grouped ();
    }
    BatchSwitchVectorAVX_Grouped & operator+=(const Interval& fst){
        *this = *this+fst;
        return *this;
    }
    BatchSwitchVectorAVX_Grouped & operator-=(const Interval& fst){
        *this = *this-fst;
        return *this;
    }
    BatchSwitchVectorAVX_Grouped & operator*=(const Interval& fst){
        *this = *this * fst;
        return *this;
    }
    BatchSwitchVectorAVX_Grouped & operator/=(const Interval& fst){
        *this = *this / fst;
        return *this;
    }
    IntervalProxy<Accessor, Index> operator[](size_t i){
        Index index{i/4,i%4};
        Accessor acc{lower,upper};
        return IntervalProxy<Accessor,Index>(acc,index);
    }

    ~BatchSwitchVectorAVX_Grouped(){
        delete[] lower;
        delete[] upper;
    }
};

#endif
