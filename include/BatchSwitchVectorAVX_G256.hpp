#ifndef VECTOR_BATCH_AVX_HPP
#define VECTOR_BATCH_AVX_HPP

#include <iostream>
#include <capd/rounding/DoubleRounding.h>
#include <capd/filib/Interval.h>
#include <immintrin.h>
#include <Utilities.hpp>
#include <IntervalProxy.hpp>

template<size_t N>
class BatchSwitchVectorAVX_G256
{
private:
    __m256d* data = nullptr;
    typedef capd::filib::Interval<double> Interval;
    static constexpr size_t vectors_count = (N+3)/4;
    static constexpr size_t full_vectors = N/4;
    static constexpr size_t rest = N % 4;

    BatchSwitchVectorAVX_G256(bool allocateOnly)
    {
        if(allocateOnly)    data = new __m256d[2*vectors_count];
        else                data = nullptr;
    }

    struct Index{
        size_t ind;
        size_t poz;
    };
    struct Accessor {
        __m256d* data;
        Interval get(Index index) const {
            __m256d vec_lower = data[index.ind * 2];
            __m256d vec_upper = data[index.ind * 2 + 1];
            switch (index.poz)
            {
            case 3:{
                __m128d lower = _mm256_extractf128_pd(vec_lower,1);
                double low_val = _mm_cvtsd_f64(_mm_shuffle_pd(lower,lower,1));
                __m128d upper = _mm256_extractf128_pd(vec_upper,1);
                double upp_val = _mm_cvtsd_f64(_mm_shuffle_pd(upper,upper,1));
                return Interval(low_val,upp_val);
            }
            case 2:{
                __m128d lower = _mm256_extractf128_pd(vec_lower,1);
                double low_val = _mm_cvtsd_f64(lower);
                __m128d upper = _mm256_extractf128_pd(vec_upper,1);
                double upp_val = _mm_cvtsd_f64(upper);
                return Interval(low_val,upp_val);
            }
            case 1:{
                __m128d lower = _mm256_castpd256_pd128(vec_lower);
                double low_val = _mm_cvtsd_f64(_mm_shuffle_pd(lower,lower,1));
                __m128d upper = _mm256_castpd256_pd128(vec_upper);
                double upp_val = _mm_cvtsd_f64(_mm_shuffle_pd(upper,upper,1));
                return Interval(low_val,upp_val);
            }
            default:
                 __m128d lower = _mm256_castpd256_pd128(vec_lower);
                double low_val = _mm_cvtsd_f64(lower);
                __m128d upper = _mm256_castpd256_pd128(vec_upper);
                double upp_val = _mm_cvtsd_f64(upper);
                return Interval(low_val,upp_val);
            }
        }
        void set(Index index, const Interval& interval) {
            __m256d low = _mm256_set1_pd(interval.leftBound());
            __m256d upp = _mm256_set1_pd(interval.rightBound());
            __m256d vec_lower = data[index.ind * 2];
            __m256d vec_upper = data[index.ind * 2 + 1];
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
            data[index.ind*2] = vec_lower;
            data[index.ind*2+1] = vec_upper;
        }
    };

public:
    BatchSwitchVectorAVX_G256(){
        data = new __m256d[2*vectors_count]();
    }
    BatchSwitchVectorAVX_G256(const BatchSwitchVectorAVX_G256& cpy){
        data =  static_cast<__m256d*>(operator new[](vectors_count*2* sizeof(__m256d)));
        for(size_t i =0; i < 2*vectors_count;i++){
            data[i] = cpy.data[i];
        }
    }
    BatchSwitchVectorAVX_G256(BatchSwitchVectorAVX_G256&& cpy) noexcept{
        data = cpy.data;
        cpy.data = nullptr;
    }
    BatchSwitchVectorAVX_G256& operator=(const BatchSwitchVectorAVX_G256& fst) {
         if (this != &fst) {
            delete[] data;
            if (fst.data) {
                data = static_cast<__m256d*>(operator new[](vectors_count * 2 * sizeof(__m256d)));
                for (size_t i = 0; i < 2 * vectors_count; ++i) {
                    data[i] = fst.data[i];
                }
            } else {
                data = nullptr;
            }
        }
        return *this;
    }

    BatchSwitchVectorAVX_G256& operator=(BatchSwitchVectorAVX_G256&& fst) noexcept {
        if (this != &fst) {
            delete[] data;
            data = fst.data;
            fst.data = nullptr;
        }
        return *this;
    }
    BatchSwitchVectorAVX_G256(const double(&array)[2*N]){
        data = new __m256d[vectors_count*2];
        for(size_t i = 0; i<full_vectors*8;i+=8){
            __m256d a = _mm256_set_pd(array[i+6],array[i+4],array[i+2],array[i]);
            __m256d b = _mm256_set_pd(array[i+7],array[i+5],array[i+3],array[i+1]);
            data[i/4] = a;
            data[i/4+1] = b;
        }
        if (rest > 0) {
            double temp_lower[4] = {};
            double temp_upper[4] = {};
            for (size_t i = 0; i < rest; ++i) {
                temp_lower[i] = array[2 * N - 2 * rest + 2 * i];
                temp_upper[i] = array[2 * N - 2 * rest + 2 * i + 1];
            }
            data[vectors_count*2-2] = _mm256_load_pd(temp_lower);
            data[vectors_count*2-1] = _mm256_load_pd(temp_upper);
        }
    }
    BatchSwitchVectorAVX_G256(const std::vector<double>& array) {
        if (array.size() != 2 * N) {
            throw std::invalid_argument("Rozmiar wektora musi wynosić dokładnie 2 * N.");
        }
        data = new __m256d[vectors_count*2];
        for(size_t i = 0; i<full_vectors*8;i+=8){
            __m256d a = _mm256_set_pd(array[i+6],array[i+4],array[i+2],array[i]);
            __m256d b = _mm256_set_pd(array[i+7],array[i+5],array[i+3],array[i+1]);
            data[i/4] = a;
            data[i/4+1] = b;
        }
        if (rest > 0) {
            double temp_lower[4] = {}; 
            double temp_upper[4] = {};
            for (size_t i = 0; i < rest; ++i) {
                temp_lower[i] = array[2 * N - 2 * rest + 2 * i];
                temp_upper[i] = array[2 * N - 2 * rest + 2 * i + 1];
            }
            data[vectors_count*2-2] = _mm256_load_pd(temp_lower);
            data[vectors_count*2-1] = _mm256_load_pd(temp_upper);
        }
    }
    friend BatchSwitchVectorAVX_G256 operator+(const BatchSwitchVectorAVX_G256& fst, const BatchSwitchVectorAVX_G256& scd){
        BatchSwitchVectorAVX_G256 result(true);
        capd::rounding::DoubleRounding::roundDown();
        for(size_t i = 0; i<result.vectors_count*2;i+=2){
            result.data[i] = _mm256_add_pd(fst.data[i],scd.data[i]);
        }
        capd::rounding::DoubleRounding::roundUp();
        for(size_t i = 1; i<result.vectors_count*2;i+=2){
            result.data[i] = _mm256_add_pd(fst.data[i],scd.data[i]);
        }
        return result;
    }
    friend BatchSwitchVectorAVX_G256 operator-(const BatchSwitchVectorAVX_G256& fst, const BatchSwitchVectorAVX_G256& scd){
        BatchSwitchVectorAVX_G256 result(true);
        capd::rounding::DoubleRounding::roundDown();
        for(size_t i = 0; i<result.vectors_count*2;i+=2){
            result.data[i] = _mm256_sub_pd(fst.data[i],scd.data[i+1]);
        }
        capd::rounding::DoubleRounding::roundUp();
        for(size_t i = 1; i<result.vectors_count*2;i+=2){
            result.data[i] = _mm256_sub_pd(fst.data[i],scd.data[i-1]);
        }
        return result;
    }

    BatchSwitchVectorAVX_G256 & operator+=(const BatchSwitchVectorAVX_G256& fst){
        *this = *this+fst;
        return *this;
    }
    BatchSwitchVectorAVX_G256 & operator-=(const BatchSwitchVectorAVX_G256& fst){
        *this = *this-fst;
        return *this;
    }

    friend BatchSwitchVectorAVX_G256 operator+(const BatchSwitchVectorAVX_G256 &fst, const Interval & scd){
        BatchSwitchVectorAVX_G256 result(true);
        __m256d scalar = _mm256_set1_pd(scd.leftBound());
        capd::rounding::DoubleRounding::roundDown();
        for(size_t i = 0; i<result.vectors_count*2;i+=2){
            result.data[i] = _mm256_add_pd(fst.data[i],scalar);
        }
        scalar = _mm256_set1_pd(scd.rightBound());
        capd::rounding::DoubleRounding::roundUp();
        for(size_t i = 1; i<result.vectors_count*2;i+=2){
            result.data[i] = _mm256_add_pd(fst.data[i],scalar);
        }
        return result;
    }
    friend BatchSwitchVectorAVX_G256 operator-(const BatchSwitchVectorAVX_G256 &fst, const Interval & scd){
        BatchSwitchVectorAVX_G256 result(true);
        __m256d scalar = _mm256_set1_pd(scd.rightBound());
        capd::rounding::DoubleRounding::roundDown();
        for(size_t i = 0; i<result.vectors_count*2;i+=2){
            result.data[i] = _mm256_sub_pd(fst.data[i],scalar);
        }
        scalar = _mm256_set1_pd(scd.leftBound());
        capd::rounding::DoubleRounding::roundUp();
        for(size_t i = 1; i<result.vectors_count*2;i+=2){
            result.data[i] = _mm256_sub_pd(fst.data[i],scalar);
        }
        return result;
    }
    friend BatchSwitchVectorAVX_G256 operator*(const BatchSwitchVectorAVX_G256 &fst, const Interval & scd){
        char type_scalar = type(scd.leftBound(),scd.rightBound());
        switch(type_scalar)
        {
            case 0:
                return BatchSwitchVectorAVX_G256 ();
            case 1:{
                __m256d scalar = _mm256_set1_pd(scd.leftBound());
                BatchSwitchVectorAVX_G256 result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<result.vectors_count*2;i+=2){
                    result.data[i] = _mm256_mul_pd(fst.data[i],scalar);
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 1; i<result.vectors_count*2;i+=2){
                    result.data[i] = _mm256_mul_pd(fst.data[i],scalar);
                }
                return result;
            }
            case 2:{
                __m256d scalar = _mm256_set1_pd(scd.leftBound());
                BatchSwitchVectorAVX_G256 result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<result.vectors_count*2;i+=2){
                    result.data[i] = _mm256_mul_pd(fst.data[i+1],scalar);
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 1; i<result.vectors_count*2;i+=2){
                result.data[i] = _mm256_mul_pd(fst.data[i-1],scalar);
                }
                return result;
            }
            case 3:{
                __m256d scalar = _mm256_set1_pd(scd.rightBound());
                __m256d zero = _mm256_setzero_pd();
                BatchSwitchVectorAVX_G256 result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<result.vectors_count*2;i+=2){
                    result.data[i] = _mm256_min_pd(_mm256_mul_pd(fst.data[i],scalar),zero);
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 1; i<result.vectors_count*2;i+=2){
                    result.data[i] = _mm256_max_pd(_mm256_mul_pd(fst.data[i],scalar),zero);
                }
                return result;
            }
            case 4:{
                __m256d scalar1 = _mm256_set1_pd(scd.leftBound());
                __m256d scalar2 = _mm256_set1_pd(scd.rightBound());

                BatchSwitchVectorAVX_G256 result(true);

                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<result.vectors_count*2;i+=2){
                    result.data[i] = _mm256_min_pd(_mm256_mul_pd(fst.data[i],scalar1),_mm256_mul_pd(fst.data[i],scalar2));
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 1; i<result.vectors_count*2;i+=2){
                    result.data[i] = _mm256_max_pd(_mm256_mul_pd(fst.data[i],scalar1),_mm256_mul_pd(fst.data[i],scalar2));
                }
                return result;
            }
            case 5:{
                __m256d scalar = _mm256_set1_pd(scd.leftBound());
                __m256d zero = _mm256_setzero_pd();

                BatchSwitchVectorAVX_G256 result(true);

                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<result.vectors_count*2;i+=2){
                    result.data[i] = _mm256_min_pd(_mm256_mul_pd(fst.data[i+1],scalar),zero);
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 1; i<result.vectors_count*2;i+=2){
                     result.data[i] = _mm256_max_pd(_mm256_mul_pd(fst.data[i-1],scalar),zero);
                }
                return result;
            }
            case 6:{
                __m256d scalar1 = _mm256_set1_pd(scd.leftBound());
                __m256d scalar2 = _mm256_set1_pd(scd.rightBound());

                BatchSwitchVectorAVX_G256 result(true);

                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<result.vectors_count*2;i+=2){
                    result.data[i] = _mm256_min_pd(_mm256_mul_pd(fst.data[i+1],scalar1),_mm256_mul_pd(fst.data[i+1],scalar2));
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 1; i<result.vectors_count*2;i+=2){
                    result.data[i] = _mm256_max_pd(_mm256_mul_pd(fst.data[i-1],scalar1),_mm256_mul_pd(fst.data[i-1],scalar2));
                }
                return result;
            }
            case 7:{
                __m256d scalar1 = _mm256_set1_pd(scd.leftBound());
                __m256d scalar2 = _mm256_set1_pd(scd.rightBound());

                BatchSwitchVectorAVX_G256 result(true);

                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<result.vectors_count*2;i+=2){
                    result.data[i] = _mm256_min_pd(_mm256_mul_pd(fst.data[i+1],scalar1),_mm256_mul_pd(fst.data[i],scalar2));
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 1; i<result.vectors_count*2;i+=2){
                    result.data[i] = _mm256_max_pd(_mm256_mul_pd(fst.data[i-1],scalar1),_mm256_mul_pd(fst.data[i],scalar2));
                }
                return result;
            }
        }
        return BatchSwitchVectorAVX_G256 ();
    }
    friend BatchSwitchVectorAVX_G256 operator/(const BatchSwitchVectorAVX_G256 &fst, const Interval & scd){
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
            BatchSwitchVectorAVX_G256 result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<result.vectors_count*2;i+=2){
                result.data[i] = _mm256_div_pd(fst.data[i],scalar);
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 1; i<result.vectors_count*2;i+=2){
                result.data[i] = _mm256_div_pd(fst.data[i],scalar);
            }
            return result;
        }
        case 2:{
            __m256d scalar = _mm256_set1_pd(scd.leftBound());
            BatchSwitchVectorAVX_G256 result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<result.vectors_count*2;i+=2){
                result.data[i] = _mm256_div_pd(fst.data[i+1],scalar);
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 1; i<result.vectors_count*2;i+=2){
                result.data[i] = _mm256_div_pd(fst.data[i-1],scalar);
            }
            return result;
        }
        case 4:{
            __m256d scalar1 = _mm256_set1_pd(scd.leftBound());
            __m256d scalar2 = _mm256_set1_pd(scd.rightBound());

            BatchSwitchVectorAVX_G256 result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<result.vectors_count*2;i+=2){
                result.data[i] = _mm256_min_pd(_mm256_div_pd(fst.data[i],scalar1),_mm256_div_pd(fst.data[i],scalar2));
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 1; i<result.vectors_count*2;i+=2){
                result.data[i] = _mm256_max_pd(_mm256_div_pd(fst.data[i],scalar1),_mm256_div_pd(fst.data[i],scalar2));
            }
            return result;
        }
        case 6:{
            __m256d scalar1 = _mm256_set1_pd(scd.leftBound());
            __m256d scalar2 = _mm256_set1_pd(scd.rightBound());

            BatchSwitchVectorAVX_G256 result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<result.vectors_count*2;i+=2){
                result.data[i] = _mm256_min_pd(_mm256_div_pd(fst.data[i+1],scalar1),_mm256_div_pd(fst.data[i+1],scalar2));
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 1; i<result.vectors_count*2;i+=2){
                result.data[i] = _mm256_min_pd(_mm256_div_pd(fst.data[i-1],scalar1),_mm256_div_pd(fst.data[i-1],scalar2));
            }
            return result;
        }
        }
        return BatchSwitchVectorAVX_G256 ();
    }
   bool operator==(const VectorBasic<Interval,N> & fst){
        for(size_t i = 0; i <N; i++){
                Interval interval = (*this)[i];
                if(interval != fst[i]) {
                    return false;
                }
            }
        return true;
    }

    IntervalProxy<Accessor,Index> operator[](size_t i){
        Index ind(i/4,i%4);
        Accessor acc = {data};
        return IntervalProxy<Accessor,Index>(acc,ind);
    }
    IntervalProxy<Accessor,Index> operator[](size_t i) const{
        Index ind(i/4,i%4);
        Accessor acc = {data};
        return IntervalProxy<Accessor,Index>(acc,ind);
    }
    BatchSwitchVectorAVX_G256 & operator+=(const Interval& fst){
        *this = *this+fst;
        return *this;
    }
    BatchSwitchVectorAVX_G256 & operator-=(const Interval& fst){
        *this = *this-fst;
        return *this;
    }
    BatchSwitchVectorAVX_G256 & operator*=(const Interval& fst){
        *this = *this * fst;
        return *this;
    }
    BatchSwitchVectorAVX_G256 & operator/=(const Interval& fst){
        *this = *this / fst;
        return *this;
    }
    ~BatchSwitchVectorAVX_G256(){
        delete[] data;
    }

    friend std::ostream& operator<<(std::ostream& os, const BatchSwitchVectorAVX_G256& vec) {
        os << "( ";
        for (size_t i = 0; i < N; ++i) {
            os << vec[i] << " ";
        }
        os << ")";
        return os;
    }
};

#endif