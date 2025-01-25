#ifndef VECTOR_BATCH_GRP_HPP
#define VECTOR_BATCH_GRP_HPP

#include <iostream>
#include <capd/intervals/Interval.h>
#include <Utilities.hpp>
#include <IntervalProxy.hpp>

template<size_t N>
class BatchSwitchVector_Grouped
{
private:
    typedef capd::intervals::Interval<double> Interval;
    double* low = nullptr; // Dolne wartości
    double* up = nullptr;  // Górne wartości
    
    struct Accessor {
        double* low;
        double* up;
        Interval get(size_t index) const {
            return Interval(low[index], up[index]);
        }
        void set(size_t index, const Interval& interval) {
            low[index] = interval.leftBound();
            up[index] = interval.rightBound();
        }
    };

    BatchSwitchVector_Grouped(bool allocateOnly) {
        if (allocateOnly) {
            low = new double[N];
            up = new double[N];
        }
    }

public:
    BatchSwitchVector_Grouped() {
        low = new double[N]();
        up = new double[N]();
    }
    BatchSwitchVector_Grouped(const BatchSwitchVector_Grouped& cpy) {
        low = new double[N];
        up = new double[N];
        for (size_t i = 0; i < N; ++i) {
            low[i] = cpy.low[i];
            up[i] = cpy.up[i];
        }
    }
    BatchSwitchVector_Grouped(BatchSwitchVector_Grouped&& cpy) noexcept {
        low = cpy.low;
        up = cpy.up;
        cpy.low = nullptr;
        cpy.up = nullptr;
    }
    BatchSwitchVector_Grouped(const double(&array)[2 * N]) {
        low = new double[N];
        up = new double[N];
        for (size_t i = 0; i < N; ++i) {
            low[i] = array[2 * i];
            up[i] = array[2 * i + 1];
        }
    }
    BatchSwitchVector_Grouped(const std::vector<double>& array) {
        if (array.size() != 2 * N) {
            throw std::invalid_argument("Rozmiar wektora musi wynosić dokładnie 2 * N.");
        }
        low = new double[N];
        up = new double[N];
        for (size_t i = 0; i < N; ++i) {
            low[i] = array[2 * i];
            up[i] = array[2 * i + 1];
        }
    }

    BatchSwitchVector_Grouped& operator=(const BatchSwitchVector_Grouped& fst) {
        if (this != &fst) {
            delete[] low;
            delete[] up;
            low = new double[N];
            up = new double[N];
            for (size_t i = 0; i < N; ++i) {
                low[i] = fst.low[i];
                up[i] = fst.up[i];
            }
        }
        return *this;
    }

    BatchSwitchVector_Grouped& operator=(BatchSwitchVector_Grouped&& fst) noexcept {
        if (this != &fst) {
            delete[] low;
            delete[] up;
            low = fst.low;
            up = fst.up;
            fst.low = nullptr;
            fst.up = nullptr;
        }
        return *this;
    }

    friend BatchSwitchVector_Grouped operator+(const BatchSwitchVector_Grouped& fst, const BatchSwitchVector_Grouped& scd) {
        BatchSwitchVector_Grouped result(true);
        capd::rounding::DoubleRounding::roundDown();
        for (size_t i = 0; i < N; ++i) {
            result.low[i] = fst.low[i] + scd.low[i];
        }
        capd::rounding::DoubleRounding::roundUp();
        for (size_t i = 0; i < N; ++i) {
            result.up[i] = fst.up[i] + scd.up[i];
        }
        return result;
    }
    friend BatchSwitchVector_Grouped operator-(const BatchSwitchVector_Grouped& fst, const BatchSwitchVector_Grouped& scd) {
        BatchSwitchVector_Grouped result(true);
        capd::rounding::DoubleRounding::roundDown();
        for (size_t i = 0; i < N; ++i) {
            result.low[i] = fst.low[i] - scd.up[i];
        }
        capd::rounding::DoubleRounding::roundUp();
        for (size_t i = 0; i < N; ++i) {
            result.up[i] = fst.up[i] - scd.low[i];
        }
        return result;
    }

    BatchSwitchVector_Grouped & operator+=(const BatchSwitchVector_Grouped& fst){
        *this = *this+fst;
        return *this;
    }
    BatchSwitchVector_Grouped & operator-=(const BatchSwitchVector_Grouped& fst){
        *this = *this-fst;
        return *this;
    }

    friend BatchSwitchVector_Grouped operator+(const BatchSwitchVector_Grouped &fst, const Interval & scd){
        BatchSwitchVector_Grouped result(true);
        capd::rounding::DoubleRounding::roundDown();
         for (size_t i = 0; i < N; ++i) {
            result.low[i] = fst.low[i]+scd.leftBound();
        }
        capd::rounding::DoubleRounding::roundUp();
         for (size_t i = 0; i < N; ++i) {
            result.up[i] = fst.up[i]+scd.rightBound();
        }
        return result;
    }
    friend BatchSwitchVector_Grouped operator-(const BatchSwitchVector_Grouped &fst, const Interval & scd){
        BatchSwitchVector_Grouped result(true);
        capd::rounding::DoubleRounding::roundDown();
         for (size_t i = 0; i < N; ++i) {
            result.low[i] = fst.low[i]-scd.rightBound();
        }
        capd::rounding::DoubleRounding::roundUp();
        for(size_t i = 1; i<N;i+=2){
            result.up[i] = fst.up[i]-scd.leftBound();
        }
        return result;
    }
    friend BatchSwitchVector_Grouped operator*(const BatchSwitchVector_Grouped &fst, const Interval & scd){
        char type_scalar = type(scd.leftBound(),scd.rightBound());
        switch(type_scalar)
        {
            case 0:
                return BatchSwitchVector_Grouped ();
            case 1:{
                BatchSwitchVector_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<N;i++){
                    result.low[i] = fst.low[i]*scd.leftBound();
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<N;i++){
                    result.up[i] = fst.up[i]*scd.leftBound();
                }
                return result;
            }
            case 2:{
                BatchSwitchVector_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<N;i++){
                    result.low[i] = fst.up[i]*scd.leftBound();
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<N;i++){
                result.up[i] = fst.low[i]*scd.leftBound();
                }
                return result;
            }
            case 3:{
                BatchSwitchVector_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<N;i++){
                    if(fst.low[i] < 0) result.low[i] = fst.low[i]*scd.rightBound();
                    else result.low[i] = 0.;
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<N;i++){
                if(fst.up[i] > 0) result.up[i] = fst.up[i]*scd.rightBound();
                else result.up[i] = 0.;
                }
                return result;
            }
            case 4:{
                BatchSwitchVector_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<N;i++){
                    if(fst.low[i] > 0) result.low[i] = fst.low[i]*scd.leftBound();
                    else if(fst.low[i] == 0)result.low[i] = 0.;
                    else result.low[i] = fst.low[i]*scd.rightBound();
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<N;i++){
                if(fst.up[i] > 0) result.up[i] = fst.up[i]*scd.rightBound();
                else if(fst.up[i] == 0)result.up[i] = 0.;
                else result.up[i] = fst.up[i]*scd.leftBound();
                }
                return result;
            }
            case 5:{
                BatchSwitchVector_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<N;i++){
                    if(fst.up[i] > 0) result.low[i] = fst.up[i]*scd.leftBound();
                    else result.low[i] = 0.;
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<N;i++){
                    if(fst.low[i] < 0) result.up[i] = fst.low[i]*scd.leftBound();
                    else result.up[i] = 0.;
                }
                return result;
            }
            case 6:{
                BatchSwitchVector_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<N;i++){
                    if(fst.up[i] > 0) result.low[i] = fst.up[i]*scd.leftBound();
                    else if(fst.up[i] ==0.) result.low[i] = 0.;
                    else result.low[i] = fst.up[i]*scd.rightBound();
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<N;i++){
                    if(fst.low[i] > 0) result.up[i] = fst.low[i]*scd.rightBound();
                    else if(fst.low[i] ==0.) result.up[i] = 0.;
                    else result.up[i] = fst.low[i]*scd.leftBound();
                }
                return result;
            }
            case 7:{
                BatchSwitchVector_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<N;i++){
                    if(fst.up[i] > 0.)
                    {
                        double k = fst.up[i]*scd.leftBound();
                        if(fst.low[i] < 0. ){
                            result.low[i] = std::min(k,fst.low[i]*scd.rightBound());
                        }
                        else result.low[i] = k;
                    }
                    else if(fst.low[i] <0.) result.low[i] = fst.low[i]*scd.rightBound();
                    else result.low[i] = 0;
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<N;i++){
                    if(fst.low[i] < 0.)
                    {
                        double k = fst.low[i]*scd.leftBound();
                        if(fst.up[i] > 0. ){
                            result.up[i] = std::min(k,fst.up[i]*scd.rightBound());
                        }
                        else result.up[i] = k;
                    }
                    else if(fst.up[i] > 0.) result.up[i] = fst.up[i]*scd.rightBound();
                    else result.up[i] = 0;
                }
                return result;
            }
        }
        return BatchSwitchVector_Grouped ();
    }
    friend BatchSwitchVector_Grouped operator/(const BatchSwitchVector_Grouped &fst, const Interval & scd){
        char type_scalar = type(scd.leftBound(),scd.rightBound());
        switch (type_scalar)
        {
        case 0:
        case 3:
        case 5:
        case 7:
            throw std::invalid_argument("Scalat has 0 in it");
        case 1:{
            BatchSwitchVector_Grouped result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<N;i++){
                result.low[i] = fst.low[i]/scd.leftBound();
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 0; i<N;i++){
                result.up[i] = fst.up[i]/scd.leftBound();
            }
            return result;
        }
        case 2:{
            BatchSwitchVector_Grouped result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<N;i++){
                result.low[i] = fst.up[i]/scd.leftBound();
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 0; i<N;i++){
                result.up[i] = fst.low[i]/scd.leftBound();
            }
            return result;
        }
        case 4:{
            BatchSwitchVector_Grouped result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<N;i++){
                if(fst.low[i] > 0) result.low[i] = fst.low[i]/scd.rightBound();
                else if(fst.low[i] == 0)result.low[i] = 0.;
                else result.low[i] = fst.low[i]/scd.leftBound();
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 0; i<N;i++){
                if(fst.up[i] > 0) result.up[i] = fst.up[i]/scd.leftBound();
                else if(fst.up[i] == 0)result.up[i] = 0.;
                else result.up[i] = fst.up[i]/scd.rightBound();
            }
            return result;
        }
        case 6:{
            BatchSwitchVector_Grouped result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<N;i++){
                if(fst.up[i] > 0) result.low[i] = fst.up[i]/scd.rightBound();
                else if(fst.up[i] ==0.) result.low[i] = 0.;
                else result.low[i] = fst.up[i]/scd.leftBound();
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 0; i<N;i++){
                if(fst.low[i] > 0) result.up[i] = fst.low[i]/scd.leftBound();
                else if(fst.low[i] ==0.) result.up[i] = 0.;
                else result.up[i] = fst.low[i]/scd.rightBound();
            }
            return result;
        }
        }
        return BatchSwitchVector_Grouped ();
    }

    BatchSwitchVector_Grouped & operator+=(const Interval& fst){
        *this = *this+fst;
        return *this;
    }
    BatchSwitchVector_Grouped & operator-=(const Interval& fst){
        *this = *this-fst;
        return *this;
    }
    BatchSwitchVector_Grouped & operator*=(const Interval& fst){
        *this = *this * fst;
        return *this;
    }
    BatchSwitchVector_Grouped & operator/=(const Interval& fst){
        *this = *this / fst;
        return *this;
    }
    IntervalProxy<Accessor,size_t> operator[](size_t index){
        Accessor acc{low, up};
        return IntervalProxy<Accessor,size_t>(acc,index);
    }

    ~BatchSwitchVector_Grouped() {
        delete[] low;
        delete[] up;
    }
};


#endif
