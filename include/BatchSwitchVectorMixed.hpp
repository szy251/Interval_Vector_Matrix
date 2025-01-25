#ifndef VECTOR_BATCH_HPP
#define VECTOR_BATCH_HPP

#include <iostream>
#include <capd/intervals/Interval.h>
#include <Utilities.hpp>
#include <IntervalProxy.hpp>


template<size_t N>
class BatchSwitchVectorMixed
{
private:
    typedef capd::intervals::Interval<double> Interval;
    double *data =  nullptr;

    struct Accessor {
        double* data;
        Interval get(size_t index) const {
            return Interval(data[2 * index], data[2 * index + 1]);
        }
        void set(size_t index, const Interval& interval) {
            data[2 * index] = interval.leftBound();
            data[2 * index + 1] = interval.rightBound();
        }
    };

    BatchSwitchVectorMixed(bool allocateOnly)
    {
        if(allocateOnly)    data = new double[2*N];
        else                data = nullptr;
    }

public:
    BatchSwitchVectorMixed(){
        data = new double[2*N]();
    }
    BatchSwitchVectorMixed(const BatchSwitchVectorMixed& cpy){
        data =  new double[2*N];
        for(size_t i =0; i < 2*N;i++){
            data[i] = cpy.data[i];
        }
    }
    BatchSwitchVectorMixed(BatchSwitchVectorMixed&& cpy) noexcept{
        data = cpy.data;
        cpy.data = nullptr;
    }
    BatchSwitchVectorMixed(const double(&array)[2*N]) {
        data = new double[2*N];
        std::copy(std::begin(array), std::end(array), data);
    }
    BatchSwitchVectorMixed(const std::vector<double>& vec) {
        if (vec.size() != 2 * N) {
            throw std::invalid_argument("Rozmiar wektora musi wynosić dokładnie 2 * N.");
        }

        data = new double[2 * N]; 
        std::copy(vec.begin(), vec.end(), data);
    }

    BatchSwitchVectorMixed& operator=(const BatchSwitchVectorMixed& fst) {
        if (this != &fst) {
            delete[] data;
            data = new double[N*2];
            for (size_t i = 0; i < N*2; ++i) {
                data[i] = fst.data[i];
            }
        }
        return *this;
    }

    BatchSwitchVectorMixed& operator=(BatchSwitchVectorMixed&& fst) noexcept {
        if (this != &fst) {
            delete[] data;
            data = fst.data;
            fst.data = nullptr;
        }
        return *this;
    }
    
    friend BatchSwitchVectorMixed operator+(const BatchSwitchVectorMixed& fst, const BatchSwitchVectorMixed& scd){
        BatchSwitchVectorMixed result(true);
        capd::rounding::DoubleRounding::roundDown();
        for(size_t i = 0; i<2*N;i+=2){
            result.data[i] = fst.data[i]+scd.data[i];
        }
        capd::rounding::DoubleRounding::roundUp();
        for(size_t i = 1; i<2*N;i+=2){
            result.data[i] = fst.data[i]+scd.data[i];
        }
        return result;
    }
    friend BatchSwitchVectorMixed operator-(const BatchSwitchVectorMixed& fst, const BatchSwitchVectorMixed& scd){
        BatchSwitchVectorMixed result(true);
        capd::rounding::DoubleRounding::roundDown();
        for(size_t i = 0; i<2*N;i+=2){
            result.data[i] = fst.data[i]-scd.data[i+1];
        }
        capd::rounding::DoubleRounding::roundUp();
        for(size_t i = 1; i<2*N;i+=2){
            result.data[i] = fst.data[i]-scd.data[i-1];
        }
        return result;
    }
    BatchSwitchVectorMixed & operator+=(const BatchSwitchVectorMixed& fst){
        *this = *this+fst;
        return *this;
    }
    BatchSwitchVectorMixed & operator-=(const BatchSwitchVectorMixed& fst){
        *this = *this-fst;
        return *this;
    }
    
    friend BatchSwitchVectorMixed operator+(const BatchSwitchVectorMixed &fst, const Interval & scd){
        BatchSwitchVectorMixed result(true);
        capd::rounding::DoubleRounding::roundDown();
        for(size_t i = 0; i<2*N;i+=2){
            result.data[i] = fst.data[i]+scd.leftBound();
        }
        capd::rounding::DoubleRounding::roundUp();
        for(size_t i = 1; i<2*N;i+=2){
            result.data[i] = fst.data[i]+scd.rightBound();
        }
        return result;
    }
    friend BatchSwitchVectorMixed operator-(const BatchSwitchVectorMixed &fst, const Interval & scd){
        BatchSwitchVectorMixed result(true);
        capd::rounding::DoubleRounding::roundDown();
        for(size_t i = 0; i<2*N;i+=2){
            result.data[i] = fst.data[i]-scd.rightBound();
        }
        capd::rounding::DoubleRounding::roundUp();
        for(size_t i = 1; i<2*N;i+=2){
            result.data[i] = fst.data[i]-scd.leftBound();
        }
        return result;
    }
    friend BatchSwitchVectorMixed operator*(const BatchSwitchVectorMixed &fst, const Interval & scd){
        char type_scalar = type(scd.leftBound(),scd.rightBound());
        switch(type_scalar)
        {
            case 0:
                return BatchSwitchVectorMixed ();
            case 1:{
                BatchSwitchVectorMixed result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<2*N;i+=2){
                    result.data[i] = fst.data[i]*scd.leftBound();
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 1; i<2*N;i+=2){
                    result.data[i] = fst.data[i]*scd.leftBound();
                }
                return result;
            }
            case 2:{
                BatchSwitchVectorMixed result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<2*N;i+=2){
                    result.data[i] = fst.data[i+1]*scd.leftBound();
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 1; i<2*N;i+=2){
                result.data[i] = fst.data[i-1]*scd.leftBound();
                }
                return result;
            }
            case 3:{
                BatchSwitchVectorMixed result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<2*N;i+=2){
                    if(fst.data[i] < 0) result.data[i] = fst.data[i]*scd.rightBound();
                    else result.data[i] = 0.;
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 1; i<2*N;i+=2){
                if(fst.data[i] > 0) result.data[i] = fst.data[i]*scd.rightBound();
                else result.data[i] = 0.;
                }
                return result;
            }
            case 4:{
                BatchSwitchVectorMixed result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<2*N;i+=2){
                    if(fst.data[i] > 0) result.data[i] = fst.data[i]*scd.leftBound();
                    else if(fst.data[i] == 0)result.data[i] = 0.;
                    else result.data[i] = fst.data[i]*scd.rightBound();
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 1; i<2*N;i+=2){
                if(fst.data[i] > 0) result.data[i] = fst.data[i]*scd.rightBound();
                else if(fst.data[i] == 0)result.data[i] = 0.;
                else result.data[i] = fst.data[i]*scd.leftBound();
                }
                return result;
            }
            case 5:{
                BatchSwitchVectorMixed result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<2*N;i+=2){
                    if(fst.data[i+1] > 0) result.data[i] = fst.data[i+1]*scd.leftBound();
                    else result.data[i] = 0.;
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 1; i<2*N;i+=2){
                if(fst.data[i-1] < 0) result.data[i] = fst.data[i-1]*scd.leftBound();
                else result.data[i] = 0.;
                }
                return result;
            }
            case 6:{
                BatchSwitchVectorMixed result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<2*N;i+=2){
                    if(fst.data[i+1] > 0) result.data[i] = fst.data[i+1]*scd.leftBound();
                    else if(fst.data[i+1] ==0.) result.data[i] = 0.;
                    else result.data[i] = fst.data[i+1]*scd.rightBound();
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 1; i<2*N;i+=2){
                    if(fst.data[i-1] > 0) result.data[i] = fst.data[i-1]*scd.rightBound();
                    else if(fst.data[i-1] ==0.) result.data[i] = 0.;
                    else result.data[i] = fst.data[i-1]*scd.leftBound();
                }
                return result;
            }
            case 7:{
                BatchSwitchVectorMixed result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<2*N;i+=2){
                    if(fst.data[i+1] > 0.)
                    {
                        double k = fst.data[i+1]*scd.leftBound();
                        if(fst.data[i] < 0. ){
                            result.data[i] = std::min(k,fst.data[i]*scd.rightBound());
                        }
                        else result.data[i] = k;
                    }
                    else if(fst.data[i] <0.) result.data[i] = fst.data[i]*scd.rightBound();
                    else result.data[i] = 0;
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 1; i<2*N;i+=2){
                    if(fst.data[i-1] < 0.)
                    {
                        double k = fst.data[i-1]*scd.leftBound();
                        if(fst.data[i] > 0. ){
                            result.data[i] = std::min(k,fst.data[i]*scd.rightBound());
                        }
                        else result.data[i] = k;
                    }
                    else if(fst.data[i] > 0.) result.data[i] = fst.data[i]*scd.rightBound();
                    else result.data[i] = 0;
                }
                return result;
            }
        }
        return BatchSwitchVectorMixed ();
    }
    friend BatchSwitchVectorMixed operator/(const BatchSwitchVectorMixed &fst, const Interval & scd){
        char type_scalar = type(scd.leftBound(),scd.rightBound());
        switch (type_scalar)
        {
        case 0:
        case 3:
        case 5:
        case 7:
            throw std::invalid_argument("Scalat has 0 in it");
        case 1:{
            BatchSwitchVectorMixed result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<2*N;i+=2){
                result.data[i] = fst.data[i]/scd.leftBound();
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 1; i<2*N;i+=2){
                result.data[i] = fst.data[i]/scd.leftBound();
            }
            return result;
        }
        case 2:{
            BatchSwitchVectorMixed result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<2*N;i+=2){
                result.data[i] = fst.data[i+1]/scd.leftBound();
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 1; i<2*N;i+=2){
                result.data[i] = fst.data[i-1]/scd.leftBound();
            }
            return result;
        }
        case 4:{
            BatchSwitchVectorMixed result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<2*N;i+=2){
                if(fst.data[i] > 0) result.data[i] = fst.data[i]/scd.rightBound();
                else if(fst.data[i] == 0)result.data[i] = 0.;
                else result.data[i] = fst.data[i]/scd.leftBound();
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 1; i<2*N;i+=2){
                if(fst.data[i] > 0) result.data[i] = fst.data[i]/scd.leftBound();
                else if(fst.data[i] == 0)result.data[i] = 0.;
                else result.data[i] = fst.data[i]/scd.rightBound();
            }
            return result;
        }
        case 6:{
            BatchSwitchVectorMixed result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<2*N;i+=2){
                if(fst.data[i+1] > 0) result.data[i] = fst.data[i+1]/scd.rightBound();
                else if(fst.data[i+1] ==0.) result.data[i] = 0.;
                else result.data[i] = fst.data[i+1]/scd.leftBound();
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 1; i<2*N;i+=2){
                if(fst.data[i-1] > 0) result.data[i] = fst.data[i-1]/scd.leftBound();
                else if(fst.data[i-1] ==0.) result.data[i] = 0.;
                else result.data[i] = fst.data[i-1]/scd.rightBound();
            }
            return result;
        }
        }
        return BatchSwitchVectorMixed ();
    }

    BatchSwitchVectorMixed & operator+=(const Interval& fst){
        *this = *this+fst;
        return *this;
    }
    BatchSwitchVectorMixed & operator-=(const Interval& fst){
        *this = *this-fst;
        return *this;
    }
    BatchSwitchVectorMixed & operator*=(const Interval& fst){
        *this = *this * fst;
        return *this;
    }
    BatchSwitchVectorMixed & operator/=(const Interval& fst){
        *this = *this / fst;
        return *this;
    }
    IntervalProxy<Accessor,size_t> operator[](size_t index) {
        Accessor accessor = {data};
        return IntervalProxy<Accessor,size_t>(accessor, index);
    }

    ~BatchSwitchVectorMixed(){
        delete[] data;
    }
};


#endif