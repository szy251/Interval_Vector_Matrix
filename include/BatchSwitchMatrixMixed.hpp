#ifndef MATRIX_BATCH_HPP
#define MATRIX_BATCH_HPP

#include <iostream>
#include <capd/intervals/Interval.h>
#include <Utilities.hpp>
#include <IntervalProxy.hpp>

template<size_t N, size_t M>
class BatchSwitchMatrixMixed{
    private:
    typedef capd::intervals::Interval<double> Interval;
    double *data =  nullptr;
    template<size_t N1,size_t M1>
    friend class BatchSwitchMatrixMixed;

    struct Accessor {
        double* data;
        Interval get(size_t index) const {
            return Interval(data[index], data[index + 1]);
        }
        void set(size_t index, const Interval& interval) {
            data[index] = interval.leftBound();
            data[index + 1] = interval.rightBound();
        }
    };

    public:
    BatchSwitchMatrixMixed(bool allocateOnly)
    {
        if(allocateOnly)    data = new double[2*N*M];
        else                data = nullptr;
    }
    BatchSwitchMatrixMixed(){
        data = new double[2*N*M]();
    }
    BatchSwitchMatrixMixed(const BatchSwitchMatrixMixed& cpy){
        data =  new double[2*N*M];
        for(size_t i =0; i < 2*N*M;i++){
            data[i] = cpy.data[i];
        }
    }
    BatchSwitchMatrixMixed(BatchSwitchMatrixMixed&& cpy) noexcept{
        data = cpy.data;
        cpy.data = nullptr;
    }
    BatchSwitchMatrixMixed(const double(&array)[2*N*M]) {
        data = new double[2*N*M];
        std::copy(std::begin(array), std::end(array), data);
    }
    BatchSwitchMatrixMixed(const std::vector<double>& vec) {
        if (vec.size() != 2 * N*M) {
            throw std::invalid_argument("Vector size need to conatain 2*N*M doube.");
        }

        data = new double[2 * N*M]; 
        std::copy(vec.begin(), vec.end(), data);
    }

    BatchSwitchMatrixMixed& operator=(const BatchSwitchMatrixMixed& fst) {
        if (this != &fst) {
            delete[] data;
            data = new double[N*M*2];
            for (size_t i = 0; i < M*N*2; ++i) {
                data[i] = fst.data[i];
            }
        }
        return *this;
    }

    BatchSwitchMatrixMixed& operator=(BatchSwitchMatrixMixed&& fst) noexcept {
        if (this != &fst) {
            delete[] data;
            data = fst.data;
            fst.data = nullptr;
        }
        return *this;
    }

    template<size_t N1, size_t M1>
    friend BatchSwitchMatrixMixed<M1, N1> transposition(const BatchSwitchMatrixMixed<N1,M1>& fst);

    friend BatchSwitchMatrixMixed operator+(const BatchSwitchMatrixMixed& fst, const BatchSwitchMatrixMixed& scd){
        BatchSwitchMatrixMixed result(true);
        capd::rounding::DoubleRounding::roundDown();
        for(size_t i = 0; i<2*N*M;i+=2){
            result.data[i] = fst.data[i]+scd.data[i];
        }
        capd::rounding::DoubleRounding::roundUp();
        for(size_t i = 1; i<2*N*M;i+=2){
            result.data[i] = fst.data[i]+scd.data[i];
        }
        return result;
    }
    friend BatchSwitchMatrixMixed operator-(const BatchSwitchMatrixMixed& fst, const BatchSwitchMatrixMixed& scd){
        BatchSwitchMatrixMixed result(true);
        capd::rounding::DoubleRounding::roundDown();
        for(size_t i = 0; i<2*N*M;i+=2){
            result.data[i] = fst.data[i]-scd.data[i+1];
        }
        capd::rounding::DoubleRounding::roundUp();
        for(size_t i = 1; i<2*N*M;i+=2){
            result.data[i] = fst.data[i]-scd.data[i-1];
        }
        return result;
    }
    /*
    template <size_t P>
    BatchSwitchMatrixMixed<N, P> operator*(const BatchSwitchMatrixMixed<M, P>& fst) {
        BatchSwitchMatrixMixed<N,P> result(true);
        BatchSwitchMatrixMixed<P,M> pom = transposition(fst);

        capd::rounding::DoubleRounding::roundDown();
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < P; ++j) {
            double sum = 0;
                for (size_t k = 0; k < M; ++k) {
                    // Obliczenie wspólnych indeksów
                    size_t idx1 = (i * M + k) * 2;
                    size_t idx2 = (j * M + k) * 2;

                    // Pobranie wartości z macierzy
                    double a_min = data[idx1];
                    double a_max = data[idx1 + 1];
                    double b_min = fst.data[idx2];
                    double b_max = fst.data[idx2 + 1];

                    // Pomijanie zerowych przedziałów
                    if ((a_min == 0 && a_max == 0) || (b_min == 0 && b_max == 0)) {
                        continue;
                    }

                    // Różne przypadki obliczeniowe
                    if (a_min >= 0) {
                        // Przedział dodatni (++) 
                        if (b_min >= 0) {
                            sum += a_min * b_min; // (++) * (++)
                        } else {
                            sum += a_max * b_min; // (++) * (-+), (++) * (--)
                        }
                    } else if (a_max <= 0) {
                        // Przedział ujemny (--)
                        if (b_max <= 0) {
                            sum += a_max * b_max; // (--) * (--)
                        } else {
                            sum += a_min * b_max; // (--) * (-+), (--) * (++)
                        }
                    } else {
                        // Przedział mieszany (-+)
                        if (b_max <= 0) {
                            sum += a_max * b_min; // (-+) * (--)
                        } else if (b_min >= 0) {
                            sum += a_min * b_max; // (-+) * (++)
                        } else {
                            // (-+) * (-+), wybór mniejszej wartości
                            double t = a_min * b_max;
                            double z = a_max * b_min;
                            sum += (z < t) ? z : t;
                        }
                    }
                }
                result.data[(i*P+j)*2] = sum;
            }
        }


         capd::rounding::DoubleRounding::roundUp();
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < P; ++j) {
            double sum = 0;
                for (size_t k = 0; k < M; ++k) {
                    // Obliczenie wspólnych indeksów
                    size_t idx1 = (i * M + k) * 2;
                    size_t idx2 = (j * M + k) * 2;

                    // Pobranie wartości z macierzy
                    double a_min = data[idx1];
                    double a_max = data[idx1 + 1];
                    double b_min = fst.data[idx2];
                    double b_max = fst.data[idx2 + 1];

                    // Pomijanie zerowych przedziałów
                    if ((a_min == 0 && a_max == 0) || (b_min == 0 && b_max == 0)) {
                        continue;
                    }

                    // Różne przypadki obliczeniowe
                    if (a_min >= 0) {
                        // Przedział dodatni (++) 
                        if (b_max <= 0) {
                            sum += a_min * b_max; // (++) * (--)
                        } else {
                            sum += a_max * b_max; // (++) * (-+), (++) * (++)
                        }
                    } else if (a_max <= 0) {
                        // Przedział ujemny (--)
                        if (b_min >= 0) {
                            sum += a_max * b_min; // (--) * (++)
                        } else {
                            sum += a_min * b_min; // (--) * (-+), (--) * (--)
                        }
                    } else {
                        // Przedział mieszany (-+)
                        if (b_max <= 0) {
                            sum += a_min * b_min; // (-+) * (--)
                        } else if (b_min >= 0) {
                            sum += a_max * b_max; // (-+) * (++)
                        } else {
                            // (-+) * (-+), wybór mniejszej wartości
                            double t = a_min * b_min;
                            double z = a_max * b_max;
                            sum += (z < t) ? z : t;
                        }
                    }
                }
                result.data[(i*P+j)*2+1] = sum;
            }
        }

        return result;
    } 

    */


//    template <size_t P>
//     BatchSwitchMatrixMixed<N, P> operator*(const BatchSwitchMatrixMixed<M, P>& fst) {
//     BatchSwitchMatrixMixed<N, P> result(true);
//     BatchSwitchMatrixMixed<P, M> pom = transposition(fst);

//     char* a_type = new char[N * M];
//     for (size_t i = 0; i < N * M; i++) {
//         a_type[i] = type_mtrx(data[2 * i], data[2 * i + 1]);
//     }

//     char* b_type = new char[P * M];
//     for (size_t i = 0; i < P * M; i++) {
//         b_type[i] = type_mtrx(fst.data[2 * i], fst.data[2 * i + 1]);
//     }

//     // Round down pass
//         capd::rounding::DoubleRounding::roundDown();
//         for (size_t i = 0; i < N; ++i) {
//             for (size_t j = 0; j < P; ++j) {
//                 double sum = 0;
//                 for (size_t k = 0; k < M; ++k) {
//                     size_t idx1 = i * M + k;
//                     size_t idx2 = j * M + k;

//                     char type1 = a_type[idx1];
//                     char type2 = b_type[idx2];

//                     if (type1 == 0 || type2 == 0) {
//                         continue;
//                     }

//                     switch (type1) {
//                         case 1:
//                             if (type2 == 1) sum += data[idx1*2] * pom.data[idx2*2];
//                             else sum += data[idx1*2+1] * pom.data[idx2*2];
//                             break;
//                         case 2:
//                             if (type2 == 2) sum += data[idx1*2+1] * pom.data[idx2*2+1];
//                             else sum += data[idx1*2] * pom.data[idx2*2+1];
//                             break;
//                         default:
//                             switch (type2) {
//                                 case 1:
//                                     sum += data[idx1*2] * pom.data[idx2*2+1];
//                                     break;
//                                 case 2:
//                                     sum += data[idx1*2+1] * pom.data[idx2*2];
//                                     break;
//                                 default:
//                                     double t = data[idx1*2] * pom.data[idx2*2+1];
//                                     double z = data[idx1*2+1] * pom.data[idx2*2];
//                                     sum += std::min(t, z);
//                                     break;
//                             }
//                             break;
//                     }
//                 }
//                 result.data[(i * P + j)*2] = sum;
//             }
//         }

//         // Round up pass
//         capd::rounding::DoubleRounding::roundUp();
//         for (size_t i = 0; i < N; ++i) {
//             for (size_t j = 0; j < P; ++j) {
//                 double sum = 0;
//                 for (size_t k = 0; k < M; ++k) {
//                     size_t idx1 = i * M + k;
//                     size_t idx2 = j * M + k;

//                     char type1 = a_type[idx1];
//                     char type2 = b_type[idx2];

//                     if (type1 == 0 || type2 == 0) {
//                         continue;
//                     }

//                     switch (type1) {
//                         case 1:
//                             if (type2 == 2) sum += data[idx1*2] * pom.data[idx2*2+1];
//                             else sum += data[idx1*2+1] * pom.data[idx2*2+1];
//                             break;
//                         case 2:
//                             if (type2 == 1) sum += data[idx1*2+1] * pom.data[idx2*2];
//                             else sum += data[idx1*2] * pom.data[idx2*2];
//                             break;
//                         default:
//                             switch (type2) {
//                                 case 1:
//                                     sum += data[idx1*2+1] * pom.data[idx2*2+1];
//                                     break;
//                                 case 2:
//                                     sum += data[idx1*2] * pom.data[idx2*2];
//                                     break;
//                                 default:
//                                     double t = data[idx1*2] * pom.data[idx2*2];
//                                     double z = data[idx1*2+1] * pom.data[idx2*2+1];
//                                     sum += std::max(t, z);
//                                     break;
//                             }
//                             break;
//                     }
//                 }
//                 result.data[(i * P + j)*2+1] = sum;
//             }
//         }

//     delete[] a_type;
//     delete[] b_type;

//     return result;
// }

   template <size_t P>
    BatchSwitchMatrixMixed<N, P> operator*(const BatchSwitchMatrixMixed<M, P>& fst) {
    BatchSwitchMatrixMixed<N, P> result{};
    double tmp;
    char* a_type = new char[N * M];
    for (size_t i = 0; i < N * M; i++) {
        a_type[i] = type_mtrx(data[2 * i], data[2 * i + 1]);
    }

    char* b_type = new char[P * M];
    for (size_t i = 0; i < P * M; i++) {
        b_type[i] = type_mtrx(fst.data[2 * i], fst.data[2 * i + 1]);
    }

    // Round down pass
        capd::rounding::DoubleRounding::roundDown();
        for (size_t i = 0; i < N; ++i) {
            for (size_t k = 0; k < M; ++k) {
                size_t idx1 = i * M + k;
                double low =  data[idx1*2];
                double upp = data[idx1*2+1];
                for (size_t j = 0; j < P; ++j) {
                    size_t idx2 = k * P + j;

                    char type1 = a_type[idx1];
                    char type2 = b_type[idx2];

                    if (type1 == 0 || type2 == 0) {
                        continue;
                    }

                    switch (type1) {
                        case 1:
                            if (type2 == 1) tmp = low * fst.data[idx2*2];
                            else tmp =  upp * fst.data[idx2*2];
                            break;
                        case 2:
                            if (type2 == 2) tmp =  upp * fst.data[idx2*2+1];
                            else tmp =  low * fst.data[idx2*2+1];
                            break;
                        default:
                            switch (type2) {
                                case 1:
                                    tmp =  low * fst.data[idx2*2+1];
                                    break;
                                case 2:
                                    tmp =  upp * fst.data[idx2*2];
                                    break;
                                default:
                                    double t = low * fst.data[idx2*2+1];
                                    double z = upp * fst.data[idx2*2];
                                    tmp =  std::min(t, z);
                                    break;
                            }
                            break;
                    }
                    result.data[(i * P + j)*2] = result.data[(i * P + j)*2] + tmp;
                }
            }
        }

        // Round up pass
        capd::rounding::DoubleRounding::roundUp();
        for (size_t i = 0; i < N; ++i) {
            for (size_t k = 0; k < M; ++k) {
                size_t idx1 = i * M + k;
                double low =  data[idx1*2];
                double upp = data[idx1*2+1];
                for (size_t j = 0; j < P; ++j) {
                    size_t idx2 = k * P + j;

                    char type1 = a_type[idx1];
                    char type2 = b_type[idx2];

                    if (type1 == 0 || type2 == 0) {
                        continue;
                    }

                    switch (type1) {
                        case 1:
                            if (type2 == 2) tmp = low * fst.data[idx2*2+1];
                            else tmp = upp * fst.data[idx2*2+1];
                            break;
                        case 2:
                            if (type2 == 1) tmp = upp* fst.data[idx2*2];
                            else tmp = low * fst.data[idx2*2];
                            break;
                        default:
                            switch (type2) {
                                case 1:
                                    tmp = upp * fst.data[idx2*2+1];
                                    break;
                                case 2:
                                    tmp = low * fst.data[idx2*2];
                                    break;
                                default:
                                    double t = low * fst.data[idx2*2];
                                    double z = upp * fst.data[idx2*2+1];
                                    tmp = std::max(t, z);
                                    break;
                            }
                            break;
                    }
                    result.data[(i * P + j)*2+1] = result.data[(i * P + j)*2+1]+tmp;
                }
            }
        }

    delete[] a_type;
    delete[] b_type;

    return result;
}

    BatchSwitchMatrixMixed & operator+=(const BatchSwitchMatrixMixed& fst){
        *this = *this+fst;
        return *this;
    }
    BatchSwitchMatrixMixed & operator-=(const BatchSwitchMatrixMixed& fst){
        *this = *this-fst;
        return *this;
    }

    friend BatchSwitchMatrixMixed operator+(const BatchSwitchMatrixMixed &fst, const Interval & scd){
        BatchSwitchMatrixMixed result(true);
        capd::rounding::DoubleRounding::roundDown();
        for(size_t i = 0; i<2*N*M;i+=2){
            result.data[i] = fst.data[i]+scd.leftBound();
        }
        capd::rounding::DoubleRounding::roundUp();
        for(size_t i = 1; i<2*N*M;i+=2){
            result.data[i] = fst.data[i]+scd.rightBound();
        }
        return result;
    }
    friend BatchSwitchMatrixMixed operator-(const BatchSwitchMatrixMixed &fst, const Interval & scd){
        BatchSwitchMatrixMixed result(true);
        capd::rounding::DoubleRounding::roundDown();
        for(size_t i = 0; i<2*N*M;i+=2){
            result.data[i] = fst.data[i]-scd.rightBound();
        }
        capd::rounding::DoubleRounding::roundUp();
        for(size_t i = 1; i<2*N*M;i+=2){
            result.data[i] = fst.data[i]-scd.leftBound();
        }
        return result;
    }
    friend BatchSwitchMatrixMixed operator*(const BatchSwitchMatrixMixed &fst, const Interval & scd){
        char type_scalar = type(scd.leftBound(),scd.rightBound());
        switch(type_scalar)
        {
            case 0:
                return BatchSwitchMatrixMixed ();
            case 1:{
                BatchSwitchMatrixMixed result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<2*N*M;i+=2){
                    result.data[i] = fst.data[i]*scd.leftBound();
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 1; i<2*N*M;i+=2){
                    result.data[i] = fst.data[i]*scd.leftBound();
                }
                return result;
            }
            case 2:{
                BatchSwitchMatrixMixed result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<2*N*M;i+=2){
                    result.data[i] = fst.data[i+1]*scd.leftBound();
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 1; i<2*N*M;i+=2){
                result.data[i] = fst.data[i-1]*scd.leftBound();
                }
                return result;
            }
            case 3:{
                BatchSwitchMatrixMixed result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<2*N*M;i+=2){
                    if(fst.data[i] < 0) result.data[i] = fst.data[i]*scd.rightBound();
                    else result.data[i] = 0.;
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 1; i<2*N*M;i+=2){
                if(fst.data[i] > 0) result.data[i] = fst.data[i]*scd.rightBound();
                else result.data[i] = 0.;
                }
                return result;
            }
            case 4:{
                BatchSwitchMatrixMixed result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<2*N*M;i+=2){
                    if(fst.data[i] > 0) result.data[i] = fst.data[i]*scd.leftBound();
                    else if(fst.data[i] == 0)result.data[i] = 0.;
                    else result.data[i] = fst.data[i]*scd.rightBound();
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 1; i<2*N*M;i+=2){
                if(fst.data[i] > 0) result.data[i] = fst.data[i]*scd.rightBound();
                else if(fst.data[i] == 0)result.data[i] = 0.;
                else result.data[i] = fst.data[i]*scd.leftBound();
                }
                return result;
            }
            case 5:{
                BatchSwitchMatrixMixed result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<2*N*M;i+=2){
                    if(fst.data[i+1] > 0) result.data[i] = fst.data[i+1]*scd.leftBound();
                    else result.data[i] = 0.;
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 1; i<2*N*M;i+=2){
                if(fst.data[i-1] < 0) result.data[i] = fst.data[i-1]*scd.leftBound();
                else result.data[i] = 0.;
                }
                return result;
            }
            case 6:{
                BatchSwitchMatrixMixed result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<2*N*M;i+=2){
                    if(fst.data[i+1] > 0) result.data[i] = fst.data[i+1]*scd.leftBound();
                    else if(fst.data[i+1] ==0.) result.data[i] = 0.;
                    else result.data[i] = fst.data[i+1]*scd.rightBound();
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 1; i<2*N*M;i+=2){
                    if(fst.data[i-1] > 0) result.data[i] = fst.data[i-1]*scd.rightBound();
                    else if(fst.data[i-1] ==0.) result.data[i] = 0.;
                    else result.data[i] = fst.data[i-1]*scd.leftBound();
                }
                return result;
            }
            case 7:{
                BatchSwitchMatrixMixed result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<2*N*M;i+=2){
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
                for(size_t i = 1; i<2*N*M;i+=2){
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
        return BatchSwitchMatrixMixed ();
    }
    friend BatchSwitchMatrixMixed operator/(const BatchSwitchMatrixMixed &fst, const Interval & scd){
        char type_scalar = type(scd.leftBound(),scd.rightBound());
        switch (type_scalar)
        {
        case 0:
        case 3:
        case 5:
        case 7:
            throw std::invalid_argument("Scalat has 0 in it");
        case 1:{
            BatchSwitchMatrixMixed result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<2*N*M;i+=2){
                result.data[i] = fst.data[i]/scd.leftBound();
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 1; i<2*N*M;i+=2){
                result.data[i] = fst.data[i]/scd.leftBound();
            }
            return result;
        }
        case 2:{
            BatchSwitchMatrixMixed result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<2*N*M;i+=2){
                result.data[i] = fst.data[i+1]/scd.leftBound();
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 1; i<2*N*M;i+=2){
                result.data[i] = fst.data[i-1]/scd.leftBound();
            }
            return result;
        }
        case 4:{
            BatchSwitchMatrixMixed result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<2*N*M;i+=2){
                if(fst.data[i] > 0) result.data[i] = fst.data[i]/scd.rightBound();
                else if(fst.data[i] == 0)result.data[i] = 0.;
                else result.data[i] = fst.data[i]/scd.leftBound();
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 1; i<2*N*M;i+=2){
                if(fst.data[i] > 0) result.data[i] = fst.data[i]/scd.leftBound();
                else if(fst.data[i] == 0)result.data[i] = 0.;
                else result.data[i] = fst.data[i]/scd.rightBound();
            }
            return result;
        }
        case 6:{
            BatchSwitchMatrixMixed result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<2*N*M;i+=2){
                if(fst.data[i+1] > 0) result.data[i] = fst.data[i+1]/scd.rightBound();
                else if(fst.data[i+1] ==0.) result.data[i] = 0.;
                else result.data[i] = fst.data[i+1]/scd.leftBound();
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 1; i<2*N*M;i+=2){
                if(fst.data[i-1] > 0) result.data[i] = fst.data[i-1]/scd.leftBound();
                else if(fst.data[i-1] ==0.) result.data[i] = 0.;
                else result.data[i] = fst.data[i-1]/scd.rightBound();
            }
            return result;
        }
        }
        return BatchSwitchMatrixMixed ();
    }

    BatchSwitchMatrixMixed & operator+=(const Interval& fst){
        *this = *this+fst;
        return *this;
    }
    BatchSwitchMatrixMixed & operator-=(const Interval& fst){
        *this = *this-fst;
        return *this;
    }
    BatchSwitchMatrixMixed & operator*=(const Interval& fst){
        *this = *this * fst;
        return *this;
    }
    BatchSwitchMatrixMixed & operator/=(const Interval& fst){
        *this = *this / fst;
        return *this;
    }

    IntervalProxy<Accessor,size_t> operator()(size_t j, size_t k){
        Accessor acc = {data};
        size_t ind = (j*M+k)*2;
        return IntervalProxy<Accessor,size_t>(acc,ind);
    }
    IntervalProxy<Accessor,size_t> operator()(size_t j, size_t k) const{
        Accessor acc = {data};
        size_t ind = (j*M+k)*2;
        return IntervalProxy<Accessor,size_t>(acc,ind);
    }
    ~BatchSwitchMatrixMixed(){
        delete[] data;
    }
};

template<size_t N, size_t M>
BatchSwitchMatrixMixed<M, N> transposition(const BatchSwitchMatrixMixed<N, M>& fst) {
    BatchSwitchMatrixMixed<M, N> result(true);

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            result.data[2 * (j * N + i)] = fst.data[2 * (i * M + j)];
            result.data[2 * (j * N + i) + 1] = fst.data[2 * (i * M + j) + 1]; 
        }
    }

    return result;
}

#endif