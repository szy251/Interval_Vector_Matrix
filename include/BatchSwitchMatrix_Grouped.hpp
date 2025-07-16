#ifndef MATRIX_BATCH_GRP_HPP
#define MATRIX_BATCH_GRP_HPP

#include <iostream>
#include <capd/rounding/DoubleRounding.h>
#include <capd/filib/Interval.h>
#include <Utilities.hpp>
#include <MatrixBasic.hpp>
#include <BatchSwitchMatrixMixed.hpp>
#include <IntervalProxy.hpp>

template<size_t N, size_t M>
class BatchSwitchMatrix_Grouped
{
private:
    typedef capd::filib::Interval<double> Interval;
    double* low = nullptr; // Dolne wartości
    double* up = nullptr;  // Górne wartości

    template<size_t N1, size_t M1>
    friend class BatchSwitchMatrix_Grouped;

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
public:
    static constexpr size_t rows = N;
    static constexpr size_t cols = M;
    BatchSwitchMatrix_Grouped(bool allocateOnly) {
        if (allocateOnly) {
            low = new double[N*M];
            up = new double[N*M];
        }
    }
    BatchSwitchMatrix_Grouped() {
        low = new double[N*M]();
        up = new double[N*M]();
    }
    BatchSwitchMatrix_Grouped(const BatchSwitchMatrix_Grouped& cpy) {
        low = new double[N*M];
        up = new double[N*M];
        for (size_t i = 0; i < N*M; ++i) {
            low[i] = cpy.low[i];
            up[i] = cpy.up[i];
        }
    }
    BatchSwitchMatrix_Grouped(BatchSwitchMatrix_Grouped&& cpy) noexcept {
        low = cpy.low;
        up = cpy.up;
        cpy.low = nullptr;
        cpy.up = nullptr;
    }
    BatchSwitchMatrix_Grouped(const double(&array)[2 * N*M]) {
        low = new double[N*M];
        up = new double[N*M];
        for (size_t i = 0; i < N*M; ++i) {
            low[i] = array[2 * i];
            up[i] = array[2 * i + 1];
        }
    }
    BatchSwitchMatrix_Grouped(const std::vector<double>& array) {
        if (array.size() != 2 * N * M) {
            throw std::invalid_argument("Vector size need to conatain 2*N*M doube.");
        }
        low = new double[N*M];
        up = new double[N*M];
        for (size_t i = 0; i < N*M; ++i) {
            low[i] = array[2 * i];
            up[i] = array[2 * i + 1];
        }
    }

    BatchSwitchMatrix_Grouped& operator=(const BatchSwitchMatrix_Grouped& fst) {
        if (this != &fst) {
            delete[] low;
            delete[] up;
            low = new double[N*M];
            up = new double[N*M];
            for (size_t i = 0; i < N*M; ++i) {
                low[i] = fst.low[i];
                up[i] = fst.up[i];
            }
        }
        return *this;
    }

    BatchSwitchMatrix_Grouped& operator=(BatchSwitchMatrix_Grouped&& fst) noexcept {
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

    template<size_t N1, size_t M1>
    friend BatchSwitchMatrix_Grouped<M1, N1> transposition(const BatchSwitchMatrix_Grouped<N1,M1>& fst);

    friend BatchSwitchMatrix_Grouped operator+(const BatchSwitchMatrix_Grouped& fst, const BatchSwitchMatrix_Grouped& scd) {
        
        BatchSwitchMatrix_Grouped result(true);
        capd::rounding::DoubleRounding::roundDown();
        for (size_t i = 0; i < N*M; ++i) {
            result.low[i] = fst.low[i] + scd.low[i];
        }
        capd::rounding::DoubleRounding::roundUp();
        for (size_t i = 0; i < N*M; ++i) {
            result.up[i] = fst.up[i] + scd.up[i];
        }
        return result;
    }
    friend BatchSwitchMatrix_Grouped operator-(const BatchSwitchMatrix_Grouped& fst, const BatchSwitchMatrix_Grouped& scd) {
        BatchSwitchMatrix_Grouped result(true);
        capd::rounding::DoubleRounding::roundDown();
        for (size_t i = 0; i < N*M; ++i) {
            result.low[i] = fst.low[i] - scd.up[i];
        }
        capd::rounding::DoubleRounding::roundUp();
        for (size_t i = 0; i < N*M; ++i) {
            result.up[i] = fst.up[i] - scd.low[i];
        }
        return result;
    }
    
    //mnożenie z reorganizacją danych

    // template <size_t P>
    // BatchSwitchMatrix_Grouped<N, P> operator*(const BatchSwitchMatrix_Grouped<M, P>& fst) {
    //     BatchSwitchMatrix_Grouped<N,P> result(true);
    //     BatchSwitchMatrix_Grouped<P,M> pom = transposition(fst);

    //     capd::rounding::DoubleRounding::roundDown();
    //     for (size_t i = 0; i < N; ++i) {
    //         for (size_t j = 0; j < P; ++j) {
    //         double sum = 0;
    //             for (size_t k = 0; k < M; ++k) {
    //                 // Obliczenie wspólnych indeksów
    //                 size_t idx1 = i * M + k;
    //                 size_t idx2 = j * M + k;

    //                 // Pobranie wartości z macierzy
    //                 double a_min = low[idx1];
    //                 double a_max = up[idx1];
    //                 double b_min = fst.low[idx2];
    //                 double b_max = fst.up[idx2];

    //                 // Pomijanie zerowych przedziałów
    //                 if ((a_min == 0 && a_max == 0) || (b_min == 0 && b_max == 0)) {
    //                     continue;
    //                 }

    //                 // Różne przypadki obliczeniowe
    //                 if (a_min >= 0) {
    //                     // Przedział dodatni (++) 
    //                     if (b_min >= 0) {
    //                         sum += a_min * b_min; // (++) * (++)
    //                     } else {
    //                         sum += a_max * b_min; // (++) * (-+), (++) * (--)
    //                     }
    //                 } else if (a_max <= 0) {
    //                     // Przedział ujemny (--)
    //                     if (b_max <= 0) {
    //                         sum += a_max * b_max; // (--) * (--)
    //                     } else {
    //                         sum += a_min * b_max; // (--) * (+-), (--) * (++)
    //                     }
    //                 } else {
    //                     // Przedział mieszany (-+)
    //                     if (b_max <= 0) {
    //                         sum += a_max * b_min; // (-+) * (--)
    //                     } else if (b_min >= 0) {
    //                         sum += a_min * b_max; // (-+) * (++)
    //                     } else {
    //                         // (-+) * (-+), wybór mniejszej wartości
    //                         double t = a_min * b_max;
    //                         double z = a_max * b_min;
    //                         sum += (z < t) ? z : t;
    //                     }
    //                 }
    //             }
    //             result.low[i*P+j] = sum;
    //         }
    //     }


    //      capd::rounding::DoubleRounding::roundUp();
    //     for (size_t i = 0; i < N; ++i) {
    //         for (size_t j = 0; j < P; ++j) {
    //         double sum = 0;
    //             for (size_t k = 0; k < M; ++k) {
    //                 // Obliczenie wspólnych indeksów
    //                 size_t idx1 = i * M + k;
    //                 size_t idx2 = j * M + k;

    //                 // Pobranie wartości z macierzy
    //                 double a_min = low[idx1];
    //                 double a_max = up[idx1];
    //                 double b_min = fst.low[idx2];
    //                 double b_max = fst.up[idx2];

    //                 // Pomijanie zerowych przedziałów
    //                 if ((a_min == 0 && a_max == 0) || (b_min == 0 && b_max == 0)) {
    //                     continue;
    //                 }

    //                 // Różne przypadki obliczeniowe
    //                 if (a_min >= 0) {
    //                     // Przedział dodatni (++) 
    //                     if (b_max <= 0) {
    //                         sum += a_min * b_max; // (++) * (--)
    //                     } else {
    //                         sum += a_max * b_max; // (++) * (-+), (++) * (++)
    //                     }
    //                 } else if (a_max <= 0) {
    //                     // Przedział ujemny (--)
    //                     if (b_min >= 0) {
    //                         sum += a_max * b_min; // (--) * (++)
    //                     } else {
    //                         sum += a_min * b_min; // (--) * (-+), (--) * (--)
    //                     }
    //                 } else {
    //                     // Przedział mieszany (-+)
    //                     if (b_max <= 0) {
    //                         sum += a_min * b_min; // (-+) * (--)
    //                     } else if (b_min >= 0) {
    //                         sum += a_max * b_max; // (-+) * (++)
    //                     } else {
    //                         // (-+) * (-+), wybór większej wartości
    //                         double t = a_min * b_min;
    //                         double z = a_max * b_max;
    //                         sum += (z > t) ? z : t;
    //                     }
    //                 }
    //             }
    //             result.up[i*P+j] = sum;
    //         }
    //     }

    //     return result;
    // } 


    //mnożenie z roarganizacją + preklasyfikacja

    // template <size_t P>
    // BatchSwitchMatrix_Grouped<N, P> operator*(const BatchSwitchMatrix_Grouped<M, P>& fst) {
    //     BatchSwitchMatrix_Grouped<N, P> result(true);
    //     auto pom = transposition(fst);

    //     // Precompute types for matrices
    //     char* a_type = new char[N * M];
    //     char* b_type = new char[P * M];

    //     for (size_t i = 0; i < N * M; ++i) {
    //         a_type[i] = type_mtrx(low[i], up[i]);
    //     }

    //     for (size_t i = 0; i < P * M; ++i) {
    //         b_type[i] = type_mtrx(pom.low[i], pom.up[i]);
    //     }

    //     // Round down pass
    //     capd::rounding::DoubleRounding::roundDown();
    //     for (size_t i = 0; i < N; ++i) {
    //         for (size_t j = 0; j < P; ++j) {
    //             double sum = 0;
    //             for (size_t k = 0; k < M; ++k) {
    //                 size_t idx1 = i * M + k;
    //                 size_t idx2 = j * M + k;

    //                 char type1 = a_type[idx1];
    //                 char type2 = b_type[idx2];

    //                 if (type1 == 0 || type2 == 0) {
    //                     continue;
    //                 }

    //                 switch (type1) {
    //                     case 1:
    //                         if (type2 == 1) sum += low[idx1] * pom.low[idx2];
    //                         else sum += up[idx1] * pom.low[idx2];
    //                         break;
    //                     case 2:
    //                         if (type2 == 2) sum += up[idx1] * pom.up[idx2];
    //                         else sum += low[idx1] * pom.up[idx2];
    //                         break;
    //                     default:
    //                         switch (type2) {
    //                             case 1:
    //                                 sum += low[idx1] * pom.up[idx2];
    //                                 break;
    //                             case 2:
    //                                 sum += up[idx1] * pom.low[idx2];
    //                                 break;
    //                             default:
    //                                 double t = low[idx1] * pom.up[idx2];
    //                                 double z = up[idx1] * pom.low[idx2];
    //                                 sum += std::min(t, z);
    //                                 break;
    //                         }
    //                         break;
    //                 }
    //             }
    //             result.low[i * P + j] = sum;
    //         }
    //     }

    //     // Round up pass
    //     capd::rounding::DoubleRounding::roundUp();
    //     for (size_t i = 0; i < N; ++i) {
    //         for (size_t j = 0; j < P; ++j) {
    //             double sum = 0;
    //             for (size_t k = 0; k < M; ++k) {
    //                 size_t idx1 = i * M + k;
    //                 size_t idx2 = j * M + k;

    //                 char type1 = a_type[idx1];
    //                 char type2 = b_type[idx2];

    //                 if (type1 == 0 || type2 == 0) {
    //                     continue;
    //                 }

    //                 switch (type1) {
    //                     case 1:
    //                         if (type2 == 2) sum += low[idx1] * pom.up[idx2];
    //                         else sum += up[idx1] * pom.up[idx2];
    //                         break;
    //                     case 2:
    //                         if (type2 == 1) sum += up[idx1] * pom.low[idx2];
    //                         else sum += low[idx1] * pom.low[idx2];
    //                         break;
    //                     default:
    //                         switch (type2) {
    //                             case 1:
    //                                 sum += up[idx1] * pom.up[idx2];
    //                                 break;
    //                             case 2:
    //                                 sum += low[idx1] * pom.low[idx2];
    //                                 break;
    //                             default:
    //                                 double t = low[idx1] * pom.low[idx2];
    //                                 double z = up[idx1] * pom.up[idx2];
    //                                 sum += std::max(t, z);
    //                                 break;
    //                         }
    //                         break;
    //                 }
    //             }
    //             result.up[i * P + j] = sum;
    //         }
    //     }

    //     delete[] a_type;
    //     delete[] b_type;
    //     return result;
    // }


    // standardowe 

     template <size_t P>
    BatchSwitchMatrix_Grouped<N, P> operator*(const BatchSwitchMatrix_Grouped<M, P>& fst) {
        BatchSwitchMatrix_Grouped<N, P> result{};
        double tmp;
        // Round down pass
        capd::rounding::DoubleRounding::roundDown();
        for (size_t i = 0; i < N; ++i) {
            for (size_t k = 0; k < M; ++k) {
                size_t idx1 = i * M + k;
                double m_right =  up[idx1];
                double m_left = low[idx1];

                    if(m_right == 0.0 && m_left == 0.0){}
                    else if(m_right <= 0.0)     // (--)
                    {
                        for (size_t j = 0; j < P; ++j){
                            size_t idx2 = k * P +j;
                            if(fst.up[idx2] <= 0.0)   // (--)(--)
                            {
                                tmp= m_right * fst.up[idx2];
                            }
                            else                      // (--)(-+) (--)(++)
                            {
                                tmp = m_left * fst.up[idx2];
                            }
                            result.low[i * P + j] = result.low[i * P + j] + tmp;

                        }
                    }
                    else                // (m_right > 0)
                    if(m_left >= 0.0)   //  (++)
                    {
                        for (size_t j = 0; j < P; ++j){
                            size_t idx2 = k * P +j;
                            if(fst.low[idx2] >= 0.0)   // (++)(++)
                            {
                                tmp = m_left * fst.low[idx2];
                            }
                            else                       // (++)(-+) (++)(--)
                            {
                                tmp = m_right * fst.low[idx2];
                            }
                            result.low[i * P + j] = result.low[i * P + j] + tmp;
                        }
                    }
                    else //  m_left<=0 && m_right>=0 (-+)
                    {
                        for (size_t j = 0; j < P; ++j){
                            size_t idx2 = k * P +j;
                            if(fst.up[idx2] <= 0.0)    // (-+)(--)
                            {
                                tmp = m_right * fst.low[idx2];
                            }
                            else
                            if(fst.low[idx2] >= 0.0)    // (-+)(++)
                            {
                            tmp = m_left * fst.up[idx2];
                            }
                            else                        // (-+)(-+)
                            {
                                double t1 = m_left * fst.up[idx2];
                                double t2 = m_right * fst.low[idx2];
                                tmp = (t1 > t2)? t2 : t1;
                            }
                            result.low[i * P + j] = result.low[i * P + j] + tmp;
                        }
                    }

            }
        }

        // Round up pass
        capd::rounding::DoubleRounding::roundUp();
        for (size_t i = 0; i < N; ++i) {
            for (size_t k = 0; k < M; ++k) {
                size_t idx1 = i * M + k;
                double m_right =  up[idx1];
                double m_left = low[idx1];
                if(m_right == 0.0 && m_left == 0.0){}
                else if (m_right <= 0.0)     // (--)
                {
                    for (size_t j = 0; j < P; ++j){
                        size_t idx2 = k * P +j;
                        if(fst.low[idx2] >= 0.0)  // (--)(++)
                        {
                            tmp = m_right * fst.low[idx2];
                        }
                        else                      // (--)(-+) (--)(--)
                        {
                            tmp = m_left * fst.low[idx2];
                        }
                        result.up[i * P + j] = result.up[i * P + j] + tmp;
                    }
                }
                else                // (m_right > 0)
                if(m_left >= 0.0)   //  (++)
                {
                    for (size_t j = 0; j < P; ++j){
                        size_t idx2 = k * P +j;
                        if(fst.up[idx2] <= 0.0)   // (++)(--)
                        {
                            tmp = m_left * fst.up[idx2];
                        }
                        else                       // (++)(-+) (++)(++)
                        {
                            tmp = m_right * fst.up[idx2];
                        }
                        result.up[i * P + j] = result.up[i * P + j] + tmp;
                    }
                }
                else //  m_left<=0 && m_right>=0 (-+)
                {
                    for (size_t j = 0; j < P; ++j){
                        size_t idx2 = k * P +j;
                        if(fst.up[idx2] <= 0.0)    // (-+)(--)
                        {
                            tmp = m_left * fst.low[idx2];
                        }
                        else
                        if(fst.low[idx2] >= 0.0 )    // (-+)(++)
                        {
                            tmp = m_right * fst.up[idx2];
                        }
                        else                        // (-+)(-+)
                        {
                            double t1 = m_left * fst.low[idx2];
                            double t2 = m_right * fst.up[idx2];
                            tmp = (t1 < t2)? t2 : t1;
                        }
                        result.up[i * P + j] = result.up[i * P + j] + tmp;
                    }
                }
                   
            }
        }
        return result;
    }



    //mnożenie z preklasyfikacją


    //  template <size_t P>
    // BatchSwitchMatrix_Grouped<N, P> operator*(const BatchSwitchMatrix_Grouped<M, P>& fst) {
    //     BatchSwitchMatrix_Grouped<N, P> result{};

    //     // Precompute types for matrices
    //     char* a_type = new char[N * M];
    //     char* b_type = new char[P * M];

    //     for (size_t i = 0; i < N * M; ++i) {
    //         a_type[i] = type_mtrx(low[i], up[i]);
    //     }

    //     for (size_t i = 0; i < P * M; ++i) {
    //         b_type[i] = type_mtrx(fst.low[i], fst.up[i]);
    //     }
    //     double tmp;
    //     // Round down pass
    //     capd::rounding::DoubleRounding::roundDown();
    //     for (size_t i = 0; i < N; ++i) {
    //         for (size_t k = 0; k < M; ++k) {
    //             size_t idx1 = i * M + k;
    //             double lower =  low[idx1];
    //             double upp = up[idx1];
    //             char type1 = a_type[idx1];
    //                 switch (type1) {
    //                     case 1:{
    //                         for (size_t j = 0; j < P; ++j) {
    //                             size_t idx2 = k * P +j;

    //                             char type2 = b_type[idx2];

    //                             // if (type2 == 0) {
    //                             //     continue;
    //                             // }
    //                             if (type2 == 1) tmp = lower * fst.low[idx2];
    //                             else tmp= upp * fst.low[idx2];
    //                             result.low[i * P + j] = result.low[i * P + j] + tmp;
    //                         }
    //                         break;
    //                     }
    //                     case 2:{
    //                         for (size_t j = 0; j < P; ++j) {
    //                             size_t idx2 = k * P +j;

    //                             char type2 = b_type[idx2];

    //                             // if (type2 == 0) {
    //                             //     continue;
    //                             // }
    //                         if (type2 == 2) tmp =  upp * fst.up[idx2];
    //                         else tmp= lower * fst.up[idx2];
    //                         result.low[i * P + j] = result.low[i * P + j] + tmp;
    //                         }
    //                         break;
    //                     }
    //                     case 3:{
    //                         for (size_t j = 0; j < P; ++j) {
    //                             size_t idx2 = k * P +j;

    //                             char type2 = b_type[idx2];

    //                             switch (type2) {
    //                                 case 1:
    //                                     tmp = lower * fst.up[idx2];
    //                                     break;
    //                                 case 2:
    //                                     tmp = upp * fst.low[idx2];
    //                                     break;
    //                                 case 3:{
    //                                     double t = lower * fst.up[idx2];
    //                                     double z = upp * fst.low[idx2];
    //                                     tmp = std::min(t, z);
    //                                     break;
    //                                 }
    //                                 default:
    //                                     continue;
    //                             }
    //                             result.low[i * P + j] = result.low[i * P + j] + tmp;
    //                         }
    //                         break;
    //                     }
    //                 }   
    //         }
    //     }

    //     // Round up pass
    //     capd::rounding::DoubleRounding::roundUp();
    //     for (size_t i = 0; i < N; ++i) {
    //         for (size_t k = 0; k < M; ++k) {
    //             size_t idx1 = i * M + k;
    //             double lower =  low[idx1];
    //             double upp = up[idx1];
    //             char type1 = a_type[idx1];
    //                 switch (type1) {
    //                     case 1:{
    //                         for (size_t j = 0; j < P; ++j) {
    //                             size_t idx2 = k * P +j;

    //                             char type2 = b_type[idx2];

    //                             // if (type2 == 0) {
    //                             //     continue;
    //                             // }
    //                             if (type2 == 2) tmp = lower * fst.up[idx2];
    //                             else tmp = upp * fst.up[idx2];
    //                             result.up[i * P + j] = result.up[i * P + j] + tmp;
    //                         }
    //                         break;
    //                     }
    //                     case 2:{
    //                         for (size_t j = 0; j < P; ++j) {
    //                             size_t idx2 = k * P +j;

    //                             char type2 = b_type[idx2];

    //                             // if (type2 == 0) {
    //                             //     continue;
    //                             // }
    //                             if (type2 == 1) tmp = upp * fst.low[idx2];
    //                             else tmp = lower * fst.low[idx2];
    //                             result.up[i * P + j] = result.up[i * P + j] + tmp;
    //                         }
    //                         break;
    //                     }
    //                     case 3:{
    //                         for (size_t j = 0; j < P; ++j) {
    //                             size_t idx2 = k * P +j;

    //                             char type2 = b_type[idx2];

    //                             switch (type2) {
    //                             case 1:
    //                                 tmp = upp * fst.up[idx2];
    //                                 break;
    //                             case 2:
    //                                 tmp = lower * fst.low[idx2];
    //                                 break;
    //                             case 3:{
    //                                 double t = lower * fst.low[idx2];
    //                                 double z = upp * fst.up[idx2];
    //                                 tmp = std::max(t, z);
    //                                 break;
    //                             }
    //                             default:
    //                                 continue;
    //                             }
    //                             result.up[i * P + j] = result.up[i * P + j] + tmp;
    //                         }
    //                         break;
    //                     }
    //                 }
                    
    //         }
    //     }

    //     delete[] a_type;
    //     delete[] b_type;
    //     return result;
    // }

    BatchSwitchMatrix_Grouped & operator+=(const BatchSwitchMatrix_Grouped& fst){
        *this = *this+fst;
        return *this;
    }
    BatchSwitchMatrix_Grouped & operator-=(const BatchSwitchMatrix_Grouped& fst){
        *this = *this-fst;
        return *this;
    }

    friend BatchSwitchMatrix_Grouped operator+(const BatchSwitchMatrix_Grouped &fst, const Interval & scd){
        BatchSwitchMatrix_Grouped result(true);
        capd::rounding::DoubleRounding::roundDown();
         for (size_t i = 0; i < N*M; ++i) {
            result.low[i] = fst.low[i]+scd.leftBound();
        }
        capd::rounding::DoubleRounding::roundUp();
         for (size_t i = 0; i < N*M; ++i) {
            result.up[i] = fst.up[i]+scd.rightBound();
        }
        return result;
    }
    friend BatchSwitchMatrix_Grouped operator-(const BatchSwitchMatrix_Grouped &fst, const Interval & scd){
        BatchSwitchMatrix_Grouped result(true);
        capd::rounding::DoubleRounding::roundDown();
         for (size_t i = 0; i < N*M; ++i) {
            result.low[i] = fst.low[i]-scd.rightBound();
        }
        capd::rounding::DoubleRounding::roundUp();
        for(size_t i = 1; i<N*M;i+=2){
            result.up[i] = fst.up[i]-scd.leftBound();
        }
        return result;
    }
    friend BatchSwitchMatrix_Grouped operator*(const BatchSwitchMatrix_Grouped &fst, const Interval & scd){
        char type_scalar = type(scd.leftBound(),scd.rightBound());
        switch(type_scalar)
        {
            case 0:
                return BatchSwitchMatrix_Grouped ();
            case 1:{
                BatchSwitchMatrix_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<N*M;i++){
                    result.low[i] = fst.low[i]*scd.leftBound();
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<N*M;i++){
                    result.up[i] = fst.up[i]*scd.leftBound();
                }
                return result;
            }
            case 2:{
                BatchSwitchMatrix_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<N*M;i++){
                    result.low[i] = fst.up[i]*scd.leftBound();
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<N*M;i++){
                result.up[i] = fst.low[i]*scd.leftBound();
                }
                return result;
            }
            case 3:{
                BatchSwitchMatrix_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<N*M;i++){
                    if(fst.low[i] < 0) result.low[i] = fst.low[i]*scd.rightBound();
                    else result.low[i] = 0.;
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<N*M;i++){
                if(fst.up[i] > 0) result.up[i] = fst.up[i]*scd.rightBound();
                else result.up[i] = 0.;
                }
                return result;
            }
            case 4:{
                BatchSwitchMatrix_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<N*M;i++){
                    if(fst.low[i] > 0) result.low[i] = fst.low[i]*scd.leftBound();
                    else if(fst.low[i] == 0)result.low[i] = 0.;
                    else result.low[i] = fst.low[i]*scd.rightBound();
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<N*M;i++){
                if(fst.up[i] > 0) result.up[i] = fst.up[i]*scd.rightBound();
                else if(fst.up[i] == 0)result.up[i] = 0.;
                else result.up[i] = fst.up[i]*scd.leftBound();
                }
                return result;
            }
            case 5:{
                BatchSwitchMatrix_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<N*M;i++){
                    if(fst.up[i] > 0) result.low[i] = fst.up[i]*scd.leftBound();
                    else result.low[i] = 0.;
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<N*M;i++){
                    if(fst.low[i] < 0) result.up[i] = fst.low[i]*scd.leftBound();
                    else result.up[i] = 0.;
                }
                return result;
            }
            case 6:{
                BatchSwitchMatrix_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<N*M;i++){
                    if(fst.up[i] > 0) result.low[i] = fst.up[i]*scd.leftBound();
                    else if(fst.up[i] ==0.) result.low[i] = 0.;
                    else result.low[i] = fst.up[i]*scd.rightBound();
                }
                capd::rounding::DoubleRounding::roundUp();
                for(size_t i = 0; i<N*M;i++){
                    if(fst.low[i] > 0) result.up[i] = fst.low[i]*scd.rightBound();
                    else if(fst.low[i] ==0.) result.up[i] = 0.;
                    else result.up[i] = fst.low[i]*scd.leftBound();
                }
                return result;
            }
            case 7:{
                BatchSwitchMatrix_Grouped result(true);
                capd::rounding::DoubleRounding::roundDown();
                for(size_t i = 0; i<N*M;i++){
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
                for(size_t i = 0; i<N*M;i++){
                    if(fst.low[i] < 0.)
                    {
                        double k = fst.low[i]*scd.leftBound();
                        if(fst.up[i] > 0. ){
                            result.up[i] = std::max(k,fst.up[i]*scd.rightBound());
                        }
                        else result.up[i] = k;
                    }
                    else if(fst.up[i] > 0.) result.up[i] = fst.up[i]*scd.rightBound();
                    else result.up[i] = 0;
                }
                return result;
            }
        }
        return BatchSwitchMatrix_Grouped ();
    }
    bool operator==(const MatrixBasic<Interval,N,M>& fst){
        for(size_t i = 0; i<N*M;i++){
            if(low[i] != fst[i].leftBound()||up[i]!=fst[i].rightBound()) {
                std::cout<< i <<std::endl;
                return false;
            }
        }
        return true;
    }

    bool operator==(const BatchSwitchMatrixMixed<N,M>& fst){
        for(size_t i = 0; i<N;i++){
            for(size_t j = 0; j<M;j++){
                Interval interval = fst(i,j);
                if(low[i*M+j] != interval.leftBound()||up[i*M+j]!=interval.rightBound()) {
                    std::cout<< i << "," << j <<std::endl;
                    return false;
                }
            }
        }
        return true;
    }
    friend BatchSwitchMatrix_Grouped operator/(const BatchSwitchMatrix_Grouped &fst, const Interval & scd){
        char type_scalar = type(scd.leftBound(),scd.rightBound());
        switch (type_scalar)
        {
        case 0:
        case 3:
        case 5:
        case 7:
            throw std::invalid_argument("Scalar has 0 in it");
        case 1:{
            BatchSwitchMatrix_Grouped result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<N*M;i++){
                result.low[i] = fst.low[i]/scd.leftBound();
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 0; i<N*M;i++){
                result.up[i] = fst.up[i]/scd.leftBound();
            }
            return result;
        }
        case 2:{
            BatchSwitchMatrix_Grouped result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<N*M;i++){
                result.low[i] = fst.up[i]/scd.leftBound();
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 0; i<N*M;i++){
                result.up[i] = fst.low[i]/scd.leftBound();
            }
            return result;
        }
        case 4:{
            BatchSwitchMatrix_Grouped result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<N*M;i++){
                if(fst.low[i] > 0) result.low[i] = fst.low[i]/scd.rightBound();
                else if(fst.low[i] == 0)result.low[i] = 0.;
                else result.low[i] = fst.low[i]/scd.leftBound();
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 0; i<N*M;i++){
                if(fst.up[i] > 0) result.up[i] = fst.up[i]/scd.leftBound();
                else if(fst.up[i] == 0)result.up[i] = 0.;
                else result.up[i] = fst.up[i]/scd.rightBound();
            }
            return result;
        }
        case 6:{
            BatchSwitchMatrix_Grouped result(true);
            capd::rounding::DoubleRounding::roundDown();
            for(size_t i = 0; i<N*M;i++){
                if(fst.up[i] > 0) result.low[i] = fst.up[i]/scd.rightBound();
                else if(fst.up[i] ==0.) result.low[i] = 0.;
                else result.low[i] = fst.up[i]/scd.leftBound();
            }
            capd::rounding::DoubleRounding::roundUp();
            for(size_t i = 0; i<N*M;i++){
                if(fst.low[i] > 0) result.up[i] = fst.low[i]/scd.leftBound();
                else if(fst.low[i] ==0.) result.up[i] = 0.;
                else result.up[i] = fst.low[i]/scd.rightBound();
            }
            return result;
        }
        }
        return BatchSwitchMatrix_Grouped ();
    }

    BatchSwitchMatrix_Grouped & operator+=(const Interval& fst){
        *this = *this+fst;
        return *this;
    }
    BatchSwitchMatrix_Grouped & operator-=(const Interval& fst){
        *this = *this-fst;
        return *this;
    }
    BatchSwitchMatrix_Grouped & operator*=(const Interval& fst){
        *this = *this * fst;
        return *this;
    }
    BatchSwitchMatrix_Grouped & operator/=(const Interval& fst){
        *this = *this / fst;
        return *this;
    }

    IntervalProxy<Accessor,size_t> operator()(size_t j, size_t k){
        Accessor acc = {low,up};
        size_t ind = j*M+k;
        return IntervalProxy<Accessor,size_t>(acc,ind);
    }
    IntervalProxy<Accessor,size_t> operator()(size_t j, size_t k) const{
        Accessor acc = {low,up};
        size_t ind = j*M+k;
        return IntervalProxy<Accessor,size_t>(acc,ind);
    }

    friend std::ostream& operator<<(std::ostream & ost, const BatchSwitchMatrix_Grouped & Mtr){
        ost << "[ ";
        for(size_t i = 0; i < N-1; i++){
            for(size_t j=0 ; j< M; j++){
                ost << Mtr(i,j) << " ";
            }
            ost << '\n' << "  ";
        }
        for(size_t j = 0; j< M; j++){
            ost << Mtr(N-1,j) << " ";
        }
        ost<<"]";
        return ost;

    }

    ~BatchSwitchMatrix_Grouped() {
        delete[] low;
        delete[] up;
    }
};

template< size_t N, size_t M>
BatchSwitchMatrix_Grouped<M, N> transposition(const BatchSwitchMatrix_Grouped<N,M>& fst){
    BatchSwitchMatrix_Grouped<M, N> result(true);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            result.low[j * N + i] = fst.low[i * M + j];
            result.up[j * N + i] = fst.up[i * M + j];
        }
    }
    return result;
}

#endif