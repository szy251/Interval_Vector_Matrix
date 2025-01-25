#ifndef MATRIX_BASIC_HPP
#define MATRIX_BASIC_HPP

#include <iostream>
#include <EqualityComparable.hpp>


template<typename T, size_t N, size_t M>
class MatrixBasic{
    private:

    T* data = nullptr;
    template<typename U, size_t N1, size_t M1>
    friend class MatrixBasic;

    MatrixBasic(bool allocateOnly) {
        if (allocateOnly) {
            data = static_cast<T*>(operator new[](N * M * sizeof(T))); // alokacja pamięci
        } else {
            data = nullptr;
        }
    }
public:
    MatrixBasic(){
        data = new T[N*M]();
    }
    MatrixBasic(const T& value) {
        data = new T[N*M];
        for (size_t i = 0; i < N; ++i) {
            data[i] = value;
        }
    }

    MatrixBasic(std::initializer_list<T> values) {
        if (values.size() != (N*M)) {
            throw std::invalid_argument("Initializer list size must match vector size.");
        }
        data = new T[N*M];
        std::copy(values.begin(), values.end(), data);
    }
    MatrixBasic(double* ptr) {
        data = new T[N*M];
        for (size_t i = 0; i < N*M; ++i) {
            if(ptr[2 * i] <=ptr[2 * i + 1])     data[i] = T(ptr[2 * i], ptr[2 * i + 1]);
            else                                data[i] = T(ptr[2 * i+1], ptr[2 * i]);
        }
    }

    MatrixBasic(const std::vector<double>& vec) {
        if (vec.size() != 2 * N*M) {
            throw std::invalid_argument("Vector size need to conatain 2*N*M doube.");
        }

         data = new T[N*M]; 
         for (size_t i = 0; i < N*M; ++i) {
            data[i] = T(vec[2 * i], vec[2 * i + 1]);
         }
    }

    MatrixBasic(const MatrixBasic& other) {
        data = static_cast<T*>(operator new[](N * M * sizeof(T)));
        for (size_t i = 0; i < N; ++i) {
            data[i] = other.data[i];
        }
    }

    MatrixBasic(MatrixBasic&& other) noexcept {
        data = other.data;
        other.data = nullptr;
    }

    MatrixBasic(const T(&array)[N*M]) {
        data = new T[N*M];
        std::copy(std::begin(array), std::end(array), data);
    }

    ~MatrixBasic() {
        if(!(data==nullptr)) {
            delete[] data;
            data = nullptr;
        }
    }

    void transpose() {
        if constexpr (N == M) {
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = i + 1; j < M; ++j) {
                    std::swap(data[i * N + j], data[j * N + i]);
                }
            }
        } else {
            std::cout << "Transpose not available for non-square matrices." << std::endl;
        }
    }

    template<typename U, size_t N1, size_t M1>
    friend MatrixBasic<U, M1, N1> transposition(const MatrixBasic<U,N1,M1>& fst);

    //Macierze
    friend MatrixBasic<T, N, M> operator+(const MatrixBasic<T, N, M>& fst, const MatrixBasic<T, N, M>& scd) {
        MatrixBasic<T, N, M> res(true);
        for (size_t i = 0; i < N * M; ++i) {
            res.data[i] = fst.data[i] + scd.data[i];
        }
        return res;
    }
    friend MatrixBasic<T, N, M> operator-(const MatrixBasic<T, N, M>& fst, const MatrixBasic<T, N, M>& scd) {
        MatrixBasic<T, N, M> res(true);
        for (size_t i = 0; i < N * M; ++i) {
            res.data[i] = fst.data[i] - scd.data[i];
        }
        return res;
    }

    MatrixBasic<T, N, M>& operator=(const MatrixBasic<T, N, M>& fst) {
        if (this != &fst) {
            delete[] data;  // Zwolnienie starej pamięci
            data = new T[N * M];  // Alokacja nowej pamięci
            for (size_t i = 0; i < N * M; ++i) {
                data[i] = fst.data[i];  // Kopiowanie danych
            }
        }
        return *this;
    }

    MatrixBasic<T, N, M>& operator=(MatrixBasic<T, N, M>&& fst) noexcept {
        if (this != &fst) {
            delete[] data;
            data = fst.data;
            fst.data = nullptr; 
        }
        return *this;
    }
    MatrixBasic<T, N, M>& operator+=(const MatrixBasic<T, N, M>& fst) {
        for (size_t i = 0; i < N * M; ++i) {
            data[i] += fst.data[i];
        }
        return *this;
    }
    MatrixBasic<T, N, M>& operator-=(const MatrixBasic<T, N, M>& fst) {
        for (size_t i = 0; i < N * M; ++i) {
            data[i] -= fst.data[i];
        }
        return *this;
    }
    // template <size_t P>
    // MatrixBasic<T, N, P> operator*(const MatrixBasic<T, M, P>& b) {
    //     MatrixBasic<T, N, P> result(true);
    //     MatrixBasic<T,P,M> pom = transposition(b);

    //     for (size_t i = 0; i < N; ++i) {
    //         for (size_t j = 0; j < P; ++j) {
    //             T sum = T();
    //             for (size_t k = 0; k < M; ++k) {
    //                 sum += (*this)(i, k) * pom(j, k);
    //             }
    //             result(i, j) = sum;
    //         }
    //     }

    //     return result;
    // }

    template <size_t P>
    MatrixBasic<T, N, P> operator*(const MatrixBasic<T, M, P>& b) {
        MatrixBasic<T, N, P> result{};

        for (size_t i = 0; i < N; ++i) {
            for (size_t k = 0; k < M; ++k) {
                T a_ik = (*this)(i, k); // Wyciągamy wartość z pierwszej macierzy, by uniknąć wielokrotnego dostępu
                for (size_t j = 0; j < P; ++j) {
                    result(i, j) += a_ik * b(k, j); // Dodajemy wynik bez potrzeby tymczasowego "sum"
                }
            }
        }

        return result;
    }


    //Macierz i skalar
    MatrixBasic& operator=(const T& scl){
        for(size_t i = 0; i<N*M;i++){
            data[i]=scl;
        }
        return *this;
    }

    MatrixBasic& operator*=(const T& scl){
        for(size_t i = 0; i<N*M;i++){
            data[i]*=scl;
        }
        return *this;
    }
    MatrixBasic& operator/=(const T& scl){
        for(size_t i = 0; i<N*M;i++){
            data[i]/=scl;
        }
        return *this;
    }
    MatrixBasic& operator+=(const T& scl){
        for(size_t i = 0; i<N*M;i++){
            data[i]+=scl;
        }
        return *this;
    }
    MatrixBasic& operator-=(const T& scl){
        for(size_t i = 0; i<N*M;i++){
            data[i]-=scl;
        }
        return *this;
    }
    friend MatrixBasic operator*(const MatrixBasic &fst,const T& scl){
        MatrixBasic result(true);
        for(size_t i = 0; i<N*M;i++){
            result.data[i] = fst.data[i]*scl;
        }
        return result;
    }
    friend MatrixBasic operator/(const MatrixBasic &fst,const T& scl){
        MatrixBasic result(true);
        for(size_t i = 0; i<N*M;i++){
            result.data[i] = fst.data[i]/scl;
        }
        return result;
    }
    friend MatrixBasic operator+(const MatrixBasic &fst,const T& scl){
        MatrixBasic result(true);
        for(size_t i = 0; i<N*M;i++){
            result.data[i] = fst.data[i]+scl;
        }
        return result;
    }
    friend MatrixBasic operator-(const MatrixBasic &fst,const T& scl){
        MatrixBasic result(true);
        for(size_t i = 0; i<N*M;i++){
            result.data[i] = fst.data[i]-scl;
        }
        return result;
    }

    template <EqualityComparableWith<T> T2>
    bool operator==(const MatrixBasic<T2,N,M>& fst) const {
        for (size_t i = 0; i < N*M; ++i) {
            if (!(data[i] == fst[i]))   return false;
        }
        return true;
    }
    
    T& operator()(size_t i, size_t j) {
        return data[i * M + j];
    }

    const T& operator()(size_t i, size_t j) const {
        return data[i * M + j];
    }

    T& operator[](size_t j) {
        return data[j];
    }

    const T& operator[](size_t j) const {
        return data[j];
    }
    static constexpr size_t size() {
        return N * M;
    }
    
};

template<typename T, size_t N, size_t M>
MatrixBasic<T, M, N> transposition(const MatrixBasic<T,N,M>& fst){
    MatrixBasic<T, M, N> result(true);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            result[j * N + i] = fst[i * M + j];
        }
    }
    return result;
}

#endif