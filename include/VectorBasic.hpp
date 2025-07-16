#ifndef VECTOR_BASIC_HPP
#define VECTOR_BASIC_HPP

#include <iostream>
#include <stdexcept>
#include <initializer_list>
#include <Interval.hpp>
#include <EqualityComparable.hpp>

template <typename T, size_t N>
class VectorBasic
{
private:
    void allocateImpl(std::true_type) {
        data = static_cast<T*>(operator new[](N * sizeof(T)));
    }

    void allocateImpl(std::false_type) {
        data = new T[N];
    }
    VectorBasic(bool allocateOnly) {
        if (allocateOnly) {
            allocateImpl(std::is_same<T, opt::Interval>{});
        } else {
            data = nullptr;
        }
    }
public:
    

    VectorBasic() {
        data = new T[N]();
    }

    VectorBasic(const T& value) {
        allocateImpl(std::is_same<T, opt::Interval>{});
        for (size_t i = 0; i < N; ++i) {
            data[i] = value;
        }
    }

    VectorBasic(std::initializer_list<T> values) {
        if (values.size() != N) {
            throw std::invalid_argument("Initializer list size must match vector size.");
        }
        allocateImpl(std::is_same<T, opt::Interval>{});
        std::copy(values.begin(), values.end(), data);
    }

    VectorBasic(const VectorBasic& other) {
        allocateImpl(std::is_same<T, opt::Interval>{});
        for (size_t i = 0; i < N; ++i) {
            data[i] = other.data[i];
        }
    }
    VectorBasic(const std::vector<double>& array) {
        if (array.size() != 2 * N) {
            throw std::invalid_argument("Rozmiar wektora musi wynosić dokładnie 2 * N.");
        }
        allocateImpl(std::is_same<T, opt::Interval>{});
        for (size_t i = 0; i < N; ++i) {
            data[i] = T(array[2 * i],array[2 * i + 1]);
        }
    }

    VectorBasic(VectorBasic&& other) noexcept {
        data = other.data;
        other.data = nullptr;
    }

    VectorBasic(const T(&array)[N]) {
        allocateImpl(std::is_same<T, opt::Interval>{});
        std::copy(std::begin(array), std::end(array), data);
    }

    VectorBasic(double* ptr) {
        allocateImpl(std::is_same<T, opt::Interval>{});
        for (size_t i = 0; i < N; ++i) {
            if(ptr[2 * i] <=ptr[2 * i + 1])     data[i] = T(ptr[2 * i], ptr[2 * i + 1]);
            else                                data[i] = T(ptr[2 * i+1], ptr[2 * i]);
        }
    }

    
    friend VectorBasic<T, N> operator+(const VectorBasic<T, N>& fst, const VectorBasic<T, N>& scd) {
        VectorBasic<T, N> res(true);
        for (size_t i = 0; i < N; ++i) {
            res[i] = fst[i] + scd[i];
        }
        return res;
    }
    
    friend VectorBasic<T, N> operator-(const VectorBasic<T, N>& fst, const VectorBasic<T, N>& scd) {
        VectorBasic<T, N> res(true);
        for (size_t i = 0; i < N; ++i) {
            res[i] = fst[i] - scd[i];
        }
        return res;
    }

    VectorBasic<T, N>& operator=(const VectorBasic<T, N>& fst) {
        if (this != &fst) {
            delete[] data;
            data = new T[N];
            for (size_t i = 0; i < N; ++i) {
                data[i] = fst.data[i];
            }
        }
        return *this;
    }

    VectorBasic<T, N>& operator=(VectorBasic<T, N>&& fst) noexcept {
        if (this != &fst) {
            delete[] data;
            data = fst.data;
            fst.data = nullptr;
        }
        return *this;
    }

    VectorBasic<T, N>& operator+=(const VectorBasic<T, N>& fst) {
        for (size_t i = 0; i < N; ++i) {
            data[i] += fst.data[i];
        }
        return *this;
    }

    VectorBasic<T, N>& operator-=(const VectorBasic<T, N>& fst) {
        for (size_t i = 0; i < N; ++i) {
            data[i] -= fst.data[i];
        }
        return *this;
    }

    // Wektor i skalar
    VectorBasic<T, N>& operator=(const T& scl) {
        for (size_t i = 0; i < N; ++i) {
            data[i] = scl;
        }
        return *this;
    }

    VectorBasic<T, N>& operator+=(const T& scl) {
        for (size_t i = 0; i < N; ++i) {
            data[i] += scl;
        }
        return *this;
    }

    VectorBasic<T, N>& operator-=(const T& scl) {
        for (size_t i = 0; i < N; ++i) {
            data[i] -= scl;
        }
        return *this;
    }

    VectorBasic<T, N>& operator*=(const T& scl) {
        std::cout << "c";
        for (size_t i = 0; i < N; ++i) {
            data[i] *= scl;
        }
        return *this;
    }

    VectorBasic<T, N>& operator/=(const T& scl) {
        for (size_t i = 0; i < N; ++i) {
            data[i] /= scl;
        }
        return *this;
    }

    friend VectorBasic<T, N> operator+(const VectorBasic<T, N>& fst, const T& scl) {
        VectorBasic<T, N> res(true);
        for (size_t i = 0; i < N; ++i) {
            res[i] = fst[i] + scl;
        }
        return res;
    }

    friend VectorBasic<T, N> operator-(const VectorBasic<T, N>& fst, const T& scl) {
        VectorBasic<T, N> res(true);
        for (size_t i = 0; i < N; ++i) {
            res[i] = fst[i] - scl;
        }
        return res;
    }

    friend VectorBasic<T, N> operator*(const VectorBasic<T, N>& fst, const T& scl) {
        VectorBasic<T, N> res(true);
        for (size_t i = 0; i < N; ++i) {
           res[i] = fst[i] * scl;
        }
        return res;
    }

    friend VectorBasic<T, N> operator/(const VectorBasic<T, N>& fst, const T& scl) {
        VectorBasic<T, N> res(true);
        for (size_t i = 0; i < N; ++i) {
            res[i] = fst[i] / scl;
        }
        return res;
    }

    template <EqualityComparableWith<T> T2>
    bool operator==(const VectorBasic<T2,N>& fst) const {
        for (size_t i = 0; i < N; ++i) {
            if (!(data[i] == fst[i])) return false;
        }
        return true;
    }

    T& operator[](std::size_t index) {
        if (index >= N) throw std::out_of_range("Index out of bounds");
        return data[index];
    }

    const T& operator[](std::size_t index) const {
        if (index >= N) throw std::out_of_range("Index out of bounds");
        return data[index];
    }

    friend std::ostream& operator<<(std::ostream& os, const VectorBasic<T, N>& vec) {
        os << "( ";
        for (size_t i = 0; i < N; ++i) {
            os << vec.data[i] << " ";
        }
        os << ")";
        return os;
    }

    size_t size() const { return N; }

    ~VectorBasic() {
        delete[] data;
    }

private:
    T* data = nullptr;
};

#endif
