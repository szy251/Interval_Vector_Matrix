#ifndef INTERVAL_HPP
#define INTERVAL_HPP

#include <immintrin.h>
#include <capd/intervals/Interval.hpp>
#include <iostream>



namespace opt {

class Interval {
public:
    Interval();
    Interval(double a, double b);
    // Interval(const Interval& a) = default;
    // Interval(Interval&& a) = default;
    // ~Interval() = default;

    // Interval& operator=(const Interval& other) = default;
    // Interval& operator=(Interval&& other) = default;

    Interval operator+(const Interval& other) const;
    Interval operator-(const Interval& other) const;
    Interval operator*(const Interval& other) const;
    Interval operator/(const Interval& other) const;
    

    Interval& operator +=(const Interval& other);
    Interval& operator -=(const Interval& other);
    Interval& operator *=(const Interval& other);
    Interval& operator /=(const Interval& other);

    bool operator==(const Interval& other) const;
    friend bool operator==(const Interval& it, const capd::intervals::Interval<double>& other);
    friend bool operator==(const capd::intervals::Interval<double>& other,const Interval& it);
    double leftBound() const;
    double rightBound() const;

    void print() const;
    friend std::ostream& operator<<(std::ostream& os, const Interval& interval);

private:
    __m128d values; 
    Interval(__m128d vec);
};


}
#endif 
