#include "Interval.hpp"
#include "internal.cpp"

namespace opt {

Interval::Interval() :values{-0.0,0.0}{}

Interval::Interval(double a, double b):values{-a,b} {}

Interval::Interval(__m128d vec) : values(vec) {}

Interval Interval::operator+(const Interval& other) const {
    return Interval(opt::add(this->values, other.values));
}

Interval Interval::operator-(const Interval& other) const {
    return Interval(opt::sub(this->values,other.values));
}
Interval Interval::operator*(const Interval& other) const{
    return Interval(opt::mult(this->values,other.values));
}
Interval Interval::operator/(const Interval& other) const{
    return Interval(opt::div(this->values,other.values));
}
Interval& Interval::operator+=(const Interval& other){
    this->values = opt::add(this->values, other.values);
    return *this;
}
Interval& Interval::operator-=(const Interval& other){
    this->values = opt::sub(this->values,other.values);
    return *this;
}
Interval& Interval::operator*=(const Interval& other){
    this->values = opt::mult(this->values, other.values);
    return *this;
}
Interval& Interval::operator/=(const Interval& other){
    this->values = opt::div(this->values, other.values);
    return *this;
}

bool Interval::operator==(const Interval &other) const
{
    return equal(this->values,other.values);
}

bool operator==(const Interval& it, const capd::intervals::Interval<double> &other)
{
    __m128d cos = _mm_set_pd(other.rightBound(),(-other.leftBound()));
    return _mm_movemask_pd(_mm_cmpeq_pd(it.values,cos))==3;
}
bool operator==(const capd::intervals::Interval<double> &other, const Interval& it)
{
    __m128d cos = _mm_set_pd(other.rightBound(),(-other.leftBound()));
    return _mm_movemask_pd(_mm_cmpeq_pd(it.values,cos))==3;
}

std::ostream &operator<<(std::ostream &os, const Interval &interval)
{
    os <<"[" << interval.leftBound() << ", " << interval.rightBound() << "]";
    return os;
}

double Interval::leftBound() const
{
    return _leftBound(values);
}
double Interval::rightBound()const {
    return _rightBound(values);
}

void Interval::print() const {
    alignas(16) double result[2];
    _mm_store_pd(result, values);
    std::cout << "[" << -result[0] << ", " << result[1] << "]\n";
}


} 
