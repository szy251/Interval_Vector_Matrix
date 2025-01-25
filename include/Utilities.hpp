#ifndef UTIL_HPP
#define UTIL_HPP

#include<iostream>
#include <random>

inline char type(double a, double b){
    if(a == 0. && b == 0.) return 0;
    if(a==b && a>0.) return 1;
    if(a==b && a<0.) return 2;
    if(a == 0. && b > 0.) return 3;
    if(a > 0. && b > 0.) return 4;
    if(a < 0. && b == 0.) return 5;
    if(a < 0. && b < 0.) return 6;
    if(a < 0. && b > 0.) return 7;
}
inline char type_mtrx(double a, double b){
    if(a == 0. && b == 0.) return 0;
    if(a >= 0.) return 1;
    if(b <= 0.) return 2;
    return 3;
}

inline std::vector<double> generateArray(int n, unsigned int seed) {
    std::vector<double> result(n * 2);
    std::mt19937 generator(seed); 
    std::uniform_real_distribution<double> dist(-5000.0, 5000.0);

    for (int i = 0; i < n; ++i) {
        double a = dist(generator);
        double b = dist(generator);
        if (a > b) {
            std::swap(a, b);
        }
        result[i * 2] = a;
        result[i * 2 + 1] = b;
    }
    return result;
}

#endif