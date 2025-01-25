#ifndef EQUALITY_COMPARABLE_HPP
#define EQUALITY_COMPARABLE_HPP
#include<iostream>
template<class T1, class T2>
concept EqualityComparableWith = requires(T1 a, T2 b) {
    { a == b } -> std::convertible_to<bool>;
};

#endif 