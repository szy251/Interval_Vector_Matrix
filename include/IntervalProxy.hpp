#ifndef INTERVAL_PROXY_HPP
#define INTERVAL_PROXY_HPP
#include<iostream>
#include<capd/intervals/Interval.hpp>

template<typename Accesor, typename Index>
class IntervalProxy
{
private:
    typedef capd::intervals::Interval<double> Interval;
    Accesor acc;
    Index index;
public:
    IntervalProxy(Accesor & acc, Index & index): acc(acc), index(index) {}
    operator Interval(){
        return acc.get(index);
    }
    IntervalProxy& operator=(const Interval& a){
        acc.set(index,a);
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, const IntervalProxy<Accesor,Index>& proxy) {
        os<<proxy.acc.get(proxy.index);
        return os;
    }
};


#endif