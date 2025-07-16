// main.cpp
#include <iostream>
#include <iomanip>
#include<capd/filib/Interval.h>
#include <VecMacAll.hpp>

 typedef capd::filib::Interval<double> Interval;


int main() {
    //generowanie danych
    const size_t wymiar = 10;
    auto dane_vec = generateArray(wymiar,42);
    auto dane_mtrx = generateArray(wymiar*wymiar,42);
    auto skalar = Interval(-2.,6.);

    //tworzenie wektorów z wygenerowanych danych
    VectorBasic<Interval,wymiar> a(dane_vec);
    BatchSwitchVectorAVX512_Grouped<wymiar> b(dane_vec);

    //wypisywanie na standardowe wyjćsie
    std::cout << a << std::endl;
    std::cout << b << std::endl;

    //dodawanie wektorów
    auto ga = a+a;
    auto gb = b+b;

    std::cout << std::endl;
    //porównywanie wektorów
    if(ga==gb) std::cout << "Równe wekotry\n";

    //monżenie wektora przez skalar
    gb *= skalar;


    //tworzenie macierzy z wygenerowanych danych
    MatrixBasic<Interval,wymiar,wymiar> c(dane_mtrx);
    BatchSwitchMatrixAVX_Grouped<wymiar,wymiar> d(dane_mtrx);


    //możenie macierzy
    auto gc = c*c;
    auto gd = d*d;

    //eksponenta macierzy
    gc = matrixExp(gc, skalar, 20);
    gd =  matrixExp(gd, skalar, 20);

    //porównanie macierzy
    if(gd==gc) std::cout << "Równe macierze\n";


    return 0;
}
