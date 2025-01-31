// main.cpp
#include <iostream>
#include <iomanip>
#include <capd/intervals/Interval.hpp>
#include "VectorBasic.hpp"
#include "MatrixBasic.hpp"
#include <Interval.hpp>
#include <capd/vectalg/Matrix.hpp>
#include<chrono>
#include<random>
#include<BatchSwitchMatrix_Grouped.hpp>
#include<BatchSwitchMatrixMixed.hpp>
#include<BatchSwitchMatrixAVX_Grouped.hpp>
#include<BatchSwitchMatrixAVX512_Grouped.hpp>
#include<Utilities.hpp>

void split_m512d_to_broadcasted_m512d(__m512d vector, __m512d scalars[8]) {
    for (int i = 0; i < 8; ++i) {
        // Wyciągamy odpowiedni element z wektora
        double value = ((double*)&vector)[i]; // Pobieramy wartość `double` z pozycji `i`

        // Tworzymy nowy wektor wypełniony tą wartością
        scalars[i] = _mm512_set1_pd(value);
    }
}



int main() {


    // __m512d vector = _mm512_set_pd(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);

    // // Tablica na 8 nowych wektorów
    // __m512d scalars[8];

    // // Rozdzielamy wektor na 8 wektorów
    // split_m512d_to_broadcasted_m512d(vector, scalars);

    // // Wypisujemy wartości
    // for (int i = 0; i < 8; ++i) {
    //     printf("Vector %d: ", i);
    //     for (int j = 0; j < 8; ++j) {
    //         printf("%f ", ((double*)&scalars[i])[j]);
    //     }
    //     printf("\n");
    // }
    static const size_t N = 300;
    static const size_t M= 300;
    const size_t array_size = N*M; // 10x10 macierz = 100 elementów
    auto intervals = generateArray(array_size,42);
    auto intervals2 = generateArray(array_size,143);
    // Macierze 10x10
    MatrixBasic<capd::intervals::Interval<double>, N, M> matrix1(intervals);
    MatrixBasic<capd::intervals::Interval<double>, M, N> matrix2(intervals2);

    BatchSwitchMatrixAVX512_Grouped<N, M> mtr1(intervals);
    BatchSwitchMatrixAVX512_Grouped<M, N> mtr2(intervals2);

    //BatchSwitchMatrixMixed<10,10> mtr3(intervals);
    //BatchSwitchMatrixMixed<10,10> mtr4(intervals2);

    auto mtr11 = mtr1* mtr2;
    //reorg(mtr1);
    //mtr3 = mtr3* mtr4;
    auto matrix11 = matrix1*matrix2;

    std::cout << std::fixed << std::setprecision(17);

   std::cout << mtr11(0,3) << " " << matrix11(0,3) << std::endl;
    if(mtr11==matrix11){
        std::cout << "dobrze";
    }




    // auto intervals = generateArray(16*16,42);
    // BatchSwitchMatrixAVX512_Grouped<16,16> mtrx(intervals);
    // std::cout << mtrx <<std::endl;
    // std::cout << reorg(mtrx) <<std::endl;
    // std::cout << std::endl;
    // __m256d low = _mm256_set1_pd(7);
    // double a[] {1,2,3,4,5,6,7,8};
    // __m512d vec_lower = _mm512_loadu_pd(a);
    // double jo = _mm256_cvtsd_f64(_mm256_permutex_pd(_mm512_extractf64x4_pd(vec_lower,1),0b00000011));
    // std::cout << jo << " " << jo << " " << a[2] << " " << a[3] << std::endl;
 //     size_t num_elements = 2e6;  // 2 miliony
//     double* arr = new double[num_elements];

//     // Ustawienie generatora liczb losowych
//     std::random_device rd;  // Generator liczb losowych
//     std::mt19937 gen(rd()); // Mersenne Twister (wydajny generator)
//     std::uniform_real_distribution<> dis(-1000.0, 1000.0); // Rozkład losowy w zakresie [-1000, 1000]

//     // Inicjalizowanie tablicy losowymi wartościami
//     for (size_t i = 0; i < num_elements; ++i) {
//         arr[i] = dis(gen);  // Losowa liczba zmiennoprzecinkowa
//     }

//     // Wypisujemy pierwsze 10 elementów tablicy jako przykład
//     for (size_t i = 0; i < 10; ++i) {
//         std::cout << arr[i] << " ";
//     }
//     std::cout << std::endl;

//     opt::Interval hh(1.0,2.0);
//     opt::Interval cc(2.5,3.0);
//     opt::Interval gg = hh*cc;
//     hh.print();
//     // Definicja przedziałów
//     auto A = capd::intervals::Interval<double> ();
//     capd::intervals::Interval<double> B(3.0, 4.0);
//     capd::intervals::Interval<double,capd::rounding::DoubleRounding> e(2.5,3.0);
//     auto b = MatrixBasic<decltype(e),1000,1000>(arr);
//     auto c = MatrixBasic<decltype(cc),1000,1000>(arr);
//     capd::rounding::DoubleRounding l;
//     l.roundUp();
//     auto d = c-c;
//     auto start1 = std::chrono::high_resolution_clock::now();
//     auto result1 = d + c;
//     auto result2 = c * d;
//     auto end1 = std::chrono::high_resolution_clock::now();
//     auto duration1 = duration_cast<std::chrono::microseconds>(end1 - start1);
//     std::cout << "C + C czas: " << duration1.count() << " mikrosekund\n";
//     // Pomiar czasu dla drugiej pętli (e + e)
//      auto u = b-b;
//     auto start2 = std::chrono::high_resolution_clock::now();
//     auto result3 = u+b;
//     auto result4 = b*u;
//     auto end2 = std::chrono::high_resolution_clock::now();
//     auto duration2 = duration_cast<std::chrono::microseconds>(end2 - start2);
//     std::cout << "B + B czas: " << duration2.count() << " mikrosekund\n";
//     delete[] arr;
//    // std::cout << (c==b) << std::endl;
//     // Dodawanie przedziałów
//     capd::intervals::Interval<double> C = A * B;

//     // Wydrukowanie wyniku
//     std::cout << std::fixed << std::setprecision(17);
//     //std::cout << "   e(\"2.5\",\"3\") = " << (result2 == result4)<< std::endl;
//     capd::rounding::DoubleRounding a;
//     a.roundNearest();
//     std::cout << a.test() << std::endl;
//     std::cout << 2.0/3.0 << std::endl;

    return 0;
}
