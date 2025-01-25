cmake_minimum_required(VERSION 3.10)
project(IntervalArithmetic)

# Ustaw standard C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx2 -mfma -mavx512f -frounding-math")
# Włączenie OpenMP


# Ścieżki do nagłówków i plików źródłowych
include_directories(include)
file(GLOB SOURCES src/main.cpp src/Interval.cpp)

# Ścieżki do CAPD (zmień, jeśli zainstalowane w niestandardowym miejscu)
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

# Konfiguracja Google Benchmark
find_package(benchmark REQUIRED) # Wymaga, aby benchmark był zainstalowany i dostępny

# Dodanie plików źródłowych do głównego projektu
add_executable(main ${SOURCES})

# Linkowanie z biblioteką CAPD
target_link_libraries(main capd)

# Dodanie benchmarków jako osobnego pliku wykonywalnego
add_executable(benchmark_tests_matrix benchmark/matrix_speed.cpp src/Interval.cpp)
add_executable(benchmark_tests_vector benchmark/vector_speed.cpp src/Interval.cpp)
# Linkowanie z biblioteką Google Benchmark i CAPD
target_link_libraries(benchmark_tests_matrix benchmark::benchmark capd)
target_link_libraries(benchmark_tests_vector benchmark::benchmark capd)
