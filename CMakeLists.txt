cmake_minimum_required(VERSION 3.10)
project(IntervalArithmetic)

# Standard C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx2 -mfma -mavx512f -frounding-math -ftree-vectorize")
# Włączenie OpenMP

find_package(OpenMP)
# Ścieżki do nagłówków i plików źródłowych
include_directories(include)
file(GLOB SOURCES src/main.cpp)

# Ścieżki do CAPD (zmień, jeśli zainstalowane w niestandardowym miejscu)
include_directories(/usr/local/include)
link_directories(/usr/local/lib)


find_package(capd REQUIRED)


# Dodanie plików źródłowych do głównego projektu
add_executable(main ${SOURCES})

# Linkowanie z biblioteką CAPD
target_link_libraries(main capd::capd OpenMP::OpenMP_CXX)


# --- Opcjonalne Benchmarki ---
# Opcja do włączania/wyłączania budowania benchmarków. DOMYŚLNIE WYŁĄCZONE.
option(BUILD_BENCHMARKS "Build the benchmark executables" OFF)


# Buduj benchmarki tylko jeśli opcja jest włączona
if(BUILD_BENCHMARKS)

# Konfiguracja Google Benchmark
find_package(benchmark REQUIRED) # Wymaga, aby benchmark był zainstalowany i dostępny

# Dodanie benchmarków jako osobnego pliku wykonywalnego
add_executable(benchmark_tests_matrix_mlt benchmark/matrix_speed_mlt.cpp)
add_executable(benchmark_tests_matrix_add benchmark/matrix_speed_add.cpp)
add_executable(benchmark_tests_matrix_sub benchmark/matrix_speed_sub.cpp)
add_executable(benchmark_tests_matrix_mlt_scl benchmark/matrix_speed_mlt_scl.cpp)
add_executable(benchmark_tests_matrix_div_scl benchmark/matrix_speed_div_scl.cpp)
add_executable(benchmark_tests_matrix_ctor benchmark/matrix_speed_ctor.cpp)
add_executable(benchmark_tests_matrix_ctor_list benchmark/matrix_speed_ctor_list.cpp)
add_executable(benchmark_tests_matrix_exp benchmark/exp_speed.cpp)
add_executable(benchmark_tests_matrix_exp_fst benchmark/exp_fst_speed.cpp)

add_executable(benchmark_tests_vector_add benchmark/vector_speed_add.cpp)
add_executable(benchmark_tests_vector_sub benchmark/vector_speed_sub.cpp)
add_executable(benchmark_tests_vector_mlt_scl benchmark/vector_speed_mlt_scl.cpp)
add_executable(benchmark_tests_vector_div_scl benchmark/vector_speed_div_scl.cpp)
add_executable(benchmark_tests_vector_ctor benchmark/vector_speed_ctor.cpp)
add_executable(benchmark_tests_vector_ctor_list benchmark/vector_speed_ctor_list.cpp)


add_executable(benchmark_tests_matrix_multicore_ctor benchmark/matrix_avx_multicore_ctor.cpp)
add_executable(benchmark_tests_matrix_multicore_ctor_list benchmark/matrix_avx_multicore_ctor_list.cpp)
add_executable(benchmark_tests_matrix_multicore_add benchmark/matrix_avx_multicore_add.cpp)
add_executable(benchmark_tests_matrix_multicore_sub benchmark/matrix_avx_multicore_sub.cpp)
add_executable(benchmark_tests_matrix_multicore_div_scl benchmark/matrix_avx_multicore_div_scl.cpp)
add_executable(benchmark_tests_matrix_multicore_mlt_scl benchmark/matrix_avx_multicore_mlt_scl.cpp)
add_executable(benchmark_tests_matrix_multicore_mlt benchmark/matrix_avx_multicore_mlt.cpp)

# Linkowanie z biblioteką Google Benchmark i CAPD
target_link_libraries(benchmark_tests_matrix_mlt benchmark::benchmark capd::capd OpenMP::OpenMP_CXX)
target_link_libraries(benchmark_tests_matrix_add benchmark::benchmark capd::capd OpenMP::OpenMP_CXX)
target_link_libraries(benchmark_tests_matrix_sub benchmark::benchmark capd::capd OpenMP::OpenMP_CXX)
target_link_libraries(benchmark_tests_matrix_mlt_scl benchmark::benchmark capd::capd OpenMP::OpenMP_CXX)
target_link_libraries(benchmark_tests_matrix_div_scl benchmark::benchmark capd::capd OpenMP::OpenMP_CXX)
target_link_libraries(benchmark_tests_matrix_ctor benchmark::benchmark capd::capd OpenMP::OpenMP_CXX)
target_link_libraries(benchmark_tests_matrix_ctor_list benchmark::benchmark capd::capd OpenMP::OpenMP_CXX)
target_link_libraries(benchmark_tests_matrix_exp benchmark::benchmark capd::capd OpenMP::OpenMP_CXX)
target_link_libraries(benchmark_tests_matrix_exp_fst benchmark::benchmark capd::capd OpenMP::OpenMP_CXX)


target_link_libraries(benchmark_tests_vector_add benchmark::benchmark capd::capd)
target_link_libraries(benchmark_tests_vector_sub benchmark::benchmark capd::capd)
target_link_libraries(benchmark_tests_vector_mlt_scl benchmark::benchmark capd::capd)
target_link_libraries(benchmark_tests_vector_div_scl benchmark::benchmark capd::capd)
target_link_libraries(benchmark_tests_vector_ctor benchmark::benchmark capd::capd)
target_link_libraries(benchmark_tests_vector_ctor_list benchmark::benchmark capd::capd)


target_link_libraries(benchmark_tests_matrix_multicore_ctor benchmark::benchmark capd::capd OpenMP::OpenMP_CXX)
target_link_libraries(benchmark_tests_matrix_multicore_ctor_list benchmark::benchmark capd::capd OpenMP::OpenMP_CXX)
target_link_libraries(benchmark_tests_matrix_multicore_add benchmark::benchmark capd::capd OpenMP::OpenMP_CXX)
target_link_libraries(benchmark_tests_matrix_multicore_sub benchmark::benchmark capd::capd OpenMP::OpenMP_CXX)
target_link_libraries(benchmark_tests_matrix_multicore_div_scl benchmark::benchmark capd::capd OpenMP::OpenMP_CXX)
target_link_libraries(benchmark_tests_matrix_multicore_mlt_scl benchmark::benchmark capd::capd OpenMP::OpenMP_CXX)
target_link_libraries(benchmark_tests_matrix_multicore_mlt benchmark::benchmark capd::capd OpenMP::OpenMP_CXX)



endif()