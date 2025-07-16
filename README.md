# Biblioteka Macierzy i Wektorów Przedziałowych

## Opis

Niniejsza biblioteka dostarcza wydajne implementacje macierzy i wektorów przedziałowych. Oferuje kilka wariantów, w tym wersje zoptymalizowane przy użyciu instrukcji **AVX** oraz implementacje oparte na typach generycznych z wykorzystaniem biblioteki **CAPD**.

## Wymagania

Przed przystąpieniem do kompilacji upewnij się, że Twoje środowisko spełnia następujące wymagania:

- **Procesor:** Wymagana obsługa instrukcji **AVX512**.
- **Biblioteka CAPD:** Niezbędna do kompilacji i uruchomienia głównego przykładu.
- **Biblioteka Google Benchmark:** Wymagana do uruchomienia testów wydajności.
- **Kompilator C++** z obsługą standardu C++20 (lub nowszego).
- **CMake:** Wersja 3.10 lub nowsza.


## Kompilacja

1. Utórz kalaog build i przejdż do niego:

```bash
mkdir build & cd build
```

2. Wygeneruj plik budowania:

```bash
cmake ..
```

3. Skompiluj:

```bash
make
```

## Użycie

### Przykład

W pliku `main.cpp` znajduje się prosty przykład demonstrujący podstawowe użycie biblioteki.

- Aby dołączyć wszystkie niezbędne nagłówki, możesz użyć jednego pliku:
  ```cpp
  #include "VecMacAll.hpp"
  ```

- Możesz dołączyć wszystkie nagłówki oddzielnie

- By uruchmić przykład z  `main.cpp ` należy uruchmić program:
```bahs
./main
```


### Benchmarki

Domyślnie są wyłączone, by je uruchmić należy wgenerować plik budowania z odopowiednią opcją:
```bash
cmake .. -DBUILD_BENCHMARKS=ON
```
