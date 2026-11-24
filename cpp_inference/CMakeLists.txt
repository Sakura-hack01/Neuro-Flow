cmake_minimum_required(VERSION 3.20)
project(BCIEdgeEngine CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags for microsecond-scale low latency (<5ms target)
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /W4)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra -Wpedantic)
endif()

add_executable(bci_benchmark bci_edge_engine.cpp)

# Output build info
message(STATUS "Configuring BCI Edge Engine with C++${CMAKE_CXX_STANDARD} standard.")
