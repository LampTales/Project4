cmake_minimum_required(VERSION 3.9)

SET(ENVIRONMENT "CC")

set(CMAKE_BUILD_TYPE "Release")

project(proMatrix C)

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
    add_definitions(-DWITH_AVX2)
    add_compile_options(-mavx2 -O3 -fopenmp)
    link_libraries(-lblas -fopenmp)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    add_definitions(-DWITH_NEON)
    add_compile_options(-mfpu=neon -mcpu=cortex-a5 -O3 -fopenmp)
    link_libraries(-fopenmp)
endif()

ADD_EXECUTABLE(test test.c Matrix.c proMul.c)
