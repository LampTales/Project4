cmake_minimum_required(VERSION 3.9)

SET(ENVIRONMENT "CC")

#add_definitions(-DWITH_NEON)
add_definitions(-DWITH_AVX2)

add_definitions(-mavx2)

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

set(CMAKE_BUILD_TYPE "Release")

project(proMatrix)

add_compile_options(-mavx2 -O3 -fopenmp)

link_libraries(-lblas -fopenmp)

ADD_EXECUTABLE(test test.c Matrix.c proMul.c)
