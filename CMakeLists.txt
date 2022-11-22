cmake_minimum_required(VERSION 3.9)

SET(ENVIRONMENT "CC")

#add_definitions(-DWITH_NEON)
add_definitions(-DWITH_AVX2)

add_definitions(-mavx2)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

set(CMAKE_BUILD_TYPE "Release")

project(proMatrix)

ADD_EXECUTABLE(test test.c Matrix.c proMul.c)

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

link_libraries(-lblas)