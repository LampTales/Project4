cmake_minimum_required(VERSION 3.9)

SET(ENVIRONMENT "CC")

#add_definitions(-DWITH_NEON)
add_definitions(-DWITH_AVX2)

add_definitions(-mavx2)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(CMAKE_BUILD_TYPE "Release")

# set ( CMAKE_VERBOSE_MAKEFILE on)

# set(CMAKE_CXX_STANDARD 17)

project(proMatrix)

ADD_EXECUTABLE(test test.c Matrix.c proMul.c)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

