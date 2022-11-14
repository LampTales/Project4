cmake_minimum_required(VERSION 3.12)

SET(ENVIRONMENT "CC")

#add_definitions(-DWITH_NEON)
add_definitions(-DWITH_AVX2)

add_definitions(-mavx2)

# add_definitions(-mavx2)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(CMAKE_BUILD_TYPE "Release")

# set(CMAKE_CXX_STANDARD 17)

project(proMatrix)

ADD_EXECUTABLE(test test.c innerTools.c Matrix.c calculate.c proMul.c)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message("OpenMP found.")
    target_link_libraries(test PUBLIC OpenMP::OpenMP_CXX)
endif()

