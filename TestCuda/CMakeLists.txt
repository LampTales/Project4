cmake_minimum_required(VERSION 3.23)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin/nvcc.exe")

project(TestCuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(TestCuda main.cu)

set_target_properties(TestCuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
