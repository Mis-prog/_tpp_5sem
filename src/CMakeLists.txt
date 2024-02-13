cmake_minimum_required(VERSION 3.25)


find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
add_executable(task1 task1.cpp)
target_link_libraries(task1 PRIVATE OpenMP::OpenMP_CXX)



if (CMAKE_CUDA_COMPILER)
    # message("cuda compliller")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_50")
    enable_language(CUDA)
    add_executable(task2 task2.cu)
    add_executable(task3 task3.cu)
endif()