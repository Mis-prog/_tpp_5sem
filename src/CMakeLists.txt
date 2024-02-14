cmake_minimum_required(VERSION 3.8)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -fopenmp")

find_package(OpenMP REQUIRED)

add_executable(multiplication_matrix multiplication_matrix.cpp)

target_link_libraries(
        multiplication_matrix
        PRIVATE
        OpenMP::OpenMP_CXX
)

set_target_properties(
        multiplication_matrix PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
)
# target_link_libraries(multiplication_matrix PRIVATE OpenMP::::OpenMP_CXX)