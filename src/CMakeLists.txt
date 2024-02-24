cmake_minimum_required(VERSION 3.22)

find_package(OpenMP REQUIRED)

add_executable(main main.cpp)

target_link_libraries(
        main
        PRIVATE
        OpenMP::OpenMP_CXX
)