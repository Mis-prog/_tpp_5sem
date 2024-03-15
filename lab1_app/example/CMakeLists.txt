cmake_minimum_required(VERSION 3.10)

find_package(MPI REQUIRED)

add_executable(example main.cpp)

target_include_directories(
        example
        PRIVATE
        ${MPI_INCLUDE_PATH}
)

target_link_libraries(
        example
        PRIVATE
        ${MPI_LIBRARIES}
)

set_target_properties(
        example PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)