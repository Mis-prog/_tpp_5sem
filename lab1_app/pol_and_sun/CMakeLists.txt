cmake_minimum_required(VERSION 3.10)

find_package(MPI REQUIRED)

add_executable(girls main.cpp)

target_include_directories(
        girls
        PRIVATE
        ${MPI_INCLUDE_PATH}
)

target_link_libraries(
        girls
        PRIVATE
        ${MPI_LIBRARIES}
)

set_target_properties(
        girls PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)