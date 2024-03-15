cmake_minimum_required(VERSION 3.10)

file(GLOB  SRC *.cpp)

add_executable(lab1 ${SRC})

set_target_properties(
        lab1 PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)