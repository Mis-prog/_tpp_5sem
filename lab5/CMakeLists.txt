cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(hello 
    LANGUAGES 
    CXX CUDA
    VERSION 0.0.1
)

enable_language(CUDA)

add_executable(add  src/add.cu)

target_compile_definitions(add PRIVATE -DDEBUG)

add_executable(add2 src/add2.cu)