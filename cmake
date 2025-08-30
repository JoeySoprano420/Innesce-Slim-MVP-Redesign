cmake_minimum_required(VERSION 3.22)
project(innesce LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_library(inrt rt/in_rt.cpp)
llvm_map_components_to_libnames(LLVM_LIBS
  Core OrcJIT MC MCJIT Support IRReader Target
  X86CodeGen X86AsmParser X86AsmPrinter X86Desc X86Info
  Passes Option
)

add_executable(innescec
  src/main.cpp src/lexer.cpp src/parser.cpp src/sema.cpp src/codegen.cpp
)
target_link_libraries(innescec PRIVATE inrt ${LLVM_LIBS})
