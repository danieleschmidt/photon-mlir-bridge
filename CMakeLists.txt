cmake_minimum_required(VERSION 3.20)
project(photon-mlir-bridge VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")

# LLVM/MLIR configuration
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

# Compile definitions
add_definitions(${LLVM_DEFINITIONS})

# MLIR tablegen
set(LLVM_TARGET_DEFINITIONS include/photon/dialects/PhotonicOps.td)
mlir_tablegen(PhotonicOps.h.inc -gen-op-decls)
mlir_tablegen(PhotonicOps.cpp.inc -gen-op-defs)
mlir_tablegen(PhotonicDialect.h.inc -gen-dialect-decls)
mlir_tablegen(PhotonicDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(PhotonicOpsIncGen)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tools)

# Testing
option(PHOTON_ENABLE_TESTS "Enable testing" ON)
if(PHOTON_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# Python bindings
option(PHOTON_ENABLE_PYTHON "Enable Python bindings" ON)
if(PHOTON_ENABLE_PYTHON)
    add_subdirectory(python)
endif()

# Documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in 
                   ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(docs
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif()