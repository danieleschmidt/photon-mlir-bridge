# Test Configuration for Photon MLIR Bridge

cmake_minimum_required(VERSION 3.20)

# Test configuration options
option(PHOTON_ENABLE_UNIT_TESTS "Enable unit tests" ON)
option(PHOTON_ENABLE_INTEGRATION_TESTS "Enable integration tests" ON)
option(PHOTON_ENABLE_BENCHMARKS "Enable performance benchmarks" OFF)
option(PHOTON_ENABLE_COVERAGE "Enable code coverage reporting" OFF)

# Find required testing frameworks
find_package(GTest REQUIRED)

if(PHOTON_ENABLE_BENCHMARKS)
    find_package(benchmark REQUIRED)
endif()

if(PHOTON_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    endif()
endif()

# Include directories for tests
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Test utilities library
add_library(photon_test_utils STATIC
    tools/test_utils.cpp
    tools/mock_hardware.cpp
    tools/test_models.cpp
)

target_link_libraries(photon_test_utils
    GTest::gtest
    GTest::gtest_main
)

# Add test subdirectories
if(PHOTON_ENABLE_UNIT_TESTS)
    add_subdirectory(unit)
endif()

if(PHOTON_ENABLE_INTEGRATION_TESTS)
    add_subdirectory(integration)
endif()

if(PHOTON_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Coverage target
if(PHOTON_ENABLE_COVERAGE)
    find_program(GCOV_PATH gcov)
    find_program(LCOV_PATH lcov)
    find_program(GENHTML_PATH genhtml)
    
    if(GCOV_PATH AND LCOV_PATH AND GENHTML_PATH)
        add_custom_target(coverage
            # Reset coverage counters
            COMMAND ${LCOV_PATH} --directory . --zerocounters
            
            # Run tests
            COMMAND ${CMAKE_CTEST_COMMAND} --verbose
            
            # Capture coverage data
            COMMAND ${LCOV_PATH} --directory . --capture --output-file coverage.info
            
            # Remove system and external library coverage
            COMMAND ${LCOV_PATH} --remove coverage.info 
                '/usr/*' 
                '*/external/*' 
                '*/build/*'
                '*/tests/*'
                --output-file coverage_filtered.info
            
            # Generate HTML report
            COMMAND ${GENHTML_PATH} coverage_filtered.info --output-directory coverage_html
            
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating code coverage report"
        )
    else()
        message(WARNING "Coverage tools not found. Coverage target will not be available.")
    endif()
endif()

# Test discovery for CTest
include(GoogleTest)

# Function to add a test with common settings
function(add_photon_test TEST_NAME TEST_SOURCES)
    add_executable(${TEST_NAME} ${TEST_SOURCES})
    
    target_link_libraries(${TEST_NAME}
        photon_test_utils
        GTest::gtest
        GTest::gtest_main
        ${ARGN}  # Additional libraries
    )
    
    # Set test properties
    set_target_properties(${TEST_NAME} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    )
    
    # Discover tests for CTest
    gtest_discover_tests(${TEST_NAME}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        PROPERTIES
            LABELS "unit"
            TIMEOUT 30
    )
endfunction()

# Function to add an integration test
function(add_photon_integration_test TEST_NAME TEST_SOURCES)
    add_executable(${TEST_NAME} ${TEST_SOURCES})
    
    target_link_libraries(${TEST_NAME}
        photon_test_utils
        GTest::gtest
        GTest::gtest_main
        ${ARGN}  # Additional libraries
    )
    
    set_target_properties(${TEST_NAME} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    )
    
    gtest_discover_tests(${TEST_NAME}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        PROPERTIES
            LABELS "integration"
            TIMEOUT 120  # Longer timeout for integration tests
    )
endfunction()

# Function to add a benchmark
function(add_photon_benchmark BENCHMARK_NAME BENCHMARK_SOURCES)
    if(PHOTON_ENABLE_BENCHMARKS)
        add_executable(${BENCHMARK_NAME} ${BENCHMARK_SOURCES})
        
        target_link_libraries(${BENCHMARK_NAME}
            benchmark::benchmark
            benchmark::benchmark_main
            ${ARGN}  # Additional libraries
        )
        
        set_target_properties(${BENCHMARK_NAME} PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks
        )
        
        # Add as test but with benchmark label
        add_test(NAME ${BENCHMARK_NAME}
            COMMAND ${BENCHMARK_NAME} --benchmark_min_time=0.1s
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks
        )
        
        set_tests_properties(${BENCHMARK_NAME} PROPERTIES
            LABELS "benchmark"
            TIMEOUT 300
        )
    endif()
endfunction()

# Test configuration summary
message(STATUS "Test Configuration:")
message(STATUS "  Unit Tests: ${PHOTON_ENABLE_UNIT_TESTS}")
message(STATUS "  Integration Tests: ${PHOTON_ENABLE_INTEGRATION_TESTS}")
message(STATUS "  Benchmarks: ${PHOTON_ENABLE_BENCHMARKS}")
message(STATUS "  Coverage: ${PHOTON_ENABLE_COVERAGE}")

# Test data directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/test_data)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/
     DESTINATION ${CMAKE_BINARY_DIR}/test_data/)

# Custom test targets
add_custom_target(test_all
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS all
    COMMENT "Running all tests"
)

add_custom_target(test_unit
    COMMAND ${CMAKE_CTEST_COMMAND} -L unit --verbose
    DEPENDS all
    COMMENT "Running unit tests"
)

add_custom_target(test_integration
    COMMAND ${CMAKE_CTEST_COMMAND} -L integration --verbose
    DEPENDS all
    COMMENT "Running integration tests"
)

if(PHOTON_ENABLE_BENCHMARKS)
    add_custom_target(benchmarks
        COMMAND ${CMAKE_CTEST_COMMAND} -L benchmark --verbose
        DEPENDS all
        COMMENT "Running benchmarks"
    )
endif()

# Test environment setup
set(ENV{PHOTON_TEST_DATA_DIR} "${CMAKE_BINARY_DIR}/test_data")
set(ENV{PHOTON_TEST_MODE} "1")