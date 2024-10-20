cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED PROJECT_NAME OR PROJECT_NAME STREQUAL "")
  get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
endif()

project(${PROJECT_NAME})
message(STATUS "Project name:   '${PROJECT_NAME}'")

if(DEFINED AOMP_DIR AND NOT AOMP_DIR STREQUAL "")
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${AOMP_DIR}")
else()
  # TODO: Handle empty values of AOMP, otherwise 'find_package' will likely fail
  message(STATUS "AOMP directory was not provided")
endif()
message(STATUS "AOMP directory: '${AOMP_DIR}'")

# Retrieve package information from omptest
find_package(omptest REQUIRED)

# Use default compiler values of omptest.
set(CMAKE_C_COMPILER   ${omptest_C_COMPILER})
set(CMAKE_CXX_COMPILER ${omptest_CXX_COMPILER})
message(STATUS "  CMAKE_C_COMPILER: '${CMAKE_C_COMPILER}'")
message(STATUS "CMAKE_CXX_COMPILER: '${CMAKE_CXX_COMPILER}'")

# Make sure to include and esp. link directories before 'add_executable'
include_directories(${omptest_INCLUDE_DIR})
link_directories(${omptest_LIBRARY_ROOT})
add_executable(${PROJECT_NAME}
               ${PROJECT_NAME}.cpp)

# Compile
set(OFFLOAD_OPTIONS "-fopenmp;--offload-arch=${TGT_OFFLOAD_ARCH}")
target_compile_options(${PROJECT_NAME} PRIVATE "-O2;${OFFLOAD_OPTIONS}")
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# Link
# Note: omp[target] will be linked automatically because of the offload options
target_link_libraries(${PROJECT_NAME} PRIVATE ${OFFLOAD_OPTIONS} omptest)

# Notify: this test does not use GoogleTest
message(STATUS "GoogleTest: OFF")
