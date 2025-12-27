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

add_executable(${PROJECT_NAME}
               ${PROJECT_NAME}.cpp)

# Compile
set(OFFLOAD_OPTIONS "-fopenmp;--offload-arch=${TGT_OFFLOAD_ARCH}")
target_compile_options(${PROJECT_NAME} PRIVATE "-O2;${OFFLOAD_OPTIONS}")
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# Link
# Note: omp[target] will be linked automatically because of the offload options
target_link_libraries(${PROJECT_NAME} PRIVATE
                      ${OFFLOAD_OPTIONS} omptest::omptest)

# Notify: Test either uses GoogleTest-based ompTest or 'standalone'
message(STATUS "Standalone: ${LIBOMPTEST_BUILD_STANDALONE}")
