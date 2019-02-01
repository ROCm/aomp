##===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
##===----------------------------------------------------------------------===##
#
# utils/CMakeLists.txt
# Written by Greg Rodgers (Gregory.Rodgers@amd.com)
#
##===----------------------------------------------------------------------===##

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  project(aomputils)
endif()
add_custom_target(aomputils ALL)

find_package(LLVM QUIET CONFIG
  PATHS
  $ENV{AOMP}
  /usr/lib/rocm/aomp
  /opt/rocm/aomp
  NO_DEFAULT_PATH
)

if (LLVM_DIR)
  if("{CMAKE_INSTALL_PREFIX}" STREQUAL "")
     set(CMAKE_INSTALL_PREFIX ${LLVM_INSTALL_PREFIX})
  endif()
  message("")
  message(STATUS "AOMP LLVM Found!")
  message(STATUS "     LLVM_PACKAGE_VERSION:     ${LLVM_PACKAGE_VERSION} ")
  message(STATUS "     LLVM_DIR:                 ${LLVM_DIR}")
  message(STATUS "     LLVM LLVM_INSTALL_PREFIX: ${LLVM_INSTALL_PREFIX}")
  message(STATUS "     CMAKE_INSTALL_PREFIX:     ${CMAKE_INSTALL_PREFIX}")
  message("")
else()
  message("")
  message("  ERROR:  Could not find an AOMP version of LLVM.")
  message("          You must build AOMP before building AOMP utils")
  message("")
  error(" Could not find an AOMP version of LLVM.")
endif()

set(AOMP_DIR_FOUND ${LLVM_DIR})
set(AOMP_INSTALL_PREFIX ${LLVM_INSTALL_PREFIX})
set(AOMP_MAIN_INCDIR ${LLVM_BUILD_MAIN_INCLUDE_DIR})
set(AOMP_BINDIR ${AOMP_INSTALL_PREFIX}/bin)
set(AOMP_INCDIR ${AOMP_INSTALL_PREFIX}/include)
set(AOMP_LIBDIR ${AOMP_INSTALL_PREFIX}/lib)
set(AOMP_LIBS
  ${AOMP_LIBDIR}/libLLVMBitWriter.a
  ${AOMP_LIBDIR}/libLLVMBitReader.a
  ${AOMP_LIBDIR}/libLLVMObject.a
  ${AOMP_LIBDIR}/libLLVMCore.a
  ${AOMP_LIBDIR}/libLLVMDemangle.a
  ${AOMP_LIBDIR}/libLLVMMC.a
  ${AOMP_LIBDIR}/libLLVMMCParser.a
  ${AOMP_LIBDIR}/libLLVMSupport.a
  ${AOMP_LIBDIR}/libLLVMBinaryFormat.a )
mark_as_advanced(
  AOMP_DIR_FOUND
  AOMP_INSTALL_PREFIX
  AOMP_BINDIR
  AOMP_INCDIR
  AOMP_LIBDIR
  AOMP_MAIN_INCDIR)

string(LENGTH ${LLVM_PACKAGE_VERSION} vlen)
math(EXPR vstart "${vlen}-5")
string(SUBSTRING ${LLVM_PACKAGE_VERSION} ${vstart} 5 version_str)

#FIXME:  This could be a loop
add_custom_command( OUTPUT cloc.sh
   COMMAND /bin/cp -p ${CMAKE_CURRENT_SOURCE_DIR}/bin/cloc.sh ${CMAKE_CURRENT_BINARY_DIR}/cloc.sh
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bin/cloc.sh)
# edit in place to preserve permissions
add_custom_command( OUTPUT cloc.sh.phantom
   COMMAND /bin/sed -i -e "s/X\\.Y\\-Z/${version_str}/" ${CMAKE_CURRENT_BINARY_DIR}/cloc.sh > cloc.sh.phantom
   DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/cloc.sh)

add_custom_command( OUTPUT mymcpu
   COMMAND /bin/cp -p ${CMAKE_CURRENT_SOURCE_DIR}/bin/mymcpu ${CMAKE_CURRENT_BINARY_DIR}/mymcpu
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bin/mymcpu)
add_custom_command( OUTPUT mymcpu.phantom
   COMMAND /bin/sed -i -e "s/X\\.Y\\-Z/${version_str}/" ${CMAKE_CURRENT_BINARY_DIR}/mymcpu > mymcpu.phantom
   DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/mymcpu)

add_custom_command( OUTPUT aompversion
   COMMAND /bin/cp -p ${CMAKE_CURRENT_SOURCE_DIR}/bin/aompversion ${CMAKE_CURRENT_BINARY_DIR}/aompversion
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bin/aompversion)
add_custom_command( OUTPUT aompversion.phantom
   COMMAND /bin/sed -i -e "s/X\\.Y\\-Z/${version_str}/" ${CMAKE_CURRENT_BINARY_DIR}/aompversion > aompversion.phantom
   DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/aompversion)

add_custom_command( OUTPUT clang-ocl
   COMMAND /bin/cp -p ${CMAKE_CURRENT_SOURCE_DIR}/bin/clang-ocl ${CMAKE_CURRENT_BINARY_DIR}/clang-ocl
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bin/clang-ocl)
add_custom_command( OUTPUT clang-ocl.phantom
   COMMAND /bin/sed -i -e "s/X\\.Y\\-Z/${version_str}/" ${CMAKE_CURRENT_BINARY_DIR}/clang-ocl > clang-ocl.phantom
   DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/clang-ocl)

# mymcpu reads the codename by searching for pciid from this generated gputable. 
add_custom_command( OUTPUT gputable.txt
   COMMAND /bin/cp -p ${CMAKE_CURRENT_SOURCE_DIR}/bin/gputable.txt ${CMAKE_CURRENT_BINARY_DIR}/gputable.txt
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bin/gputable.txt)

add_custom_target(utils_scripts ALL DEPENDS cloc.sh.phantom mymcpu.phantom aompversion.phantom clang-ocl.phantom gputable.txt)

# We need prepare-builtins in the AOMP distribution to build bc libs for components
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
include(AddLLVM)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fno-rtti")
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIR})
add_executable(prepare-builtins ${CMAKE_CURRENT_SOURCE_DIR}/src/prepare-builtins.cpp)
llvm_map_components_to_libnames(llvm_libs support core bitreader bitwriter)
target_link_libraries(prepare-builtins ${llvm_libs})

install(PROGRAMS 
   ${CMAKE_CURRENT_SOURCE_DIR}/bin/bundle.sh
   ${CMAKE_CURRENT_SOURCE_DIR}/bin/unbundle.sh
   ${CMAKE_CURRENT_BINARY_DIR}/cloc.sh
   ${CMAKE_CURRENT_BINARY_DIR}/mymcpu
   ${CMAKE_CURRENT_BINARY_DIR}/gputable.txt
   ${CMAKE_CURRENT_BINARY_DIR}/aompversion
   ${CMAKE_CURRENT_SOURCE_DIR}/bin/mygpu
   ${CMAKE_CURRENT_SOURCE_DIR}/bin/clang-ocl
   ${CMAKE_CURRENT_BINARY_DIR}/prepare-builtins
   DESTINATION "bin")
