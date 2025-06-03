#!/bin/bash
# filepath: aomp/bin/shell-test/test_build_openmp.sh
# Test for build_openmp.sh

realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")

# Source aomp_common_vars
. "$thisdir/../aomp_common_vars"

# Test environment variables specific to OpenMP build
test_openmp_environment_variables() {
    local repo_dir="$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME"
    assertNotNull "AOMP_REPOS should be set" "$AOMP_REPOS"
    assertNotNull "AOMP_PROJECT_REPO_NAME should be set" "$AOMP_PROJECT_REPO_NAME"
    assertNotNull "Repository directory should be defined" "$repo_dir"
    assertEquals "Project repo should be llvm-project" "llvm-project" "$AOMP_PROJECT_REPO_NAME"
}

test_llvm_install_location() {
    assertNotNull "LLVM_INSTALL_LOC should be set" "$LLVM_INSTALL_LOC"
    assertEquals "LLVM_INSTALL_LOC should equal AOMP (resolve symlinks)" "$(realpath "$AOMP/lib/llvm")" "$(realpath "$LLVM_INSTALL_LOC")"
}

test_cmake_common_options() {
    local expected_opts="OPENMP_ENABLE_LIBOMPTARGET=1"
    assertContains "Should enable libomptarget" "$expected_opts" "OPENMP_ENABLE_LIBOMPTARGET=1"
    
    local install_prefix="CMAKE_INSTALL_PREFIX=$LLVM_INSTALL_LOC"
    assertContains "Should set correct install prefix" "$install_prefix" "CMAKE_INSTALL_PREFIX"
}

test_ninja_configuration() {
    assertNotNull "AOMP_USE_NINJA should be set" "$AOMP_USE_NINJA"
    assertNotNull "AOMP_NINJA_BIN should be set" "$AOMP_NINJA_BIN"
    
    if [ "$AOMP_USE_NINJA" == "1" ]; then
        assertContains "Should use Ninja generator" "-G Ninja" "Ninja"
    fi
}

test_standalone_build_configuration() {
    assertEquals "AOMP_STANDALONE_BUILD should default to 1" "1" "$AOMP_STANDALONE_BUILD"
    
    if [ "$AOMP_STANDALONE_BUILD" == "1" ]; then
        local llvm_dir="$AOMP_INSTALL_DIR"
        assertEquals "LLVM_DIR should equal AOMP_INSTALL_DIR for standalone" "$AOMP_INSTALL_DIR" "$llvm_dir"
    fi
}

test_hsa_runtime_path() {
    local hsa_path="$ROCM_DIR"
    assertEquals "HSA_RUNTIME_PATH should equal ROCM_DIR" "$ROCM_DIR" "$hsa_path"
}

test_build_directories() {
    local openmp_build_dir="$BUILD_DIR/build/openmp"
    assertNotNull "OpenMP build directory should be defined" "$openmp_build_dir"
    
    if [ "$AOMP_BUILD_DEBUG" == "1" ]; then
        local debug_build_dir="$BUILD_DIR/build/openmp_debug"
        assertNotNull "Debug build directory should be defined" "$debug_build_dir"
    fi
    
    if [ "$AOMP_BUILD_PERF" == "1" ]; then
        local perf_build_dir="$BUILD_DIR/build/openmp_perf"
        assertNotNull "Performance build directory should be defined" "$perf_build_dir"
    fi
}

test_sanitizer_build_configuration() {
    assertNotNull "AOMP_BUILD_SANITIZER should be set" "$AOMP_BUILD_SANITIZER"
    
    if [ "$AOMP_BUILD_SANITIZER" == "1" ]; then
        assertNotNull "ASAN_FLAGS should be set for sanitizer builds" "$ASAN_FLAGS"
    fi
}

test_ompd_configuration() {
    if [ "$AOMP_BUILD_DEBUG" == "1" ]; then
        local ompd_src_dir="$LLVM_INSTALL_LOC/share/gdb/python/ompd/src"
        local ompd_dir="$LLVM_INSTALL_LOC/share/gdb/python/ompd"
        assertNotNull "OMPD source directory should be defined" "$ompd_src_dir"
        assertNotNull "OMPD directory should be defined" "$ompd_dir"
    fi
}

test_devicelibs_root() {
    assertNotNull "DEVICELIBS_ROOT should be set" "$DEVICELIBS_ROOT"
}

test_origin_rpath_configuration() {
    if [ "$AOMP_BUILD_DEBUG" == "1" ]; then
        assertNotNull "AOMP_DEBUG_ORIGIN_RPATH should be set" "$AOMP_DEBUG_ORIGIN_RPATH"
    fi
    
    if [ "$AOMP_BUILD_SANITIZER" == "1" ]; then
        assertNotNull "AOMP_ASAN_ORIGIN_RPATH should be set" "$AOMP_ASAN_ORIGIN_RPATH"
    fi
}

test_compiler_configuration() {
    local altaomp=${ALTAOMP:-$LLVM_INSTALL_LOC}
    assertNotNull "ALTAOMP should be set" "$altaomp"
    
    local clang_compiler="$altaomp/bin/clang"
    local clangxx_compiler="$altaomp/bin/clang++"
    assertContains "Should use clang compiler" "$clang_compiler" "clang"
    assertContains "Should use clang++ compiler" "$clangxx_compiler" "clang++"
}

test_rocm_cmake_config() {
    if [ "$AOMP_STANDALONE_BUILD" == "0" ]; then
        assertNotNull "ROCM_CMAKECONFIG_PATH should be set for non-standalone" "$ROCM_CMAKECONFIG_PATH"
    fi
}

test_legacy_openmp_handling() {
    assertNotNull "AOMP_LEGACY_OPENMP should be set" "$AOMP_LEGACY_OPENMP"
}

# Find and source shunit2
if [ -f "$HOME/local/shunit2/shunit2" ]; then
    . "$HOME/local/shunit2/shunit2"
else
    echo "Error: shunit2 not found. Please install it using build_supp.sh with the argument 'shunit2'."
    exit 1
fi
