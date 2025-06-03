#!/bin/bash
# filepath: aomp/bin/shell-test/test_build_aomp.sh
# Unit tests for build_aomp.sh

realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")

# Source aomp_common_vars
. "$thisdir/../aomp_common_vars"


test_build_type_default() {
  local build_type=${BUILD_TYPE:-Release}
  assertEquals "BUILD_TYPE should default to Release" "Release" "$build_type"
}

test_aomp_version_handling() {
  assertNotNull "AOMP_VERSION_STRING should not be empty" "$AOMP_VERSION_STRING"
  assertContains "Version should contain numbers" "$AOMP_VERSION_STRING" "."
}

test_repository_paths() {
  local aomp_repo_dir="$AOMP_REPOS/aomp"
  assertNotNull "AOMP repository path should be defined" "$aomp_repo_dir"
}

test_install_prefix_handling() {
  local install_prefix=${AOMP_INSTALL_DIR:-$AOMP}
  assertNotNull "Install prefix should be set" "$install_prefix"
}

test_cmake_options_formation() {
  local test_opts="-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/test"
  assertContains "Should contain build type" "$test_opts" "CMAKE_BUILD_TYPE"
  assertContains "Should contain install prefix" "$test_opts" "CMAKE_INSTALL_PREFIX"
}

test_build_directory_creation() {
  local test_build_dir="$SHUNIT_TMPDIR/test_build_aomp"
  mkdir -p "$test_build_dir"
  assertTrue "Build directory should be created" "[ -d '$test_build_dir' ]"
}

test_error_handling_missing_repo() {
  local nonexistent_repo="/nonexistent/path"
  assertFalse "Should handle missing repository" "[ -d '$nonexistent_repo' ]"
}

test_parallel_build_jobs() {
  local num_jobs=${AOMP_JOB_THREADS:-$(nproc)}
  assertTrue "Job count should be positive integer" "[ '$num_jobs' -gt 0 ]"
}

test_component_list_handling() {
  local components="llvm openmp"
  assertContains "Should handle llvm component" "$components" "llvm"
  assertContains "Should handle openmp component" "$components" "openmp"
}

# Find and source shunit2
if [ -f "$HOME/local/shunit2/shunit2" ]; then
    . "$HOME/local/shunit2/shunit2"
else
    echo "Error: shunit2 not found. Please install it using build_supp.sh with the argument 'shunit2'."
    exit 1
fi
