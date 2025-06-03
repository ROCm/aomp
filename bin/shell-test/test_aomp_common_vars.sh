#!/bin/bash
# Test for aomp_common_vars

# set -eo pipefail

realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")

# Source the script under test
. "$thisdir/../aomp_common_vars"

# Test basic environment variables
test_aomp_environment_variables() {
    assertNotNull "AOMP should be set" "$AOMP"
    assertNotNull "AOMP_VERSION_STRING should be set" "$AOMP_VERSION_STRING"
    assertNotNull "AOMP_INSTALL_DIR should be set" "$AOMP_INSTALL_DIR"
}

test_rocm_version_default() {
    assertEquals "ROCM_VERSION should default to 6.4.0" "6.4.0" "$ROCM_VERSION"
}

test_compiler_name_default() {
    assertEquals "AOMP_COMPILER_NAME should default to AOMP" "AOMP" "$AOMP_COMPILER_NAME"
}

test_aomp_install_directory_formation() {
    local expected_install="${AOMP}_${AOMP_VERSION_STRING}"
    assertEquals "Install directory should be versioned" "$expected_install" "$AOMP_INSTALL_DIR"
}

test_cmake_detection() {
    assertNotNull "AOMP_CMAKE should be set" "$AOMP_CMAKE"
    assertContains "CMAKE path should contain cmake" "$AOMP_CMAKE" "cmake"
}

test_ccache_configuration() {
    assertNotNull "AOMP_USE_CCACHE should be set" "$AOMP_USE_CCACHE"
    assertTrue "AOMP_USE_CCACHE should be 0 or 1" "[ '$AOMP_USE_CCACHE' -eq 0 ] || [ '$AOMP_USE_CCACHE' -eq 1 ]"
}

test_standalone_build_default() {
    assertEquals "AOMP_STANDALONE_BUILD should default to 1" "1" "$AOMP_STANDALONE_BUILD"
}

test_flang_skip_defaults() {
    assertEquals "AOMP_SKIP_FLANG should default to 1" "1" "$AOMP_SKIP_FLANG"
    assertEquals "AOMP_SKIP_FLANG_NEW should default to 0" "0" "$AOMP_SKIP_FLANG_NEW"
}

test_projects_list_formation() {
    assertNotNull "AOMP_PROJECTS_LIST should be set" "$AOMP_PROJECTS_LIST"
    assertContains "Projects list should contain clang" "$AOMP_PROJECTS_LIST" "clang"
    assertContains "Projects list should contain lld" "$AOMP_PROJECTS_LIST" "lld"
}

test_gfx_list_default() {
    assertNotNull "GFXLIST should be set" "$GFXLIST"
}

test_git_repositories() {
    assertNotNull "GITROC should be set" "$GITROC"
    assertContains "GITROC should contain github" "$GITROC" "github.com"
}

test_repo_names() {
    assertEquals "AOMP_REPO_NAME should default to aomp" "aomp" "$AOMP_REPO_NAME"
    assertEquals "AOMP_PROJECT_REPO_NAME should default to llvm-project" "llvm-project" "$AOMP_PROJECT_REPO_NAME"
}

test_build_cuda_logic() {
    assertNotNull "AOMP_BUILD_CUDA should be set" "$AOMP_BUILD_CUDA"
    assertTrue "AOMP_BUILD_CUDA should be 0 or 1" "[ '$AOMP_BUILD_CUDA' -eq 0 ] || [ '$AOMP_BUILD_CUDA' -eq 1 ]"
}

test_patch_control_file() {
    assertNotNull "AOMP_PATCH_CONTROL_FILE should be set" "$AOMP_PATCH_CONTROL_FILE"
    assertContains "Patch control file should contain patch-control-file" "$AOMP_PATCH_CONTROL_FILE" "patch-control-file"
}

test_patchrepo_function_exists() {
    type patchrepo >/dev/null 2>&1
    assertTrue "patchrepo function should be defined" $?
}

test_removepatch_function_exists() {
    type removepatch >/dev/null 2>&1
    assertTrue "removepatch function should be defined" $?
}

test_getpatchlist_function_exists() {
    type getpatchlist >/dev/null 2>&1
    assertTrue "getpatchlist function should be defined" $?
}

test_setaompgpu_function_exists() {
    type setaompgpu >/dev/null 2>&1
    assertTrue "setaompgpu function should be defined" $?
}

test_gpu_needs_xnack_for_usm_function_exists() {
    type gpu_needs_xnack_for_usm >/dev/null 2>&1
    assertTrue "gpu_needs_xnack_for_usm function should be defined" $?
}

test_is_apu_function_exists() {
    type is_apu >/dev/null 2>&1
    assertTrue "is_apu function should be defined" $?
}

test_help_build_aomp_function_exists() {
    type help_build_aomp >/dev/null 2>&1
    assertTrue "help_build_aomp function should be defined" $?
}

test_sudo_configuration() {
    assertEquals "SUDO should default to empty string" "" "$SUDO"
}

test_build_directory() {
    assertEquals "BUILD_DIR should equal BUILD_AOMP" "$BUILD_AOMP" "$BUILD_DIR"
}

test_rocm_dir_standalone() {
    if [ "$AOMP_STANDALONE_BUILD" == "1" ]; then
        assertEquals "ROCM_DIR should equal AOMP_INSTALL_DIR for standalone" "$AOMP_INSTALL_DIR" "$ROCM_DIR"
    fi
}

test_processor_type_detection() {
    assertNotNull "AOMP_PROC should be set" "$AOMP_PROC"
    assertTrue "AOMP_PROC should be valid architecture" "[ '$AOMP_PROC' = 'x86_64' ] || [ '$AOMP_PROC' = 'ppc64le' ] || [ '$AOMP_PROC' = 'aarch64' ]"
}

test_job_threads_calculation() {
    assertNotNull "AOMP_JOB_THREADS should be set" "$AOMP_JOB_THREADS"
    assertTrue "AOMP_JOB_THREADS should be positive integer" "[ '$AOMP_JOB_THREADS' -gt 0 ]"
}

test_version_string_format() {
    assertContains "Version string should contain major version" "$AOMP_VERSION_STRING" "21"
}

# Find and source shunit2
if [ -f "$HOME/local/shunit2/shunit2" ]; then
    . "$HOME/local/shunit2/shunit2"
else
    echo "Error: shunit2 not found. Please install it using build_supp.sh with the argument 'shunit2'."
    exit 1
fi
