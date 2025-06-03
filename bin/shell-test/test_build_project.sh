#!/bin/bash
# filepath: aomp/bin/shell-test/test_build_project.sh
# Unit tests for build_project.sh

realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")

# Source aomp_common_vars
. "$thisdir/../aomp_common_vars"

# Specific tests for build_project.sh
test_install_project_default() {
    local test_install_project=${INSTALL_PROJECT:-$LLVM_INSTALL_LOC}
    assertEquals "INSTALL_PROJECT should default to LLVM_INSTALL_LOC" "$LLVM_INSTALL_LOC" "$test_install_project"
}

test_targets_to_build_configuration() {
    local old_proc="$AOMP_PROC"
    
    AOMP_PROC="ppc64le"
    local targets_ppc="AMDGPU;${AOMP_NVPTX_TARGET}PowerPC"
    assertContains "PowerPC target should be included for ppc64le" "$targets_ppc" "PowerPC"
    
    AOMP_PROC="aarch64"
    local targets_aarch64="AMDGPU;${AOMP_NVPTX_TARGET}AArch64"
    assertContains "AArch64 target should be included for aarch64" "$targets_aarch64" "AArch64"
    
    AOMP_PROC="x86_64"
    local targets_x86="AMDGPU;${AOMP_NVPTX_TARGET}X86"
    assertContains "X86 target should be included for x86_64" "$targets_x86" "X86"
    
    AOMP_PROC="$old_proc"
}

test_ninja_generator_option() {
    local old_ninja="$AOMP_USE_NINJA"
    
    AOMP_USE_NINJA=0
    local ninja_gen=""
    assertEquals "Ninja generator should be empty when disabled" "" "$ninja_gen"
    
    AOMP_USE_NINJA=1
    local ninja_gen="-G Ninja"
    assertEquals "Ninja generator should be set when enabled" "-G Ninja" "$ninja_gen"
    
    AOMP_USE_NINJA="$old_ninja"
}

test_runtime_configuration() {
    local old_legacy="$AOMP_LEGACY_OPENMP"
    
    AOMP_LEGACY_OPENMP=1
    local runtimes_legacy="libcxx;libcxxabi;libunwind;compiler-rt"
    assertNotContains "Legacy config should not include openmp" "$runtimes_legacy" "openmp"
    
    AOMP_LEGACY_OPENMP=0
    local runtimes_modern="libcxx;libcxxabi;libunwind;openmp;offload;compiler-rt;flang-rt"
    assertContains "Modern config should include openmp" "$runtimes_modern" "openmp"
    assertContains "Modern config should include offload" "$runtimes_modern" "offload"
    
    AOMP_LEGACY_OPENMP="$old_legacy"
}

test_amd_flangrt_option() {
    local old_skip="$AOMP_SKIP_AMD_FLANGRT"
    
    AOMP_SKIP_AMD_FLANGRT="1"
    local amdflangrtopt=""
    assertEquals "AMD Flang runtime should be disabled" "" "$amdflangrtopt"
    
    AOMP_SKIP_AMD_FLANGRT="0"
    local amdflangrtopt="-DFLANG_RT_INCLUDE_AMD=ON"
    assertEquals "AMD Flang runtime should be enabled" "-DFLANG_RT_INCLUDE_AMD=ON" "$amdflangrtopt"
    
    AOMP_SKIP_AMD_FLANGRT="$old_skip"
}

test_quadmath_configuration() {
    local old_proc="$AOMP_PROC"
    
    AOMP_PROC="ppc64le"
    local qmathopt="-DFLANG_RUNTIME_F128_MATH_LIB=libquadmath"
    assertEquals "Quadmath should be enabled for ppc64le" "-DFLANG_RUNTIME_F128_MATH_LIB=libquadmath" "$qmathopt"
    
    AOMP_PROC="aarch64"
    local qmathopt=""
    assertEquals "Quadmath should be disabled for aarch64" "" "$qmathopt"
    
    AOMP_PROC="$old_proc"
}

test_sanitizer_build_option() {
    local old_sanitizer="$AOMP_BUILD_SANITIZER"
    
    AOMP_BUILD_SANITIZER=1
    local sanitizer_opts="-DSANITIZER_AMDGPU=1 -DSANITIZER_HSA_INCLUDE_PATH=$AOMP_REPOS/$AOMP_ROCR_REPO_NAME/runtime/hsa-runtime/inc"
    assertContains "Sanitizer should be enabled" "$sanitizer_opts" "SANITIZER_AMDGPU=1"
    
    AOMP_BUILD_SANITIZER="$old_sanitizer"
}

test_standalone_build_configuration() {
    local old_standalone="$AOMP_STANDALONE_BUILD"
    
    AOMP_STANDALONE_BUILD=1
    local standalone_word="_STANDALONE"
    assertEquals "Standalone word should be set" "_STANDALONE" "$standalone_word"
    
    AOMP_STANDALONE_BUILD=0
    local standalone_word=""
    assertEquals "Standalone word should be empty" "" "$standalone_word"
    
    AOMP_STANDALONE_BUILD="$old_standalone"
}

test_cmake_options_formation() {
    local test_mycmakeopts="-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/test/path"
    assertContains "CMAKE options should contain build type" "$test_mycmakeopts" "CMAKE_BUILD_TYPE=Release"
    assertContains "CMAKE options should contain install prefix" "$test_mycmakeopts" "CMAKE_INSTALL_PREFIX"
}

test_version_string_formation() {
    if [ "$AOMP_STANDALONE_BUILD" == "1" ]; then
        local test_mono_repo_id="abc123def456"
        local test_sourceid="Source ID:$AOMP_VERSION_STRING-$test_mono_repo_id"
        assertContains "Source ID should contain version string" "$test_sourceid" "$AOMP_VERSION_STRING"
        assertContains "Source ID should contain repo ID" "$test_sourceid" "$test_mono_repo_id"
    fi
}

test_rocm_device_libs_path() {
    local rocmdevicelib_loc_new="lib/llvm/lib/clang/$AOMP_MAJOR_VERSION/lib/amdgcn"
    assertContains "Device libs path should contain major version" "$rocmdevicelib_loc_new" "$AOMP_MAJOR_VERSION"
    assertContains "Device libs path should contain amdgcn" "$rocmdevicelib_loc_new" "amdgcn"
}

test_compiler_configuration() {
    local old_proc="$AOMP_PROC"
    
    AOMP_PROC="x86_64"
    local compilers="-DCMAKE_C_COMPILER=$AOMP_CC_COMPILER -DCMAKE_CXX_COMPILER=$AOMP_CXX_COMPILER"
    assertContains "Should use AOMP C compiler" "$compilers" "$AOMP_CC_COMPILER"
    assertContains "Should use AOMP CXX compiler" "$compilers" "$AOMP_CXX_COMPILER"
    
    AOMP_PROC="$old_proc"
}

test_website_variable() {
    local website_var="http\:\/\/github.com\/ROCm-Developer-Tools\/aomp"
    assertContains "Website should contain github URL" "$website_var" "github.com"
    assertContains "Website should contain ROCm-Developer-Tools" "$website_var" "ROCm-Developer-Tools"
}

# Find and source shunit2
if [ -f "$HOME/local/shunit2/shunit2" ]; then
    . "$HOME/local/shunit2/shunit2"
else
    echo "Error: shunit2 not found. Please install it using build_supp.sh with the argument 'shunit2'."
    exit 1
fi
