#!/bin/bash
#
#  build_project.sh:  Script to build the llvm, clang , and lld components of the AOMP compiler.
#                  This clang 9.0 compiler supports clang hip, OpenMP, and clang cuda
#                  offloading languages for BOTH nvidia and Radeon accelerator cards.
#                  This compiler has both the NVPTX and AMDGPU LLVM backends.
#                  The AMDGPU LLVM backend is referred to as the Lightning Compiler.
#
# See the help text below, run 'build_project.sh -h' for more information.
#

# Without these options, we can lose error status from command subtitutions,
# etc.
set -e
shopt -s inherit_errexit

# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath -- "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/aomp_utils"
. "$thisdir/aomp_common_vars"
# --- end standard header ----

# All user-controllable (environment) values are read through these wrappers so
# that they can later be driven by an orchestration layer.
cfgvar() {
  get_config_var_string project "$1"
}

cfgbool() {
  get_config_var_bool project "$1"
}

BUILD_TYPE=${BUILD_TYPE:-Release}
INSTALL_PROJECT=${INSTALL_PROJECT:-$LLVM_INSTALL_LOC}
WEBSITE="http\:\/\/github.com\/ROCm\/aomp"
ROCR_REPO_DIR="$(cfgvar AOMP_REPOS)/$(cfgvar AOMP_ROCR_REPO_NAME)"
REPO_DIR="$(cfgvar AOMP_REPOS)/$(cfgvar AOMP_PROJECT_REPO_NAME)"

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  help_build_aomp
fi

get_src_dir() {
   echo "$REPO_DIR/llvm"
}

# Print the build dir for a given config, passed as $1.
get_build_dir() {
   local Cfg=$1
   case "$Cfg" in
   "default")
     echo -n "$(cfgvar BUILD_DIR)/$(cfgvar AOMP_PROJECT_REPO_NAME)"
     ;;
   *)
     >&2 echo "Unknown config '$Cfg'"
     exit 1
     ;;
   esac
}

# Print the install dir for a given config, passed as $1.
get_install_dir() {
   cfgvar INSTALL_PROJECT
}

# Patch the LLVM version banner in CommandLine.cpp with the AOMP version
# string.  This edits the (git-tracked) source file in place; task_install
# reverts it with "git checkout".
fixup_source_banner() {
   local MONO_REPO_ID SOURCEID TEMPCLFILE ORIGCLFILE BUILDCLFILE VersionString
   VersionString="$(cfgvar AOMP_VERSION_STRING)"
   cd "$REPO_DIR" || exit
   MONO_REPO_ID=$(git log | grep -m1 commit | cut -d" " -f2)
   SOURCEID="Source ID:$VersionString-$MONO_REPO_ID"
   TEMPCLFILE="/tmp/clfile$$.cpp"
   ORIGCLFILE="$REPO_DIR/llvm/lib/Support/CommandLine.cpp"
   BUILDCLFILE=$ORIGCLFILE

   if ! sed "s/LLVM (http:\/\/llvm\.org\/):/AOMP-${VersionString} ($WEBSITE):\\\n $SOURCEID/" "$ORIGCLFILE" > "$TEMPCLFILE"; then
      echo "ERROR sed command to fix CommandLine.cpp failed."
      exit 1
   fi

   if [ -f "$BUILDCLFILE" ] ; then
      # only copy if there has been a change to the source.
      if ! diff "$TEMPCLFILE" "$BUILDCLFILE" >/dev/null; then
         echo "Updating $BUILDCLFILE with corrected $SOURCEID"
         cp "$TEMPCLFILE" "$BUILDCLFILE"
      else
         echo "File $BUILDCLFILE already has correct $SOURCEID"
      fi
   else
      echo "Updating $BUILDCLFILE with $SOURCEID"
      cp "$TEMPCLFILE" "$BUILDCLFILE"
   fi
   rm "$TEMPCLFILE"
}

task_precheck() {
   if "$(cfgbool AOMP_STANDALONE_BUILD)"; then
      local Aomp
      Aomp="$(cfgvar AOMP)"
      if [ ! -L "$Aomp" ] && [ -d "$Aomp" ] ; then
         echo "ERROR: Directory $Aomp is a physical directory."
         echo "       It must be a symbolic link or not exist"
         exit 1
      fi
   fi

   check_writable_installdir "$1" "$(cfgvar INSTALL_PROJECT)"
}

task_patch() {
   # Patch rocr (check-openmp prep).
   patchrepo "$ROCR_REPO_DIR"

   # Patch llvm-project with ATD patch customized for amd-staging.
   # WARNING: This patch (ATD_ASO_full.patch) rarely applies cleanly
   #          because of its size and constant trunk merges to amd-staging.
   #          This is why default is 0 (OFF).
   if "$(cfgbool AOMP_APPLY_ATD_AMD_STAGING_PATCH)"; then
      patchrepo "$REPO_DIR"
   fi
}

task_unpatch() {
   if "$(cfgbool AOMP_APPLY_ATD_AMD_STAGING_PATCH)"; then
      removepatch "$REPO_DIR"
   fi
   removepatch "$ROCR_REPO_DIR"
}

task_clean() {
   local Cfg=$1
   local BuildDir
   BuildDir=$(get_build_dir "$Cfg")
   echo "rm -rf $(shquot "$BuildDir")"
   rm -rf "$BuildDir"
}

task_cmake() {
   local Cfg=$1
   local BuildDir
   local SrcDir
   local Repos
   local AompCmake
   local AompSupp
   local BuildType
   local ProjectsList
   local VersionString
   local CcComp
   local CxxComp
   local Gfxlist
   local standalone_word
   local TARGETS_TO_BUILD
   local LLVM_RUNTIMES
   local rocmdevicelib_loc_new
   local GFXSEMICOLONS
   local -a COMPILERS
   local -a _qmathopt
   local -a _amdflangrtopt
   local -a DO_TESTS_OPTS
   local -a AOMP_SET_NINJA_GEN
   local -a MYCMAKEOPTS
   local -a MYLITOPTS

   BuildDir="$(get_build_dir "$Cfg")"
   SrcDir="$(get_src_dir)"
   Repos="$(cfgvar AOMP_REPOS)"
   AompCmake="$(cfgvar AOMP_CMAKE)"
   AompSupp="$(cfgvar AOMP_SUPP)"
   BuildType="$(cfgvar BUILD_TYPE)"
   ProjectsList="$(cfgvar AOMP_PROJECTS_LIST)"
   VersionString="$(cfgvar AOMP_VERSION_STRING)"
   CcComp="$(cfgvar AOMP_CC_COMPILER)"
   CxxComp="$(cfgvar AOMP_CXX_COMPILER)"

   echo "LLVM PROJECTS TO BUILD:$ProjectsList"

   # Enable AMD-specific Fortran runtime extensions if not skipped
   _amdflangrtopt=(-DFLANG_RT_INCLUDE_AMD=ON)
   if "$(cfgbool AOMP_SKIP_AMD_FLANGRT)"; then
      _amdflangrtopt=()
   fi

   # Enable support for real(kind=16) via libquadmath
   _qmathopt=(-DFLANG_RUNTIME_F128_MATH_LIB=libquadmath)

   if [ "$AOMP_PROC" == "ppc64le" ] ; then
      COMPILERS=(-DCMAKE_C_COMPILER=/usr/bin/gcc-7
                 -DCMAKE_CXX_COMPILER=/usr/bin/g++-7)
      TARGETS_TO_BUILD="AMDGPU;${AOMP_NVPTX_TARGET}PowerPC"
   else
      COMPILERS=(-DCMAKE_C_COMPILER="$CcComp"
                 -DCMAKE_CXX_COMPILER="$CxxComp")
      if [ "$AOMP_PROC" == "aarch64" ] ; then
         TARGETS_TO_BUILD="AMDGPU;${AOMP_NVPTX_TARGET}AArch64;SPIRV"
         _qmathopt=()
      else
         TARGETS_TO_BUILD="AMDGPU;${AOMP_NVPTX_TARGET}X86;SPIRV"
      fi
   fi

   # When building from release source (no git), turn off test items that are not distributed
   # also ubuntu 16.04 only has python 3.5 and lit testing needs 3.6 minimum, so turn off
   # testing with ubuntu 16.04 which goes EOL in April 2021.
   if [ -z ${DO_TESTS+x} ]; then
     DO_TESTS_OPTS=(-DLLVM_BUILD_TESTS=ON
                    -DLLVM_INCLUDE_TESTS=ON
                    -DCLANG_INCLUDE_TESTS=ON)
   else
     # Incoming DO_TESTS is a string with space-separated arguments.  Convert it
     # to an array.
     IFS=" " read -r -a DO_TESTS_OPTS <<< "$DO_TESTS"
   fi
   #-DCOMPILER_RT_INCLUDE_TESTS=OFF"

   if "$(cfgbool AOMP_STANDALONE_BUILD)"; then
      standalone_word="_STANDALONE"
   else
      standalone_word=""
   fi

   if ! "$(cfgbool AOMP_USE_NINJA)" ; then
       AOMP_SET_NINJA_GEN=()
   else
       AOMP_SET_NINJA_GEN=(-G Ninja)
   fi

   if "$(cfgbool AOMP_LEGACY_OPENMP)"; then
     LLVM_RUNTIMES="libcxx;libcxxabi;libunwind;compiler-rt"
   else
     LLVM_RUNTIMES="libcxx;libcxxabi;libunwind;openmp;offload;compiler-rt;flang-rt"
   fi

   rocmdevicelib_loc_new=lib/llvm/lib/clang/$AOMP_MAJOR_VERSION/lib/amdgcn

   Gfxlist="$(cfgvar GFXLIST)"
   GFXSEMICOLONS=$(echo "$Gfxlist" | tr ' ' ';')

   # Settings common to every config.
   MYCMAKEOPTS=(-DCMAKE_BUILD_TYPE="$BuildType"
                -DCMAKE_INSTALL_PREFIX="$(cfgvar INSTALL_PROJECT)"
                -DLLVM_ENABLE_ASSERTIONS=ON
                -DLLVM_TARGETS_TO_BUILD="$TARGETS_TO_BUILD"
                "${COMPILERS[@]}"
                -DLLVM_VERSION_SUFFIX="_AOMP${standalone_word}_$VersionString"
                -DCLANG_VENDOR="AOMP${standalone_word}_$VersionString"
                "${LLVM_FORCE_VC_REVISION_OPT:-}"
                "${LLVM_FORCE_VC_REPOSITORY_OPT:-}"
                -DCLANG_DEFAULT_PIE_ON_LINUX=0
                -DLLVM_ENABLE_ZLIB=ON
                -DBUG_REPORT_URL='https://github.com/ROCm/aomp'
                -DLLVM_ENABLE_BINDINGS=OFF
                -DCMAKE_PREFIX_PATH="$BuildDir/lib/cmake"
                -DLLVM_INCLUDE_BENCHMARKS=OFF
                "${DO_TESTS_OPTS[@]}"
                "${AOMP_ORIGIN_RPATH[@]}"
                -DCLANG_DEFAULT_LINKER=lld
                "${AOMP_SET_NINJA_GEN[@]}"
                "${_qmathopt[@]}"
                "${_amdflangrtopt[@]}"
                -DLIBOMPTARGET_BUILD_DEVICE_FORTRT=ON
                -DLLVM_BUILD_LLVM_DYLIB=ON
                -DLLVM_LINK_LLVM_DYLIB=ON
                -DCLANG_LINK_CLANG_DYLIB=ON
                -DLIBOMPTARGET_EXTERNAL_PROJECT_HSA_PATH="$ROCR_REPO_DIR"
                -DOFFLOAD_EXTERNAL_PROJECT_UNIFIED_ROCR=On
                -DLIBOMPTARGET_EXTERNAL_PROJECT_ROCM_DEVICE_LIBS_PATH="$REPO_DIR/amd/device-libs"
                -DLLVM_EXTERNAL_PROJECTS=SPIRV_TRANSLATOR
                -DLLVM_EXTERNAL_SPIRV_TRANSLATOR_SOURCE_DIR="$Repos/SPIRV-LLVM-Translator"
                -DROCM_DEVICE_LIBS_INSTALL_PREFIX_PATH="$AOMP_INSTALL_DIR"
                -DROCM_DEVICE_LIBS_BITCODE_INSTALL_LOC="$rocmdevicelib_loc_new"
                -DROCM_LLVM_BACKWARD_COMPAT_LINK="$AOMP_INSTALL_DIR/llvm"
                -DROCM_LLVM_BACKWARD_COMPAT_LINK_TARGET="./lib/llvm"
                -DLIBOMP_COPY_EXPORTS=OFF
                -DLIBOMPTARGET_ENABLE_DEBUG=ON
                -DLIBOMPTEST_INSTALL_COMPONENTS=ON
                -DLIBOMPTARGET_AMDGCN_GFXLIST="$GFXSEMICOLONS"
                -DLIBOMP_USE_HWLOC=ON
                -DLIBOMP_HWLOC_INSTALL_DIR="$AompSupp/hwloc"
                -DOPENMP_ENABLE_LIBOMPTARGET=1
                -DLIBOMP_SHARED_LINKER_FLAGS="-Wl,--disable-new-dtags"
                -DLIBOMP_INSTALL_RPATH="$AOMP_ORIGIN_RPATH_LIST"
                -DLIBOMPTARGET_INSTALL_RPATH="$AOMP_ORIGIN_RPATH_LIST"
                -DLIBOMPTARGET_NO_SANITIZER_AMDGPU=1
                -DLIBOMPTARGET_BUILD_DEVICE_FORTRT=On
                -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)

   if [ -f "$REPO_DIR/openmp/device/CMakeLists.txt" ]; then
     MYCMAKEOPTS=("${MYCMAKEOPTS[@]}"
                  -DLLVM_RUNTIME_TARGETS='default;amdgcn-amd-amdhsa'
                  -DRUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES='compiler-rt;libc;libcxx;libcxxabi;flang-rt;openmp'
                  -DRUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON)
   fi

   # -DCLANG_LINK_FLANG_LEGACY=ON

   # Enable amdflang, amdclang, amdclang++, amdllvm.
   # clang-tools-extra added to LLVM_ENABLE_PROJECTS above.
   MYCMAKEOPTS=("${MYCMAKEOPTS[@]}"
                "${AOMP_CCACHE_OPTS[@]}"
                -DLLVM_ENABLE_PROJECTS="$ProjectsList"
                -DCLANG_ENABLE_AMDCLANG=ON
                -DLLVM_ENABLE_RUNTIMES="$LLVM_RUNTIMES"
                -DLIBCXX_ENABLE_STATIC=ON
                -DLIBCXXABI_ENABLE_STATIC=ON
                -DLLVM_RUNTIME_TARGETS="default;amdgcn-amd-amdhsa"
                -DRUNTIMES_amdgcn-amd-amdhsa_FLANG_RT_LIBC_PROVIDER=llvm
                -DRUNTIMES_amdgcn-amd-amdhsa_FLANG_RT_LIBCXX_PROVIDER=llvm
                -DRUNTIMES_amdgcn-amd-amdhsa_CACHE_FILES="$REPO_DIR/compiler-rt/cmake/caches/AMDGPU.cmake;$REPO_DIR/libcxx/cmake/caches/AMDGPU.cmake"
                )

   # Variant-specific settings: enable Compiler-rt Sanitizer Build.
   if "$(cfgbool AOMP_BUILD_SANITIZER)"; then
       MYCMAKEOPTS=("${MYCMAKEOPTS[@]}" -DSANITIZER_AMDGPU=1
                    -DSANITIZER_HSA_INCLUDE_PATH="$ROCR_REPO_DIR/runtime/hsa-runtime/inc"
                    -DSANITIZER_COMGR_INCLUDE_PATH="$REPO_DIR/amd/comgr/include")
   fi

   # Fix the banner to print the AOMP version string.
   if "$(cfgbool AOMP_STANDALONE_BUILD)"; then
      fixup_source_banner
   fi

   mkdir -p "$BuildDir"
   pushd "$BuildDir" >& /dev/null || exit
   echo
   echo " -----Running cmake ---- "
   MYLITOPTS=(-DLLVM_LIT_ARGS='-vv --show-unsupported --show-xfail -j 16')
   echo "$AompCmake" "$(shquot "${MYLITOPTS[@]}")" \
                     "$(shquot "${MYCMAKEOPTS[@]}")" \
                     "$SrcDir"

   if ! "$AompCmake" "${MYLITOPTS[@]}" \
                     "${MYCMAKEOPTS[@]}" \
                     "$SrcDir" 2>&1; then
      echo "ERROR cmake failed. Cmake flags"
      echo "      $(shquot "${MYCMAKEOPTS[@]}")"
      exit 1
   fi
   popd >& /dev/null || exit
}

task_build() {
   local Cfg=$1
   local BuildDir
   local AompCmake
   local Jobs
   local FlangJobs
   BuildDir="$(get_build_dir "$Cfg")"
   AompCmake="$(cfgvar AOMP_CMAKE)"
   Jobs="$(cfgvar AOMP_JOB_THREADS)"

   pushd "$BuildDir" >& /dev/null || exit
   echo
   echo " -----Running make ---- "

   if "$(cfgbool AOMP_LIMIT_FLANG)"; then
      # Required for building flang on memory limited systems.
      FlangJobs="$(cfgvar AOMP_FLANG_THREADS)"
      echo "$AompCmake --build . -- -j $Jobs clang lld compiler-rt"
      "$AompCmake" --build . -- -j "$Jobs" clang lld compiler-rt || true

      echo "$AompCmake --build . -- -j $FlangJobs flang"
      "$AompCmake" --build . -- -j "$FlangJobs" flang || true
   fi

   # Build llvm-project in one step
   echo "Running CMAKE in ${PWD}"
   echo "$AompCmake --build . -j $Jobs"

   if ! "$AompCmake" --build . -j "$Jobs"; then
      echo "ERROR make -j $Jobs failed"
      exit 1
   fi
   popd >& /dev/null || exit
}

task_install() {
   local Cfg=$1
   local BuildDir
   local InstallDir
   local AompCmake
   local Aomp
   local Repos
   local SED_AOMP_REPOS
   local i
   local config_file
   local -a amd_compiler_symlinks
   local -a amd_compiler_cfg

   BuildDir="$(get_build_dir "$Cfg")"
   InstallDir="$(get_install_dir "$Cfg")"
   AompCmake="$(cfgvar AOMP_CMAKE)"
   Aomp="$(cfgvar AOMP)"
   Repos="$(cfgvar AOMP_REPOS)"

   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Installing to $InstallDir ---- "

   if ! $SUDO "$AompCmake" --install .; then
      echo "ERROR make install failed "
      exit 1
   fi
   popd >& /dev/null || exit

   if "$(cfgbool AOMP_STANDALONE_BUILD)"; then
      echo " "
      echo "------ Linking $InstallDir to $Aomp -------"
      if [ -L "$Aomp" ] ; then
         $SUDO rm "$Aomp"
      fi
      $SUDO ln -sf "$AOMP_INSTALL_DIR" "$Aomp"
   fi

   # add executables forgot by make install but needed for testing
   $SUDO cp -p "$BuildDir/bin/llvm-lit" "$LLVM_INSTALL_LOC/bin/llvm-lit"
   # update map_config and llvm_source_root paths in the copied llvm-lit file.
   # Use a sed delimiter ('|') that cannot occur in a path so the slashes in
   # $Repos need no escaping (the old slash-escaping was buggy and broke on
   # ordinary absolute paths).
   sed -i "s|\.\./\.\./\.\./|$Repos/|g" "$LLVM_INSTALL_LOC/bin/llvm-lit"

   $SUDO cp -p "$BuildDir/bin/FileCheck" "$LLVM_INSTALL_LOC/bin/FileCheck"
   $SUDO cp -p "$BuildDir/bin/count" "$LLVM_INSTALL_LOC/bin/count"
   $SUDO cp -p "$BuildDir/bin/not" "$LLVM_INSTALL_LOC/bin/not"
   $SUDO cp -p "$BuildDir/bin/yaml-bench" "$LLVM_INSTALL_LOC/bin/yaml-bench"
   cd "$REPO_DIR" || exit
   git checkout llvm/lib/Support/CommandLine.cpp
   echo
   echo "SUCCESSFUL INSTALL to $InstallDir with link to $Aomp"
   echo

   amd_compiler_symlinks=("amdclang" "amdclang++" "amdclang-cl" "amdclang-cpp" "amdflang" "amdlld")
   amd_compiler_cfg=("clang" "clang++" "clang-cpp" "clang-${AOMP_MAJOR_VERSION}" "clang-cl" "flang")

   # Leaving this in just in case we decide to add the amd* symlinks in the top level bin directory.
   for i in "${amd_compiler_symlinks[@]}"; do
      if [ -f "$LLVM_INSTALL_LOC/bin/$i" ] && [ ! -h "$AOMP_INSTALL_DIR/bin/$i" ]; then
         echo "Creating $i symlink: ${AOMP_INSTALL_DIR}/bin/$i -> ${LLVM_INSTALL_LOC}/bin/$i"
         mkdir -p "${AOMP_INSTALL_DIR}"/bin
         ln -s ../lib/llvm/bin/"$i" "${AOMP_INSTALL_DIR}"/bin/"$i"
      fi
   done

   # rocm.cfg content
   {
      echo "--rocm-path='<CFGDIR>/../../..'"
      echo "-frtlib-add-rpath"
    } > "${LLVM_INSTALL_LOC}/bin/rocm.cfg"

   for i in "${amd_compiler_cfg[@]}"; do
      if [ -f "${LLVM_INSTALL_LOC}/bin/$i" ]; then
         echo "Creating config file: ${i}.cfg in ${LLVM_INSTALL_LOC}/bin"
         config_file="${LLVM_INSTALL_LOC}/bin/${i}.cfg"
         {
            echo "@rocm.cfg"
         } > "$config_file"
         #cp ${LLVM_INSTALL_LOC}/bin/rocm.cfg $config_file
      fi
   done
}

do_list_configs() {
  echo "default"
}

do_list_init() {
  echo "precheck"
  echo "patch"
}

do_list_fini() {
  echo "unpatch"
}

# List of tasks per config.
do_list_tasks() {
  local Cfg=$1
  if valid_config "$Cfg"; then
    echo "clean"
    echo "cmake"
    echo "build"
    echo "install"
  else
    echo "Unknown config '$Cfg'"
  fi
}

command_dispatcher "$@"
