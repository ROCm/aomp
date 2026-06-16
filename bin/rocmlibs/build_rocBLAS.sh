#!/bin/bash
#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
# 
# 
#  build_rocblas.sh:  Script to build and install rocblas library
#
#

# Without these options, we can lose error status from command subtitutions,
# etc.
set -e
shopt -s inherit_errexit

# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath -- "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/../aomp_utils"
. "$thisdir/../aomp_common_vars"
# --- end standard header ----

_repo_dir=$AOMP_REPOS/rocmlibs/rocBLAS
_tensile_repo_dir=$AOMP_REPOS/rocmlibs/Tensile

AOMP_BUILD_TENSILE=${AOMP_BUILD_TENSILE:-1}
ROCBLAS_USE_HIPBLASLT=${ROCBLAS_USE_HIPBLASLT:-0}

export CC=$LLVM_INSTALL_LOC/bin/amdclang
export CXX=$LLVM_INSTALL_LOC/bin/amdclang++
export FC=$LLVM_INSTALL_LOC/bin/amdflang
export ROCM_DIR=$AOMP_INSTALL_DIR
export ROCM_PATH=$AOMP_INSTALL_DIR
export PATH=$AOMP_SUPP/cmake/bin:$AOMP_INSTALL_DIR/bin:$AOMP/llvm/bin:$PATH
export HIP_USE_PERL_SCRIPTS=1
export USE_PERL_SCRIPTS=1
export CXXFLAGS="-I$AOMP_INSTALL_DIR/include -D__HIP_PLATFORM_AMD__=1"
export LDFLAGS="-fPIC"

cfgvar() {
  get_config_var_string rocblas "$1"
}

cfgbool() {
  get_config_var_bool rocblas "$1"
}

get_src_dir() {
   echo "$_repo_dir"
}

# Print the build dir for a given config, passed as $1.
get_build_dir() {
   local Cfg=$1
   case "$Cfg" in
   "default")
     echo -n "$(cfgvar BUILD_DIR)/rocmlibs/rocBLAS"
     ;;
   *)
     >&2 echo "Unknown config '$Cfg'"
     exit 1
     ;;
   esac
}

# Print the install dir for a given config, passed as $1.
get_install_dir() {
   echo "$AOMP_INSTALL_DIR"
}

# rocBLAS does not support an incremental (nocmake) rebuild.
do_nocmake() {
   echo "ERROR: nocmake is not an option for $0 because we use rmake.py"
   exit 1
}

task_precheck() {
   if ! "$(cfgbool AOMP_STANDALONE_BUILD)"; then
      echo "ERROR: $0 only valid for AOMP_STANDALONE_BUILD=1"
      exit 1
   fi
   if [ ! -L "$AOMP" ] && [ -d "$AOMP" ] ; then
      echo "ERROR: Directory $AOMP is a physical directory."
      echo "       It must be a symbolic link or not exist"
      exit 1
   fi

   check_writable_installdir "$1" "$AOMP_INSTALL_DIR"
}

task_patch() {
   local _tensile_commit_sha
   if "$(cfgbool AOMP_BUILD_TENSILE)"; then
      cd "$_tensile_repo_dir" || exit
      # Read the commit SHA from the file rocBLAS/tensile_tag.txt
      _tensile_commit_sha=$(cat "$_repo_dir/tensile_tag.txt")
      echo "Checking out Tensile commit $_tensile_commit_sha"
      git checkout "$_tensile_commit_sha"
      patchrepo "$_tensile_repo_dir"
   fi
   patchrepo "$_repo_dir"
}

task_unpatch() {
   removepatch "$_repo_dir"
   if "$(cfgbool AOMP_BUILD_TENSILE)"; then
      removepatch "$_tensile_repo_dir"
   fi
}

task_clean() {
   local Cfg=$1
   local BuildDir
   BuildDir=$(get_build_dir "$Cfg")
   echo "rm -rf $(shquot "$BuildDir")"
   rm -rf "$BuildDir"
   if "$(cfgbool AOMP_BUILD_TENSILE)"; then
      # Cleanup possible old tensile build area
      echo "rm -rf $_tensile_repo_dir/build"
      rm -rf "$_tensile_repo_dir/build"
   fi
}

task_cmake() {
   local Cfg=$1
   local BuildDir
   local SrcDir
   local -a MYCMAKEOPTS

   BuildDir="$(get_build_dir "$Cfg")"
   SrcDir="$(get_src_dir)"

   if ! "$(cfgbool AOMP_BUILD_TENSILE)"; then
      echo
      echo "WARNING: Building rocblas without Tensile"
   fi
   if ! "$(cfgbool ROCBLAS_USE_HIPBLASLT)"; then
      echo
      echo "WARNING: Building rocblas without hipBLASLT"
   fi

   MYCMAKEOPTS=(-DCMAKE_TOOLCHAIN_FILE=toolchain-linux.cmake
                -DCMAKE_CXX_COMPILER="$CXX"
                -DCMAKE_C_COMPILER="$CC"
                -DROCM_DIR:PATH="$AOMP_INSTALL_DIR"
                -DCPACK_PACKAGING_INSTALL_PREFIX="$AOMP_INSTALL_DIR"
                -DCMAKE_INSTALL_PREFIX="$AOMP_INSTALL_DIR"
                -DROCM_PATH="$AOMP_INSTALL_DIR"
                -DCMAKE_PREFIX_PATH:PATH="$AOMP_INSTALL_DIR"
                -DCPACK_SET_DESTDIR=OFF
                -DCMAKE_BUILD_TYPE=Release
                -DTensile_CODE_OBJECT_VERSION=default
                -DTensile_LOGIC=asm_full
                -DTensile_TEST_LOCAL_PATH="$AOMP_REPOS/rocmlibs/Tensile"
                -DTensile_SEPARATE_ARCHITECTURES=ON
                -DTensile_LAZY_LIBRARY_LOADING=ON
                -DTensile_LIBRARY_FORMAT=msgpack
                -DBUILD_WITH_HIPBLASLT=OFF
                -DROCTX_PATH="$AOMP_INSTALL_DIR"
                -DGPU_TARGETS="$ROCMLIBS_GFXLIST")

   echo "Beginning cmake for rocblas..."
   mkdir -p "$BuildDir"
   pushd "$BuildDir" >& /dev/null || exit
   echo "${AOMP_CMAKE}" "$(shquot "${MYCMAKEOPTS[@]}")" "$SrcDir"

   if ! ${AOMP_CMAKE} "${MYCMAKEOPTS[@]}" "$SrcDir"; then
      echo "ERROR cmake failed. Cmake flags"
      echo "      $(shquot "${MYCMAKEOPTS[@]}")"
      exit 1
   fi
   popd >& /dev/null || exit
}

task_build() {
   local Cfg=$1
   local BuildDir
   local Jobs
   BuildDir="$(get_build_dir "$Cfg")"
   Jobs="$(cfgvar AOMP_JOB_THREADS)"

   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running make for rocblas ---- "
   if ! make -j"$Jobs"; then
      echo "ERROR make -j $Jobs failed"
      exit 1
   fi
   popd >& /dev/null || exit
}

task_install() {
   local Cfg=$1
   local BuildDir
   local InstallDir
   local Jobs
   BuildDir="$(get_build_dir "$Cfg")"
   InstallDir="$(get_install_dir "$Cfg")"
   Jobs="$(cfgvar AOMP_JOB_THREADS)"

   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Installing to $InstallDir ---- "
   if ! make -j"$Jobs" install; then
      echo "ERROR install to $InstallDir failed "
      exit 1
   fi
   popd >& /dev/null || exit
   echo
   echo "SUCCESSFUL INSTALL to $InstallDir"
   echo
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
