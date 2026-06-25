#!/bin/bash
#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
#
#  build_hipBLAS-common.sh:  Script to build and install hipBLAS-common library
#                     This build is classic  cmake, make, make install
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

# All user-controllable (environment) values are read through these wrappers so
# that they can later be driven by an orchestration layer.
cfgvar() {
  get_config_var_string hipblas_common "$1"
}

cfgbool() {
  get_config_var_bool hipblas_common "$1"
}

_repo_dir="$(cfgvar AOMP_REPOS)/rocmlibs/hipBLAS-common"

get_src_dir() {
   echo "$_repo_dir"
}

# Print the build dir for a given config, passed as $1.
get_build_dir() {
   local Cfg=$1
   case "$Cfg" in
   "default")
     echo -n "$(cfgvar BUILD_DIR)/rocmlibs/hipBLAS-common"
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

# Set the environment shared by the cmake and build steps.
setup_env() {
   local AompSupp
   local Jobs
   AompSupp="$(cfgvar AOMP_SUPP)"
   Jobs="$(cfgvar AOMP_JOB_THREADS)"
   export CXX=$AOMP_INSTALL_DIR/bin/hipcc
   export ROCM_DIR=$AOMP
   export ROCM_PATH=$AOMP
   export HIP_DIR=$AOMP
   export PATH="$AompSupp/cmake/bin:$AOMP/bin:$PATH"
   export USE_PERL_SCRIPTS=1
   export NUM_PROC="$Jobs"
   export CXXFLAGS="-I$LLVM_INSTALL_LOC/include -D__HIP_PLATFORM_AMD__=1"
   export LDFLAGS="-fPIC"
   export CMAKE_PREFIX_PATH="$LLVM_INSTALL_LOC/lib/cmake"
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
   patchrepo "$_repo_dir"
}

task_unpatch() {
   removepatch "$_repo_dir"
}

task_clean() {
   local Cfg=$1
   local BuildDir
   BuildDir=$(get_build_dir "$Cfg")
   echo "rm -rf $(shquot "$BuildDir")"
   rm -rf "$BuildDir"
   mkdir -p "$BuildDir"
}

task_cmake() {
   local Cfg=$1
   local BuildDir
   local SrcDir
   local AompCmake
   local Gfxlist
   local -a AOMP_SET_NINJA_GEN
   local -a MYCMAKEOPTS

   BuildDir="$(get_build_dir "$Cfg")"
   SrcDir="$(get_src_dir)"
   AompCmake="$(cfgvar AOMP_CMAKE)"
   Gfxlist="$(cfgvar ROCMLIBS_GFXLIST)"
   setup_env

   if "$(cfgbool AOMP_USE_NINJA)"; then
      AOMP_SET_NINJA_GEN=(-G Ninja)
   fi

   MYCMAKEOPTS=("${AOMP_SET_NINJA_GEN[@]}"
                -DCMAKE_BUILD_TYPE="$(cfgvar BUILD_TYPE)"
                -DCMAKE_CXX_COMPILER="$LLVM_INSTALL_LOC/bin/clang++"
                -DHIP_COMPILER="$LLVM_INSTALL_LOC/bin/clang"
                -DHIP_CXX_COMPILER="$AOMP_INSTALL_DIR/bin/hipcc"
                -DCMAKE_PREFIX_PATH="$LLVM_INSTALL_LOC/lib/cmake"
                -DCMAKE_INSTALL_PREFIX="$AOMP_INSTALL_DIR"
                -DAMDGPU_TARGETS="$Gfxlist"
                -DROCM_DIR="$AOMP"
                -DROCM_PATH="$AOMP"
                -DHIP_DIR="$AOMP"
                -DHIP_PLATFORM=amd)

   mkdir -p "$BuildDir"
   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running cmake for hipBLAS-common $Cfg ---- "
   echo "$AompCmake $(shquot "${MYCMAKEOPTS[@]}") $SrcDir"

   if ! "$AompCmake" "${MYCMAKEOPTS[@]}" "$SrcDir"; then
      echo "ERROR cmake failed. Cmake flags"
      echo "      $(shquot "${MYCMAKEOPTS[@]}")"
      exit 1
   fi
   popd >& /dev/null || exit
}

task_build() {
   local Cfg=$1
   # hipBLAS-common is header-only; there is no separate build step. The
   # package is generated and installed during the install task.
   echo " -----No separate build step for hipBLAS-common $Cfg ---- "
}

task_install() {
   local Cfg=$1
   local BuildDir
   local InstallDir
   BuildDir="$(get_build_dir "$Cfg")"
   InstallDir="$(get_install_dir "$Cfg")"
   setup_env

   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Installing to $InstallDir ---- "
   if ! $SUDO make package install; then
      echo "ERROR make package install failed "
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
