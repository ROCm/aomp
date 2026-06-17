#!/bin/bash
#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
#
#  build_rccl.sh: Script to build and install rccl.
#                 It has a dependency on rocm-core.
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
  get_config_var_string rccl "$1"
}

cfgbool() {
  get_config_var_bool rccl "$1"
}

_repo_dir="$(cfgvar AOMP_REPOS)/rocmlibs/rccl"
_rocm_core_info="$HOME/local/rocm-core/.info"

get_src_dir() {
   echo "$_repo_dir"
}

# Print the build dir for a given config, passed as $1.
get_build_dir() {
   local Cfg=$1
   case "$Cfg" in
   "default")
     echo -n "$(cfgvar BUILD_DIR)/rocmlibs/rccl"
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
   local CcacheBin
   AompSupp="$(cfgvar AOMP_SUPP)"
   Jobs="$(cfgvar AOMP_JOB_THREADS)"
   export CC=$LLVM_INSTALL_LOC/bin/clang
   export CXX=$LLVM_INSTALL_LOC/bin/clang++
   export FC=$LLVM_INSTALL_LOC/bin/flang
   export ROCM_DIR=$AOMP_INSTALL_DIR
   export ROCM_PATH=$AOMP_INSTALL_DIR
   # rccl needs cmake 3.25, so put prereq cmake first in path
   export PATH="$AompSupp/cmake/bin:$AOMP_INSTALL_DIR/bin:$PATH"
   export NUM_PROC="$Jobs"
   export CXXFLAGS="-I$HOME/local/rocm-core/include"
   export LDFLAGS="-fPIC"
   EXPLICIT_ROCM_VERSION=$(grep -E "[0-9]+\.[0-9]+\.[0-9]+" < "$_rocm_core_info/version")
   export EXPLICIT_ROCM_VERSION
   if "$(cfgbool AOMP_USE_CCACHE)"; then
      CcacheBin=$(which ccache)
      export CMAKE_CXX_COMPILER_LAUNCHER=$CcacheBin
   fi
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

   if [ -d "$_rocm_core_info" ]; then
      echo "Copying rocm-core .info from $_rocm_core_info to $AOMP_INSTALL_DIR"
      cp -r "$_rocm_core_info" "$AOMP_INSTALL_DIR"
   else
      echo "Error: rccl needs $AOMP_INSTALL_DIR/.info to exist. Please run ./build_prereq.sh first."
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
   local -a MYCMAKEOPTS

   BuildDir="$(get_build_dir "$Cfg")"
   SrcDir="$(get_src_dir)"
   AompCmake="$(cfgvar AOMP_CMAKE)"
   Gfxlist="$(cfgvar ROCMLIBS_GFXLIST)"
   setup_env

   MYCMAKEOPTS=(--toolchain=toolchain-linux.cmake
                -DCMAKE_BUILD_TYPE=Release
                -DGPU_TARGETS="$Gfxlist"
                -DCMAKE_INSTALL_PREFIX="$AOMP_INSTALL_DIR"
                -DROCM_PATH="$AOMP_INSTALL_DIR"
                -DCOLLTRACE=OFF
                -DNPKIT_FLAGS=""
                -DONLY_FUNCS=""
                -DEXPLICIT_ROCM_VERSION="$EXPLICIT_ROCM_VERSION")

   mkdir -p "$BuildDir"
   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running cmake for rccl $Cfg ---- "
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
   local BuildDir
   local Jobs
   BuildDir="$(get_build_dir "$Cfg")"
   Jobs="$(cfgvar AOMP_JOB_THREADS)"
   setup_env

   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running make for rccl $Cfg ---- "
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
