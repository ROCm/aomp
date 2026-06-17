#!/bin/bash
#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
#
#  build_hipRAND.sh: script to build and install hipRAND library
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
  get_config_var_string hiprand "$1"
}

cfgbool() {
  get_config_var_bool hiprand "$1"
}

_repo_dir="$(cfgvar AOMP_REPOS)/rocmlibs/hipRAND"

get_src_dir() {
   echo "$_repo_dir"
}

# Print the build dir for a given config, passed as $1.
get_build_dir() {
   local Cfg=$1
   case "$Cfg" in
   "default")
     echo -n "$(cfgvar BUILD_DIR)/rocmlibs/hipRAND"
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
   AompSupp="$(cfgvar AOMP_SUPP)"
   export CXX=$AOMP_INSTALL_DIR/bin/hipcc
   export ROCM_DIR=$AOMP_INSTALL_DIR
   export ROCM_PATH=$AOMP_INSTALL_DIR
   export PATH="$AompSupp/cmake/bin:$AOMP_INSTALL_DIR/bin:$PATH"
   export LDFLAGS="-fPIC"
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
   local -a MYCMAKEOPTS

   BuildDir="$(get_build_dir "$Cfg")"
   SrcDir="$(get_src_dir)"
   AompCmake="$(cfgvar AOMP_CMAKE)"
   Gfxlist="$(cfgvar ROCMLIBS_GFXLIST)"
   setup_env

   MYCMAKEOPTS=(-DCMAKE_CXX_COMPILER="$CXX"
                -DCMAKE_CXX_FLAGS="-I$LLVM_INSTALL_LOC/include -D__HIP_PLATFORM_AMD__=1"
                -DROCM_DIR="$AOMP_INSTALL_DIR"
                -DBUILD_FORTRAN_WRAPPER=ON
                -DROCM_PATH="$AOMP_INSTALL_DIR"
                -DHIP_ROOT_DIR="$AOMP_INSTALL_DIR"
                -DCPACK_PACKAGING_INSTALL_PREFIX="$AOMP_INSTALL_DIR"
                -DCMAKE_INSTALL_PREFIX="$AOMP_INSTALL_DIR"
                -DCMAKE_PREFIX_PATH="$AOMP_INSTALL_DIR"
                -DCPACK_SET_DESTDIR=OFF
                -DCMAKE_BUILD_TYPE=Release
                -DAMDGPU_TARGETS="$Gfxlist")

   mkdir -p "$BuildDir"
   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running cmake for hipRAND $Cfg ---- "
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
   local AompCmake
   local Jobs
   BuildDir="$(get_build_dir "$Cfg")"
   AompCmake="$(cfgvar AOMP_CMAKE)"
   Jobs="$(cfgvar AOMP_JOB_THREADS)"
   setup_env

   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running $AompCmake --build for hipRAND $Cfg ---- "
   if ! "$AompCmake" --build . -j "$Jobs"; then
      echo "ERROR $AompCmake --build . -j $Jobs failed"
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
