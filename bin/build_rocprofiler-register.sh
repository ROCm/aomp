#!/bin/bash
#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
# 
#  build_rocprofiler-register.sh:  Script to build rocprofiler-register for AOMP standalone build
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
  get_config_var_string rocprofiler-register "$1"
}

cfgbool() {
  get_config_var_bool rocprofiler-register "$1"
}

INSTALL_ROCPROF_REGISTER=${INSTALL_ROCPROF_REGISTER:-$AOMP_INSTALL_DIR}
REPO_DIR="$(cfgvar AOMP_REPOS)/$(cfgvar AOMP_PROF_REGISTER_REPO_NAME)"

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo " This script builds the ROCM runtime libraries"
  echo " It gets the source from:  $REPO_DIR"
  echo " It builds libraries in:   $(cfgvar BUILD_DIR)/$(cfgvar AOMP_PROF_REGISTER_REPO_NAME)"
  echo " It installs in:           $INSTALL_ROCPROF_REGISTER"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_rocprofiler-register.sh                   cmake, make , NO Install "
  echo "  ./build_rocprofiler-register.sh nocmake           NO cmake, make, NO install "
  echo "  ./build_rocprofiler-register.sh install           NO Cmake, make , INSTALL"
  echo " "
  echo "To build aomp, see the README file in this directory"
  echo " "
  exit 0
fi

get_src_dir() {
   echo "$REPO_DIR"
}

# Print the build dir for a given config, passed as $1.
get_build_dir() {
   local Cfg=$1
   local BuildDir
   BuildDir="$(cfgvar BUILD_DIR)/$(cfgvar AOMP_PROF_REGISTER_REPO_NAME)"

   case "$Cfg" in
   "default")
     echo -n "$BuildDir"
     ;;
   *)
     >&2 echo "Unknown config '$Cfg'"
     exit 1
     ;;
   esac
}

# Print the install dir for a given config, passed as $1.
get_install_dir() {
   cfgvar INSTALL_ROCPROF_REGISTER
}

task_precheck() {
   local SrcDir
   SrcDir="$(get_src_dir)"

   if [ ! -d "$SrcDir" ] ; then
      echo "ERROR:  Missing repository $SrcDir"
      echo "        Are environment variables AOMP_REPOS and AOMP_PROF_REGISTER_REPO_NAME set correctly?"
      exit 1
   fi

   check_writable_installdir "$1" "$(cfgvar INSTALL_ROCPROF_REGISTER)"
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
   local AompCmake
   local InstallDir
   local -a MYCMAKEOPTS

   SrcDir="$(get_src_dir)"
   BuildDir="$(get_build_dir "$Cfg")"
   AompCmake="$(cfgvar AOMP_CMAKE)"
   InstallDir="$(cfgvar INSTALL_ROCPROF_REGISTER)"

   export HIP_CLANG_PATH="$InstallDir/bin"

   MYCMAKEOPTS=(-DCMAKE_INSTALL_LIBDIR=lib
                -DCMAKE_BUILD_TYPE=Release
                -DROCM_PATH="$AOMP_INSTALL_DIR"
                -DCMAKE_INSTALL_PREFIX="$InstallDir"
                -DCMAKE_PREFIX_PATH="$ROCM_DIR/include;$ROCM_DIR/lib;$ROCM_DIR"
                "${AOMP_ORIGIN_RPATH[@]}"
                -DCMAKE_EXE_LINKER_FLAGS="-Wl,--disable-new-dtags"
                -DBUILD_SHARED_LIBS=ON
                -DENABLE_LDCONFIG=OFF
                -DROCPROFILER_REGISTER_BUILD_TESTS=0
                -DROCPROFILER_REGISTER_BUILD_SAMPLES=1)

   mkdir -p "$BuildDir"
   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running cmake for rocprofiler-register $Cfg ---- "
   echo "$AompCmake $(shquot "${MYCMAKEOPTS[@]}") $SrcDir"

   if ! "$AompCmake" "${MYCMAKEOPTS[@]}" "$SrcDir"; then
      echo "ERROR rocprofiler-register $Cfg cmake failed. cmake flags"
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
   echo " -----Running make for rocprofiler-register $Cfg ---- "
   echo "make -j $Jobs"
   if ! make -j "$Jobs"; then
      echo " "
      echo "ERROR: make -j $Jobs  FAILED"
      echo "To restart:"
      echo "  cd $BuildDir"
      echo "  make"
      exit 1
   fi
   popd >& /dev/null || exit
}

task_install() {
   local Cfg=$1
   local BuildDir
   local InstallDir
   BuildDir="$(get_build_dir "$Cfg")"
   InstallDir="$(get_install_dir "$Cfg")"

   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Installing to $InstallDir/lib ----- "
   echo "$SUDO make install"

   if ! $SUDO make install; then
      echo "ERROR make install failed "
      exit 1
   fi
   popd >& /dev/null || exit
}

do_list_configs() {
  echo "default"
}

do_list_init() {
  echo "precheck"
}

do_list_fini() {
  :
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
