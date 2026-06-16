#!/bin/bash
#
#  File: build_emissary.sh with symlink to build_emissary_mpi.sh
#
# MIT License
#
# Copyright (c) 2017 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
  get_config_var_string emissary "$1"
}

cfgbool() {
  get_config_var_bool emissary "$1"
}

# The emissary flavour (mpi/hdf5) is selected by the name of the symlink used
# to invoke this script.
_sname=${0##*/}
declare -a extra_cmake_opts=()
if [ "$_sname" == "build_emissary_mpi.sh" ] ; then
  EMISSARY_SRC_SUBDIR=MPI
  EMISSARY_BUILD_SUBDIR=emissary_mpi
  extra_cmake_opts+=("-DLLVM_EXTERNAL_EMISSARY_MPI_INSTALL=$HOME/local/rocmopenmpi")
elif [ "$_sname" == "build_emissary_hdf5.sh" ] ; then
  EMISSARY_SRC_SUBDIR=HDF5
  EMISSARY_BUILD_SUBDIR=emissary_hdf5
else
  echo "ERROR: You must run build_emissary_mpi.sh"
  echo "        or build_emissary_hdf5.sh"
  exit 1
fi

# Install EMISSARY in the compiler directory of ROCm
INSTALL_EMISSARY=${INSTALL_EMISSARY:-$AOMP_INSTALL_DIR}/lib/llvm
REPO_DIR="$(cfgvar AOMP_REPOS)/emissary"
export OPENMPI_DIR=$HOME/local/rocmopenmpi

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_emissary_mpi.sh                   cmake, make, NO Install "
  echo "  ./build_emissary_mpi.sh nocmake           NO cmake, make,  NO install "
  echo "  ./build_emissary_mpi.sh install           NO Cmake, make install "
  echo " "
  exit 0
fi

get_src_dir() {
   echo "$REPO_DIR/$EMISSARY_SRC_SUBDIR"
}

# Print the build dir for a given config, passed as $1.
get_build_dir() {
   local Cfg=$1
   local BuildDir
   BuildDir="$(cfgvar BUILD_DIR)"

   case "$Cfg" in
   "default")
     echo -n "$BuildDir/$EMISSARY_BUILD_SUBDIR"
     ;;
   *)
     >&2 echo "Unknown config '$Cfg'"
     exit 1
     ;;
   esac
}

# Print the install dir for a given config, passed as $1.
get_install_dir() {
   cfgvar INSTALL_EMISSARY
}

task_fetch() {
   echo "INFO: Getting latest sources for emissary in dir \"$REPO_DIR\""
   if [ -d "$REPO_DIR" ] ; then
      echo cd "$REPO_DIR"
      cd "$REPO_DIR" || exit
      echo git pull
      git pull
   else
      echo cd "$(cfgvar AOMP_REPOS)"
      cd "$(cfgvar AOMP_REPOS)" || exit
      echo git clone git@github.com:rocm/emissary
      git clone git@github.com:rocm/emissary
   fi
}

task_precheck() {
   local SrcDir
   SrcDir="$(get_src_dir)"

   if [ ! -d "$SrcDir" ] ; then
      echo "ERROR:  Missing repository $SrcDir/"
      exit 1
   fi

   if [ ! -f "$LLVM_INSTALL_LOC/bin/clang" ] ; then
      echo "ERROR:  Missing file $LLVM_INSTALL_LOC/bin/clang"
      echo "        Build the AOMP llvm compiler in $AOMP first"
      echo "        This is needed to build the emissary libraries"
      echo " "
      exit 1
   fi

   check_writable_installdir "$1" "$(cfgvar INSTALL_EMISSARY)"
}

task_patch() {
   patchrepo "$REPO_DIR"
}

task_unpatch() {
   removepatch "$REPO_DIR"
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
   local -a MYCMAKEOPTS

   SrcDir="$(get_src_dir)"
   BuildDir="$(get_build_dir "$Cfg")"
   AompCmake="$(cfgvar AOMP_CMAKE)"

   MYCMAKEOPTS=(-DCMAKE_BUILD_TYPE=Release
                "${extra_cmake_opts[@]}"
                "-DLLVM_MAIN_SRC_DIR=$(cfgvar AOMP_REPOS)/llvm-project"
                -DCMAKE_INSTALL_PREFIX="$(cfgvar INSTALL_EMISSARY)")

   mkdir -p "$BuildDir"
   pushd "$BuildDir" >& /dev/null || exit
   echo " ---- Running $AompCmake for emissary $Cfg ---- "
   echo "$AompCmake -B . $(shquot "${MYCMAKEOPTS[@]}") -S $SrcDir"

   if ! "$AompCmake" -B . "${MYCMAKEOPTS[@]}" -S "$SrcDir" ; then
      echo "ERROR emissary $Cfg cmake failed. Cmake flags"
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

   pushd "$BuildDir" >& /dev/null || exit
   echo " ---- Running $AompCmake --build -j $Jobs for emissary $Cfg ---- "
   if ! "$AompCmake" --build . --target all -j "$Jobs" ; then
      echo " "
      echo "ERROR: $AompCmake -j $Jobs  FAILED"
      echo "To restart:"
      echo "  cd $BuildDir"
      echo "  $AompCmake "
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
   echo " -----Installing to $InstallDir ----- "
   echo "$SUDO make install "

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
  echo "fetch"
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
