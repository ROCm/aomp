#!/bin/bash
#
#  File: build_extras.sh
#        Modify and copy former aomp-extras (now in aomp) utilities to aomp install.
#        The install option will install components into the aomp installation.
#        Note: this script does not use cmake or make steps.
#
# MIT License
#
# Copyright (c) 2019 Advanced Micro Devices, Inc. All Rights Reserved.
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
  get_config_var_string extras "$1"
}

cfgbool() {
  get_config_var_bool extras "$1"
}

INSTALL_EXTRAS=${INSTALL_EXTRAS:-$LLVM_INSTALL_LOC}
REPO_DIR="$(cfgvar AOMP_REPOS)/$(cfgvar AOMP_REPO_NAME)"
export LLVM_DIR=$LLVM_INSTALL_LOC

if [ "$(cfgbool AOMP_STANDALONE_BUILD)" == "true" ] ; then
  install_list="gpurun rebundle_hip_lib.sh raja_build.sh kokkos_build.sh aompversion blt.patch raja.patch modulefile"
else
  install_list="gpurun"
fi

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_extras.sh                   copy to build location, NO Install "
  echo "  ./build_extras.sh install           install "
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
   BuildDir="$(cfgvar BUILD_DIR)"

   case "$Cfg" in
   "default")
     echo -n "$BuildDir/extras"
     ;;
   *)
     >&2 echo "Unknown config '$Cfg'"
     exit 1
     ;;
   esac
}

# Print the install dir for a given config, passed as $1.
get_install_dir() {
   cfgvar INSTALL_EXTRAS
}

task_precheck() {
   check_writable_installdir "$1" "$(cfgvar INSTALL_EXTRAS)"
}

task_clean() {
   local Cfg=$1
   local BuildDir
   BuildDir=$(get_build_dir "$Cfg")
   echo "rm -rf $(shquot "$BuildDir")"
   rm -rf "$BuildDir"
}

# extras has no cmake/make; the "build" stage copies utility scripts into the
# build directory and patches their install path placeholders with sed.
task_build() {
   local Cfg=$1
   local BuildDir
   local SrcDir
   local SED_INSTALL_DIR
   local util
   BuildDir="$(get_build_dir "$Cfg")"
   SrcDir="$(get_src_dir)"

   if [ "$(cfgbool AOMP_STANDALONE_BUILD)" == "false" ] ; then
      export AOMP_VERSION_STRING=$ROCM_VERSION
   fi

   mkdir -p "$BuildDir"
   pushd "$BuildDir" >& /dev/null || exit

   if [ "$(cfgbool AOMP_STANDALONE_BUILD)" == "false" ] ; then
      SED_INSTALL_DIR=$(echo /opt/rocm/llvm | sed -e 's,/,\\/,g')
   else
      SED_INSTALL_DIR="$(cfgvar INSTALL_EXTRAS)"
      SED_INSTALL_DIR="${SED_INSTALL_DIR//\//\\/}"
   fi
   export SED_INSTALL_DIR

   echo "----- Copy util scripts to $BuildDir -----"
   cp "$SrcDir"/utils/* "$BuildDir"
   cp "$(cfgvar AOMP_REPOS)/$(cfgvar AOMP_PROJECT_REPO_NAME)"/offload/utils/gpurun "$BuildDir"

   for util in $install_list; do
      if [ "$util" == "rebundle_hip_lib.sh" ]; then
         /bin/sed -i -e "s/X\\.Y\\-Z/${AOMP_VERSION_STRING}/g" -e "s/_LLVM_INSTALL_DIR_/${SED_INSTALL_DIR}/g" "$util"
      else
         /bin/sed -i -e "s/X\\.Y\\-Z/${AOMP_VERSION_STRING}/g" -e "s/_AOMP_INSTALL_DIR_/${SED_INSTALL_DIR}/g" "$util"
      fi
   done
   popd >& /dev/null || exit
}

task_install() {
   local Cfg=$1
   local BuildDir
   local InstallDir
   local util
   BuildDir="$(get_build_dir "$Cfg")"
   InstallDir="$(get_install_dir "$Cfg")"

   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Installing to $InstallDir/bin ----- "
   for util in $install_list; do
      echo "-- Installing: $InstallDir/bin/$util"
      cp "$BuildDir/$util" "$InstallDir"/bin
      echo "$InstallDir/bin/$util" >> install_manifest.txt
   done
   if [ "$(cfgbool AOMP_STANDALONE_BUILD)" == "true" ] ; then
      if [ -f "$LLVM_INSTALL_LOC/bin/gpurun" ] && [ ! -h "$AOMP_INSTALL_DIR/bin/gpurun" ]; then
         echo "Creating gpurun symlink: ${AOMP_INSTALL_DIR}/bin/gpurun -> ${LLVM_INSTALL_LOC}/bin/gpurun"
         ln -s ../lib/llvm/bin/gpurun "$AOMP_INSTALL_DIR"/bin/gpurun
      fi
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

# List of tasks per config.  extras has no cmake step.
do_list_tasks() {
  local Cfg=$1
  if valid_config "$Cfg"; then
    echo "clean"
    echo "build"
    echo "install"
  else
    echo "Unknown config '$Cfg'"
  fi
}

command_dispatcher "$@"
