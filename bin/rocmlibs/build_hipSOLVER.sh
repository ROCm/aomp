#!/bin/bash
#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
#
#  build_hipSOLVER.sh:  Script to build and install hipSOLVER library
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
  get_config_var_string hipsolver "$1"
}

cfgbool() {
  get_config_var_bool hipsolver "$1"
}

_repo_dir="$(cfgvar AOMP_REPOS)/rocmlibs/hipSOLVER"

get_src_dir() {
   echo "$_repo_dir"
}

# Print the build dir for a given config, passed as $1.
#
# hipSOLVER does not follow the AOMP build directory convention because its
# install.sh assumes the build directory is a subdirectory of the source
# directory.
get_build_dir() {
   local Cfg=$1
   case "$Cfg" in
   "default")
     echo -n "$_repo_dir/build"
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
   local SrcDir
   local -a INSTALL_OPTS

   SrcDir="$(get_src_dir)"
   export ROCM_PATH=$AOMP_INSTALL_DIR

   # hipSOLVER's install.sh both configures and builds the library.
   INSTALL_OPTS=(--compiler "$AOMP_INSTALL_DIR/lib/llvm/bin/clang++"
                 --rocblas-path "$AOMP_INSTALL_DIR"
                 --hipblas-path "$AOMP_INSTALL_DIR"
                 --rocsolver-path "$AOMP_INSTALL_DIR"
                 --cmakepp "$AOMP_INSTALL_DIR"
                 --no-sparse
                 --no-hip-clang
                 --relocatable
                 --cmake-arg -DCMAKE_INSTALL_PREFIX="$AOMP_INSTALL_DIR")

   pushd "$SrcDir" >& /dev/null || exit
   echo " ----- Running hipSOLVER install.sh for $Cfg -----"
   echo "./install.sh $(shquot "${INSTALL_OPTS[@]}")"
   if ! ./install.sh "${INSTALL_OPTS[@]}"; then
      echo "ERROR ./install.sh failed."
      exit 1
   fi
   popd >& /dev/null || exit
}

task_build() {
   local Cfg=$1
   # hipSOLVER's install.sh (run in the cmake task) both configures and builds
   # the library, so there is no separate build step.
   echo " -----No separate build step for hipSOLVER $Cfg ---- "
}

task_install() {
   local Cfg=$1
   local BuildDir
   local InstallDir
   BuildDir="$(get_build_dir "$Cfg")"
   InstallDir="$(get_install_dir "$Cfg")"

   pushd "$BuildDir/release" >& /dev/null || exit
   echo " -----Installing to $InstallDir ---- "
   if ! make install; then
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
