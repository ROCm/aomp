#!/bin/bash
#
#  File: build_libdevice.sh
#        build the rocm-device-libs libraries in $AOMP/lib/libdevice
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
  get_config_var_string libdevice "$1"
}

cfgbool() {
  get_config_var_bool libdevice "$1"
}

# We now pickup HSA from the AOMP install directory because it is built
# with build_roct.sh and build_rocr.sh .
HSA_DIR=${HSA_DIR:-$AOMP}
SKIPTEST=${SKIPTEST:-"YES"}
INSTALL_LIBDEVICE=${INSTALL_LIBDEVICE:-$AOMP_INSTALL_DIR}

REPO_DIR="$(cfgvar AOMP_REPOS)/$(cfgvar AOMP_PROJECT_REPO_NAME)/amd/$(cfgvar AOMP_LIBDEVICE_REPO_NAME)"

export LLVM_DIR=$AOMP_INSTALL_DIR
export LLVM_BUILD=$AOMP_INSTALL_DIR
export HSA_DIR
export PATH="$LLVM_BUILD/bin":$PATH

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_libdevice.sh                   cmake, make, NO Install "
  echo "  ./build_libdevice.sh nocmake           NO cmake, make,  NO install "
  echo "  ./build_libdevice.sh install           NO Cmake, make install "
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
     echo -n "$BuildDir/libdevice"
     ;;
   *)
     >&2 echo "Unknown config '$Cfg'"
     exit 1
     ;;
   esac
}

# Print the install dir for a given config, passed as $1.
get_install_dir() {
   cfgvar INSTALL_LIBDEVICE
}

task_precheck() {
   if [ ! -d "$AOMP_INSTALL_DIR/lib" ]; then
      echo "ERROR: Directory $AOMP/lib is missing"
      echo "       AOMP must be installed in $AOMP_INSTALL_DIR to continue"
      exit 1
   fi
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
   # need SUDO because a previous make install was done with sudo
   $SUDO rm -rf "$BuildDir"
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
   InstallDir="$(cfgvar INSTALL_LIBDEVICE)"

   MYCMAKEOPTS=(-DLLVM_DIR="$LLVM_DIR"
                -DCMAKE_INSTALL_LIBDIR=lib
                -DCMAKE_INSTALL_PREFIX="$InstallDir")

   mkdir -p "$BuildDir"
   pushd "$BuildDir" >& /dev/null || exit
   echo "DOING BUILD in Directory $BuildDir"
   CC="$LLVM_BUILD/bin/clang"
   export CC
   echo "$AompCmake $(shquot "${MYCMAKEOPTS[@]}") $SrcDir"

   if ! "$AompCmake" "${MYCMAKEOPTS[@]}" "$SrcDir"; then
      echo "ERROR cmake failed  command was:"
      echo "      $AompCmake $(shquot "${MYCMAKEOPTS[@]}") $SrcDir"
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
   echo "make -j $Jobs"
   if ! make -j "$Jobs"; then
      echo "ERROR make failed "
      exit 1
   fi
   echo
   echo "  Done with all makes"

   if [ "$(cfgvar SKIPTEST)" != "YES" ] ; then
      echo "running tests in $BuildDir"
      make test
      echo
      echo "# done with all tests"
      echo
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

   echo
   echo mkdir -p "$InstallDir/include"
   $SUDO mkdir -p "$InstallDir/include"
   $SUDO mkdir -p "$InstallDir/lib"
   pushd "$BuildDir" >& /dev/null || exit
   echo "running make install from $BuildDir"
   echo "$SUDO make -j $Jobs install"
   $SUDO make -j "$Jobs" install
   echo
   echo " Installation complete into $InstallDir"
   echo
   popd >& /dev/null || exit
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
