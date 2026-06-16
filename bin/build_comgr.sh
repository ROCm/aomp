#!/bin/bash
#
#  build_comgr.sh:  Script to build the code object manager for aomp
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
  get_config_var_string comgr "$1"
}

cfgbool() {
  get_config_var_bool comgr "$1"
}

INSTALL_COMGR=${INSTALL_COMGR:-$AOMP_INSTALL_DIR}
REPO_DIR="$(cfgvar AOMP_REPOS)/$(cfgvar AOMP_PROJECT_REPO_NAME)/amd/$(cfgvar AOMP_COMGR_REPO_NAME)"

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo " This script builds the code object manager"
  echo " It gets the source from:  $REPO_DIR"
  echo " It builds libraries in:   $(cfgvar BUILD_DIR)/comgr"
  echo " It installs in:           $INSTALL_COMGR"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_comgr.sh                   cmake, make , NO Install "
  echo "  ./build_comgr.sh nocmake           NO cmake, make, NO install "
  echo "  ./build_comgr.sh install           NO Cmake, make , INSTALL"
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
   BuildDir="$(cfgvar BUILD_DIR)"

   case "$Cfg" in
   "default")
     echo -n "$BuildDir/comgr"
     ;;
   "asan")
     echo -n "$BuildDir/comgr/asan"
     ;;
   *)
     >&2 echo "Unknown config '$Cfg'"
     exit 1
     ;;
   esac
}

# Print the install dir for a given config, passed as $1.
get_install_dir() {
   cfgvar INSTALL_COMGR
}

asan_config() {
   local Cfg=$1
   case "$Cfg" in
     asan|*+asan)
       return 0
       ;;
     *)
       ;;
   esac
   return 1
}

task_precheck() {
   local SrcDir
   SrcDir="$(get_src_dir)"

   if [ ! -d "$SrcDir" ] ; then
      echo "ERROR:  Missing repository $SrcDir"
      echo "        Are environment variables AOMP_REPOS and AOMP_COMGR_REPO_NAME set correctly?"
      exit 1
   fi

   check_writable_installdir "$1" "$(cfgvar INSTALL_COMGR)"
}

task_patch() {
   patchrepo "$REPO_DIR"
}

task_unpatch() {
   local osversion
   osversion=$(cat /etc/os-release)
   if [ "$AOMP_MAJOR_VERSION" != "12" ] && [[ "$osversion" =~ "Ubuntu 16" ]]; then
      removepatch "$REPO_DIR"
   fi
}

task_clean() {
   local Cfg=$1
   local BuildDir
   BuildDir=$(get_build_dir "$Cfg")
   echo "$SUDO rm -rf $(shquot "$BuildDir")"
   $SUDO rm -rf "$BuildDir"
}

task_cmake() {
   local Cfg=$1
   local BuildDir
   local SrcDir
   local Aomp
   local AompCmake
   local Repos
   local DEVICELIBS_BUILD_PATH
   local PACKAGE_ROOT
   local COMMON_PREFIX_PATH
   local -a MYCMAKEOPTS

   SrcDir="$(get_src_dir)"
   BuildDir="$(get_build_dir "$Cfg")"
   Aomp="$(cfgvar AOMP)"
   AompCmake="$(cfgvar AOMP_CMAKE)"
   Repos="$(cfgvar AOMP_REPOS)"

   export LLVM_DIR=$AOMP_INSTALL_DIR
   export Clang_DIR=$AOMP_INSTALL_DIR

   DEVICELIBS_BUILD_PATH=$Repos/build/AOMP_LIBDEVICE_REPO_NAME
   PACKAGE_ROOT=$SrcDir
   COMMON_PREFIX_PATH="$Aomp/include/amd_comgr;$DEVICELIBS_BUILD_PATH;$PACKAGE_ROOT;$LLVM_INSTALL_LOC"

   # Settings common to every config.
   MYCMAKEOPTS=(-DCMAKE_INSTALL_PREFIX="$(cfgvar INSTALL_COMGR)"
                -DCMAKE_BUILD_TYPE=Release
                -DBUILD_TESTING=OFF
                -DROCM_DIR="$AOMP_INSTALL_DIR"
                -DLLVM_DIR="$AOMP_INSTALL_DIR"
                -DClang_DIR="$AOMP_INSTALL_DIR")

   # Variant-specific settings.
   if asan_config "$Cfg"; then
      MYCMAKEOPTS+=(-DCMAKE_C_COMPILER="$LLVM_INSTALL_LOC/bin/clang"
                    -DCMAKE_CXX_COMPILER="$LLVM_INSTALL_LOC/bin/clang++"
                    -DCMAKE_PREFIX_PATH="$Aomp/lib/asan/cmake;$COMMON_PREFIX_PATH;$Aomp/lib/cmake"
                    -DCMAKE_INSTALL_LIBDIR=lib/asan
                    "${AOMP_ASAN_ORIGIN_RPATH[@]}"
                    -DCMAKE_C_FLAGS="$(cmquot "${ASAN_FLAGS[@]}")"
                    -DCMAKE_CXX_FLAGS="$(cmquot "${ASAN_FLAGS[@]}")")
   else
      MYCMAKEOPTS+=(-DCMAKE_PREFIX_PATH="$Aomp/lib/cmake;$COMMON_PREFIX_PATH"
                    -DCMAKE_INSTALL_LIBDIR=lib
                    "${AOMP_ORIGIN_RPATH[@]}")
   fi

   mkdir -p "$BuildDir"
   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running comgr $Cfg cmake ---- "
   echo "$AompCmake" "$(shquot "${MYCMAKEOPTS[@]}")" "$SrcDir"

   if ! "$AompCmake" "${MYCMAKEOPTS[@]}" "$SrcDir"; then
      echo "ERROR comgr $Cfg cmake failed. cmake flags"
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
   echo " -----Running make for comgr $Cfg ---- "
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
   if asan_config "$Cfg"; then
      echo " -----Installing to $InstallDir/lib/asan ----"
   else
      echo " -----Installing to $InstallDir/lib ----- "
   fi

   if ! $SUDO make install; then
      echo "ERROR make install failed "
      exit 1
   fi
   popd >& /dev/null || exit

   if ! asan_config "$Cfg"; then
      # amd_comgr.h is now in amd_comgr/amd_comgr.h, so remove deprecated file
      if [ -f "$InstallDir/include/amd_comgr.h" ]; then
         rm "$InstallDir/include/amd_comgr.h"
      fi
   fi
}

do_list_configs() {
  echo "default"
  if "$(cfgbool AOMP_BUILD_SANITIZER)"; then
    echo "asan"
  fi
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
