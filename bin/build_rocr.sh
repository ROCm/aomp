#!/bin/bash
#
#  build_rocr.sh:  Script to build the rocm runtime and install into the 
#                  aomp compiler installation
#                  Requires that "build_roct.sh install" be installed first
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
  get_config_var_string rocr "$1"
}

cfgbool() {
  get_config_var_bool rocr "$1"
}

INSTALL_ROCM=${INSTALL_ROCM:-$AOMP_INSTALL_DIR}
REPO_DIR="$(cfgvar AOMP_REPOS)/$(cfgvar AOMP_ROCR_REPO_NAME)"
_ompd_src_dir="$LLVM_INSTALL_LOC/share/gdb/python/ompd/src"

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  echo " "
  echo " This script builds the ROCM runtime libraries"
  echo " It gets the source from:  $REPO_DIR"
  echo " It builds libraries in:   $(cfgvar BUILD_DIR)/rocr"
  echo " It installs in:           $INSTALL_ROCM"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_rocr.sh                   cmake, make , NO Install "
  echo "  ./build_rocr.sh nocmake           NO cmake, make, NO install "
  echo "  ./build_rocr.sh install           NO Cmake, make , INSTALL"
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
     echo -n "$BuildDir/rocr"
     ;;
   "asan")
     echo -n "$BuildDir/rocr/asan"
     ;;
   "debug")
     echo -n "$BuildDir/rocr_debug"
     ;;
   *)
     >&2 echo "Unknown config '$Cfg'"
     exit 1
     ;;
   esac
}

# Print the install dir for a given config, passed as $1.
get_install_dir() {
   cfgvar INSTALL_ROCM
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

debug_config() {
   local Cfg=$1
   case "$Cfg" in
     debug|debug+*)
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
      echo "        Are environment variables AOMP_REPOS and AOMP_ROCR_REPO_NAME set correctly?"
      exit 1
   fi

   check_writable_installdir "$1" "$(cfgvar INSTALL_ROCM)"
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
   local -a _prefix_map

   SrcDir="$(get_src_dir)"
   BuildDir="$(get_build_dir "$Cfg")"
   AompCmake="$(cfgvar AOMP_CMAKE)"
   export PATH=/opt/rocm/llvm/bin:$PATH

   # Settings common to every config.
   MYCMAKEOPTS=(-DCMAKE_C_COMPILER="$AOMP_INSTALL_DIR/lib/llvm/bin/clang"
                -DCMAKE_CXX_COMPILER="$AOMP_INSTALL_DIR/lib/llvm/bin/clang++"
                -DLLVM_DIR="$AOMP_INSTALL_DIR/lib/llvm/bin"
                -DCMAKE_PREFIX_PATH="$AOMP_INSTALL_DIR/lib"
                -DIMAGE_SUPPORT=OFF
                -DBUILD_SHARED_LIBS=On)

   # Variant-specific settings.
   if asan_config "$Cfg"; then
      MYCMAKEOPTS+=(-DCMAKE_INSTALL_PREFIX="$AOMP_INSTALL_DIR"
                    -DCMAKE_INSTALL_LIBDIR=lib/asan
                    -DCMAKE_BUILD_TYPE="$(cfgvar BUILD_TYPE)"
                    "${AOMP_ASAN_ORIGIN_RPATH[@]}"
                    -DCMAKE_C_FLAGS="$(cmquot "${ASAN_FLAGS[@]}")"
                    -DCMAKE_CXX_FLAGS="$(cmquot "${ASAN_FLAGS[@]}")")
   elif debug_config "$Cfg"; then
      _prefix_map=(-fdebug-prefix-map="$SrcDir=$_ompd_src_dir/rocr")
      MYCMAKEOPTS+=(-DCMAKE_INSTALL_PREFIX="$AOMP_INSTALL_DIR"
                    -DCMAKE_BUILD_TYPE=Debug
                    "${AOMP_DEBUG_ORIGIN_RPATH[@]}"
                    -DCMAKE_INSTALL_LIBDIR=lib-debug
                    -DTARGET_DEVICES="gfx900;gfx90a;gfx942;gfx1010;gfx1030;gfx1100;gfx1200"
                    -DCMAKE_C_FLAGS="$(cmquot -g "${_prefix_map[@]}")"
                    -DCMAKE_CXX_FLAGS="$(cmquot -g "${_prefix_map[@]}")")
   else
      MYCMAKEOPTS+=(-DCMAKE_INSTALL_PREFIX="$(cfgvar INSTALL_ROCM)"
                    -DCMAKE_BUILD_TYPE=Release
                    -DCMAKE_INSTALL_LIBDIR=lib
                    "${AOMP_ORIGIN_RPATH[@]}")
   fi

   mkdir -p "$BuildDir"
   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running rocr $Cfg cmake ---- "
   echo "$AompCmake" "$(shquot "${MYCMAKEOPTS[@]}")" "$SrcDir"

   if ! "$AompCmake" "${MYCMAKEOPTS[@]}" "$SrcDir"; then
      echo "ERROR rocr $Cfg cmake failed. cmake flags"
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
   echo " -----Running make for rocr $Cfg ---- "
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
   if asan_config "$Cfg"; then
      echo " ------Installing to $InstallDir/lib/asan ------ "
   elif debug_config "$Cfg"; then
      echo " -----Installing to $InstallDir/lib-debug ----- "
   else
      echo " -----Installing to $InstallDir/lib ----- "
   fi
   echo "$SUDO make install "

   if ! $SUDO make install; then
      echo "ERROR make install failed "
      exit 1
   fi
   popd >& /dev/null || exit
}

task_postinstall() {
   local Cfg=$1
   local SrcDir
   local _dirs
   local _dirname
   SrcDir="$(get_src_dir)"

   # copy rocr sources into the installation for runtime source debugging
   _dirs="runtime/hsa-runtime/image runtime/hsa-runtime/inc runtime/hsa-runtime/core runtime/hsa-runtime/loader runtime/hsa-runtime/pcs libhsakmt/src libhsakmt/include"
   for _dirname in $_dirs ; do
      $SUDO mkdir -p "$_ompd_src_dir/rocr/$_dirname"
      echo "cp -r $SrcDir/$_dirname/ $_ompd_src_dir/rocr/$_dirname/"
      $SUDO cp -r "$SrcDir/$_dirname/" "$_ompd_src_dir/rocr/$_dirname/"
   done
   # remove non-source files to save space
   find "$_ompd_src_dir/rocr" -type f | grep -v "\.cpp$\|\.h$\|\.hpp$\|\.c$\|\.s$" | xargs -r rm
}

do_list_configs() {
  echo "default"
  if "$(cfgbool AOMP_BUILD_SANITIZER)"; then
    echo "asan"
  fi
  if "$(cfgbool AOMP_BUILD_DEBUG)"; then
    echo "debug"
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
    case "$Cfg" in
      debug)
        echo "postinstall"
        ;;
      *)
        ;;
    esac
  else
    echo "Unknown config '$Cfg'"
  fi
}

command_dispatcher "$@"
