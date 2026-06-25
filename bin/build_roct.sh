#!/bin/bash
#
#  build_roct.sh:  Script to build the ROCt thunk libraries.
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
  get_config_var_string roct "$1"
}

cfgbool() {
  get_config_var_bool roct "$1"
}

INSTALL_ROCT=${INSTALL_ROCT:-$AOMP_INSTALL_DIR}
REPO_DIR="$(cfgvar AOMP_REPOS)/$(cfgvar AOMP_ROCT_REPO_NAME)"
_ompd_src_dir="$LLVM_INSTALL_LOC/share/gdb/python/ompd/src"

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  echo " "
  echo " This script builds the ROCt thunk libraries"
  echo " It gets the source from:  $REPO_DIR"
  echo " It builds libraries in:   $(cfgvar BUILD_DIR)/roct"
  echo " It installs in:           $INSTALL_ROCT"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_roct.sh                   cmake, make , NO Install "
  echo "  ./build_roct.sh nocmake           NO cmake, make, NO install "
  echo "  ./build_roct.sh install           NO Cmake, make , INSTALL"
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
     echo -n "$BuildDir/roct"
     ;;
   "asan")
     echo -n "$BuildDir/roct/asan"
     ;;
   "debug")
     echo -n "$BuildDir/roct_debug"
     ;;
   *)
     >&2 echo "Unknown config '$Cfg'"
     exit 1
     ;;
   esac
}

# Print the install dir for a given config, passed as $1.
get_install_dir() {
   cfgvar INSTALL_ROCT
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
      echo "        Are environment variables AOMP_REPOS and AOMP_ROCT_REPO_NAME set correctly?"
      exit 1
   fi

   check_writable_installdir "$1" "$(cfgvar INSTALL_ROCT)"
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
   echo "$SUDO rm -rf $(shquot "$BuildDir")"
   $SUDO rm -rf "$BuildDir"
}

task_cmake() {
   local Cfg=$1
   local BuildDir
   local SrcDir
   local AompCmake
   local ClangCC
   local ClangCXX
   local -a MYCMAKEOPTS
   local -a _prefix_map

   SrcDir="$(get_src_dir)"
   BuildDir="$(get_build_dir "$Cfg")"
   AompCmake="$(cfgvar AOMP_CMAKE)"
   ClangCC="$(cfgvar AOMP_CLANG_COMPILER)"
   ClangCXX="$(cfgvar AOMP_CLANGXX_COMPILER)"

   # Settings common to every config.
   MYCMAKEOPTS=(-DCMAKE_INSTALL_PREFIX="$(cfgvar INSTALL_ROCT)")

   # Variant-specific settings.
   if asan_config "$Cfg"; then
      MYCMAKEOPTS+=(-DCMAKE_C_COMPILER="$ClangCC"
                    -DCMAKE_CXX_COMPILER="$ClangCXX"
                    -DCMAKE_PREFIX_PATH="$AOMP_INSTALL_DIR/lib/asan/cmake"
                    -DCMAKE_BUILD_TYPE=Release
                    "${AOMP_ASAN_ORIGIN_RPATH[@]}"
                    -DCMAKE_INSTALL_LIBDIR="$AOMP_INSTALL_DIR/lib/asan"
                    -DCMAKE_C_FLAGS="$(cmquot "${ASAN_FLAGS[@]}")"
                    -DCMAKE_CXX_FLAGS="$(cmquot "${ASAN_FLAGS[@]}")")
   elif debug_config "$Cfg"; then
      _prefix_map=(-fdebug-prefix-map="$SrcDir/src=$_ompd_src_dir/roct/src")
      MYCMAKEOPTS+=(-DCMAKE_C_COMPILER="$ClangCC"
                    -DCMAKE_CXX_COMPILER="$ClangCXX"
                    -DCMAKE_BUILD_TYPE=Debug
                    "${AOMP_DEBUG_ORIGIN_RPATH[@]}"
                    -DCMAKE_INSTALL_LIBDIR=lib-debug
                    -DBUILD_SHARED_LIBS=ON
                    -DCMAKE_C_FLAGS="$(cmquot -g "${_prefix_map[@]}")"
                    -DCMAKE_CXX_FLAGS="$(cmquot -g "${_prefix_map[@]}")")
   else
      MYCMAKEOPTS+=(-DCMAKE_PREFIX_PATH="$AOMP_INSTALL_DIR/lib/cmake"
                    -DCMAKE_BUILD_TYPE=Release
                    "${AOMP_ORIGIN_RPATH[@]}"
                    -DCMAKE_INSTALL_LIBDIR=lib)
   fi

   mkdir -p "$BuildDir"
   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running roct $Cfg cmake ---- "
   echo "$AompCmake" "$(shquot "${MYCMAKEOPTS[@]}")" "$SrcDir"

   if ! "$AompCmake" "${MYCMAKEOPTS[@]}" "$SrcDir"; then
      echo "ERROR roct $Cfg cmake failed. cmake flags"
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
   echo " -----Running make for roct $Cfg ---- "
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
      echo " -----Installing to $InstallDir/lib/asan ----- "
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
   SrcDir="$(get_src_dir)"

   # copy roct sources into the installation for runtime source debugging
   $SUDO mkdir -p "$_ompd_src_dir/roct"
   echo "$SUDO cp -r $SrcDir/src $_ompd_src_dir/roct"
   $SUDO cp -r "$SrcDir/src" "$_ompd_src_dir/roct"
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
