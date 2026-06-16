#!/bin/bash
#
#  build_rocprofiler.sh:  Script to build rocprofiler for AOMP standalone build
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
  get_config_var_string rocprofiler "$1"
}

cfgbool() {
  get_config_var_bool rocprofiler "$1"
}

INSTALL_ROCPROF=${INSTALL_ROCPROF:-$AOMP_INSTALL_DIR}
REPO_DIR="$(cfgvar AOMP_REPOS)/$(cfgvar AOMP_PROF_REPO_NAME)"

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo " This script builds the ROCM runtime libraries"
  echo " It gets the source from:  $REPO_DIR"
  echo " It builds libraries in:   $(cfgvar BUILD_DIR)/rocprofiler"
  echo " It installs in:           $INSTALL_ROCPROF"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_rocprofiler.sh                   cmake, make , NO Install "
  echo "  ./build_rocprofiler.sh nocmake           NO cmake, make, NO install "
  echo "  ./build_rocprofiler.sh install           NO Cmake, make , INSTALL"
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
     echo -n "$BuildDir/rocprofiler"
     ;;
   *)
     >&2 echo "Unknown config '$Cfg'"
     exit 1
     ;;
   esac
}

# Print the install dir for a given config, passed as $1.
get_install_dir() {
   cfgvar INSTALL_ROCPROF
}

task_precheck() {
   local SrcDir
   SrcDir="$(get_src_dir)"

   if [ ! -d "$SrcDir" ] ; then
      echo "ERROR:  Missing repository $SrcDir"
      echo "        Are environment variables AOMP_REPOS and AOMP_PROF_REPO_NAME set correctly?"
      exit 1
   fi

   check_writable_installdir "$1" "$(cfgvar INSTALL_ROCPROF)"
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
   local InstallDir
   local GfxSemicolons
   local _loc
   local _gccver
   local -a CMAKE_WITH_EXPERIMENTAL
   local -a MYCMAKEOPTS

   SrcDir="$(get_src_dir)"
   BuildDir="$(get_build_dir "$Cfg")"
   AompCmake="$(cfgvar AOMP_CMAKE)"
   InstallDir="$(cfgvar INSTALL_ROCPROF)"
   GfxSemicolons=$(cfgvar GFXLIST | tr ' ' ';')

   export HIP_CLANG_PATH="$LLVM_INSTALL_LOC/bin"
   export ROCM_PATH="$AOMP_INSTALL_DIR"
   export CMAKE_BUILD_TYPE=Release
   export CMAKE_PREFIX_PATH="$ROCM_DIR/roctracer/include/ext;$ROCM_DIR/include/platform;$ROCM_DIR/include;$ROCM_DIR/lib;$ROCM_DIR;$HOME/.local/bin;$HOME/local/aqlprofile;$LLVM_INSTALL_LOC"
   export PATH=$HOME/.local/bin:$InstallDir/bin:$PATH

   CMAKE_WITH_EXPERIMENTAL=()
   if [ -d "/usr/include/c++/5/experimental" ] ; then
      _loc=$(which gcc)
      if [ "$_loc" != "" ] ; then
         _gccver=$($_loc --version | grep gcc | cut -d")" -f2 | cut -d"." -f1)
         if [ "$_gccver" == "5" ] ; then
            CMAKE_WITH_EXPERIMENTAL=(-DCMAKE_CXX_FLAGS=-I/usr/include/c++/5/experimental)
         fi
      fi
   fi

   MYCMAKEOPTS=(-DCMAKE_INSTALL_LIBDIR=lib
                -DENABLE_ASAN_PACKAGING=ON
                -DCMAKE_BUILD_TYPE=Release
                -DROCM_PATH="$AOMP_INSTALL_DIR"
                -DCMAKE_MODULE_PATH="$InstallDir/lib/cmake/hip"
                -DCMAKE_INSTALL_PREFIX="$InstallDir"
                -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH"
                "${CMAKE_WITH_EXPERIMENTAL[@]}"
                "${AOMP_ORIGIN_RPATH[@]}"
                -DGPU_TARGETS="$GfxSemicolons"
                -DPROF_API_HEADER_PATH="$InstallDir/include/roctracer/ext"
                -DHIP_ROOT_DIR="$InstallDir/hip"
                -DAQLPROFILE_LIB="$(cfgvar AOMP_SUPP)/aqlprofile/lib/libhsa-amd-aqlprofile64.so"
                -DCMAKE_CXX_FLAGS="-I$HOME/local/rocmsmilib/include"
                -DHIP_HIPCC_FLAGS="-I$HOME/local/rocmsmilib/include"
                -DCMAKE_EXE_LINKER_FLAGS="-L$HOME/local/rocmsmilib/lib -L$HOME/local/rocmsmilib/lib64 -Wl,--disable-new-dtags")

   mkdir -p "$BuildDir"
   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running cmake for rocprofiler $Cfg ---- "
   echo "$AompCmake $(shquot "${MYCMAKEOPTS[@]}") $SrcDir"

   if ! "$AompCmake" "${MYCMAKEOPTS[@]}" "$SrcDir"; then
      echo "ERROR rocprofiler $Cfg cmake failed. cmake flags"
      echo "      $(shquot "${MYCMAKEOPTS[@]}")"
      exit 1
   fi
   popd >& /dev/null || exit
}

task_build() {
   local Cfg=$1
   local BuildDir
   local Jobs
   local doxygen
   BuildDir="$(get_build_dir "$Cfg")"
   Jobs="$(cfgvar AOMP_JOB_THREADS)"

   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running make for rocprofiler $Cfg ---- "
   echo "make -j $Jobs"
   make -j "$Jobs"

   if ! make -j "$Jobs" mytest; then
      echo " "
      echo "ERROR: make -j $Jobs  FAILED"
      echo "To restart:"
      echo "  cd $BuildDir"
      echo "  make"
      exit 1
   fi

   doxygen=$(which doxygen || true)
   if [ -n "$doxygen" ] ; then
      # the rocprofiler CMakeLists.txt will prepare docs install if doxygen
      # found.  However, the make doc has issues.  But if you dont make doc, the
      # install fails.  This 'make doc' will do enough so install does not fail.
      echo "make -j $Jobs doc"
      make -j "$Jobs" doc 2>/dev/null >/dev/null || true
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
