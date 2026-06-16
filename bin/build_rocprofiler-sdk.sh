#!/bin/bash
#
#  build_rocprofiler-sdk.sh:  Script to build rocprofiler-sdk for AOMP standalone build
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
  get_config_var_string rocprofiler-sdk "$1"
}

cfgbool() {
  get_config_var_bool rocprofiler-sdk "$1"
}

INSTALL_ROCPROF_SDK=${INSTALL_ROCPROF_SDK:-$AOMP_INSTALL_DIR}
REPO_DIR="$(cfgvar AOMP_REPOS)/$(cfgvar AOMP_PROF_SDK_REPO_NAME)"

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo " This script builds the ROCM runtime libraries"
  echo " It gets the source from:  $REPO_DIR"
  echo " It builds libraries in:   $(cfgvar BUILD_DIR)/rocprofiler-sdk"
  echo " It installs in:           $INSTALL_ROCPROF_SDK"
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_rocprofiler-sdk.sh                   cmake, make , NO Install "
  echo "  ./build_rocprofiler-sdk.sh nocmake           NO cmake, make, NO install "
  echo "  ./build_rocprofiler-sdk.sh install           NO Cmake, make , INSTALL"
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
     echo -n "$BuildDir/rocprofiler-sdk"
     ;;
   *)
     >&2 echo "Unknown config '$Cfg'"
     exit 1
     ;;
   esac
}

# Print the install dir for a given config, passed as $1.
get_install_dir() {
   cfgvar INSTALL_ROCPROF_SDK
}

task_precheck() {
   local SrcDir
   SrcDir="$(get_src_dir)"

   if [ ! -d "$SrcDir" ] ; then
      echo "ERROR:  Missing repository $SrcDir"
      echo "        Are environment variables AOMP_REPOS and AOMP_PROF_SDK_REPO_NAME set correctly?"
      exit 1
   fi

   check_writable_installdir "$1" "$(cfgvar INSTALL_ROCPROF_SDK)"
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
   local pythonbinary
   local pythonversion
   local -a MYCMAKEOPTS

   SrcDir="$(get_src_dir)"
   BuildDir="$(get_build_dir "$Cfg")"
   AompCmake="$(cfgvar AOMP_CMAKE)"
   InstallDir="$(cfgvar INSTALL_ROCPROF_SDK)"
   GfxSemicolons=$(cfgvar GFXLIST | tr ' ' ';')

   export HIP_CLANG_PATH="$LLVM_INSTALL_LOC/bin"
   export ROCM_PATH="$AOMP_INSTALL_DIR"
   export PATH=$HOME/.local/bin:$InstallDir/bin:$PATH

   pythonbinary=$(which python3) || exit
   pythonversion=$("$pythonbinary" --version) || exit
   if [[ $pythonversion =~ ([Pp]ython)[[:space:]]*([0-9]+)\.([0-9]+) ]]; then
      pythonversion="${BASH_REMATCH[2]}.${BASH_REMATCH[3]}"
   else
      echo "Error: cannot determine python version"
      exit 1
   fi

   MYCMAKEOPTS=(-DCMAKE_PREFIX_PATH="$AOMP_INSTALL_DIR;$HOME/local/aqlprofile"
                -DCMAKE_INSTALL_PREFIX="$InstallDir"
                -DCMAKE_BUILD_TYPE=Release
                -DROCM_ROOT_DIR="$AOMP_INSTALL_DIR"
                -DBUILD_SHARED_LIBS=On
                -DGPU_TARGETS="$GfxSemicolons"
                -DROCPROFILER_BUILD_SAMPLES=ON
                -DROCPROFILER_BUILD_TESTS=OFF
                -DPython3_EXECUTABLE="$pythonbinary"
                -DROCPROFILER_PYTHON_VERSIONS="$pythonversion")

   mkdir -p "$BuildDir"
   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running cmake for rocprofiler-sdk $Cfg ---- "
   echo "$AompCmake $(shquot "${MYCMAKEOPTS[@]}") $SrcDir"

   if ! "$AompCmake" "${MYCMAKEOPTS[@]}" "$SrcDir"; then
      echo "ERROR rocprofiler-sdk $Cfg cmake failed. cmake flags"
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
   echo " -----Running make for rocprofiler-sdk $Cfg ---- "
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

task_postinstall() {
   local Cfg=$1
   local InstallDir
   InstallDir="$(get_install_dir "$Cfg")"

   if [ -d "$HOME/local/aqlprofile/lib" ]; then
      echo "Copying aqlprofile libraries from $HOME/local/aqlprofile/lib to $InstallDir/lib"
      cp -r "$HOME"/local/aqlprofile/lib/* "$InstallDir"/lib
   else
      echo "Error: rocprofiler-sdk needs aqlprofile libraries to exist in $InstallDir/lib. Please run ./build_prereq.sh first."
      exit 1
   fi
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
    echo "postinstall"
  else
    echo "Unknown config '$Cfg'"
  fi
}

command_dispatcher "$@"
