#!/bin/bash
#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
# 
#  build_llvm-classic.sh:  Script to build the classic LLVM used by flang-classic. binary driver
#         This driver will never call flang -fc1, it only calls binaries
#             clang, flang1, flang2, built elsewhere
#  Instead of downloading the ROCm 5.5 llvm package we have to
#  compile the 11vm/clang libs from source to support various
#  operating systems and spack. This will be the llvm-classic build step.
#  These libs/headers are not installed and will picked up from the build
#  tree for flang-classic.
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
  get_config_var_string llvm-classic "$1"
}

cfgbool() {
  get_config_var_bool llvm-classic "$1"
}

# Do not change the AOMP_LFL_DIR default because it is the subdirectory
# from where we build the flang-classic driver binary.  This is the
# Last Frozen LLVM (LFL) for which there is amd-only clang driver support
# for flang.  Originally there was no subdirectory for LFL so setting
# AOMP_LFL_DIR to "/" would build flang-classic with the original
# ROCm 5.6 sources.
AOMP_LFL_DIR=${AOMP_LFL_DIR:-"17.0-4"}

# AOMP_BUILD_FLANG_CLASSIC is computed by aomp_common_vars (from the presence
# of the llvm-classic source), not a user-controllable variable, so it is read
# directly rather than through cfgbool.
if [ "$AOMP_BUILD_FLANG_CLASSIC" == 0 ] ; then
   if [ "$1" != "install" ] ; then
      echo "WARNING:  ROCM install for $AOMP_FLANG_CLASSIC_REL/llvm-classic not found."
      echo "          This build will skip build of flang-classic."
      echo "          The flang will link to the clang driver."
   fi
   exit 0
fi

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  help_build_aomp
fi

get_src_dir() {
   echo "$(cfgvar AOMP_REPOS)/$(cfgvar AOMP_FLANG_REPO_NAME)/flang-classic/$(cfgvar AOMP_LFL_DIR)/llvm-classic/llvm"
}

# Print the build dir for a given config, passed as $1.
get_build_dir() {
   local Cfg=$1
   local BuildDir
   BuildDir="$(cfgvar BUILD_DIR)"

   case "$Cfg" in
   "default")
     echo -n "$BuildDir/flang-classic/$(cfgvar AOMP_LFL_DIR)/llvm-classic"
     ;;
   *)
     >&2 echo "Unknown config '$Cfg'"
     exit 1
     ;;
   esac
}

# Print the install dir for a given config, passed as $1.  llvm-classic is not
# installed; its libs/headers are picked up from the build tree.
get_install_dir() {
   echo "$AOMP_INSTALL_DIR"
}

task_precheck() {
   if [ "$(cfgbool AOMP_STANDALONE_BUILD)" == "true" ] ; then
      if [ ! -L "$AOMP" ] && [ -d "$AOMP" ] ; then
         echo "ERROR: Directory $AOMP is a physical directory."
         echo "       It must be a symbolic link or not exist"
         exit 1
      fi
   fi

   check_writable_installdir "$1" "$AOMP_INSTALL_DIR"
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
   local SrcDir
   local BuildDir
   local AompCmake
   local UseNinja
   local LLVM_VERSION_MAJOR
   local osversion
   local TARGETS_TO_BUILD
   local -a AOMP_SET_NINJA_GEN
   local -a _cxx_flag
   local -a LLVMCMAKEOPTS

   SrcDir="$(get_src_dir)"
   BuildDir="$(get_build_dir "$Cfg")"
   AompCmake="$(cfgvar AOMP_CMAKE)"
   UseNinja="$(cfgbool AOMP_USE_NINJA)"

   if "$UseNinja"; then
      AOMP_SET_NINJA_GEN=(-G Ninja)
   fi

   osversion=$(grep -e ^VERSION_ID < /etc/os-release)
   if [[ $osversion =~ \"7\. ]] || [[ $osversion =~ \"8\. ]]; then
      _cxx_flag=(-DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0')
   fi

   # Legacy Flang dosen't support building of compiler-rt so it
   # utilizes the clang runtime libraries build/install using build_project.sh.
   # The LLVM_VERSION_MAJOR of classic flang driver has to match with the clang
   # binaries generated from build_project.sh.
   LLVM_VERSION_MAJOR=$("${LLVM_INSTALL_LOC}"/bin/clang --version | grep -oP '(?<=clang version )[0-9]+')

   TARGETS_TO_BUILD="AMDGPU;X86"

   LLVMCMAKEOPTS=(-DLLVM_ENABLE_PROJECTS=clang
                  -DCMAKE_BUILD_TYPE=Release
                  -DLLVM_ENABLE_ASSERTIONS=ON
                  -DLLVM_TARGETS_TO_BUILD="$TARGETS_TO_BUILD"
                  -DCLANG_DEFAULT_LINKER=lld
                  -DLLVM_VERSION_MAJOR="$LLVM_VERSION_MAJOR"
                  -DLLVM_INCLUDE_BENCHMARKS=0
                  -DLLVM_INCLUDE_RUNTIMES=0
                  -DLLVM_INCLUDE_EXAMPLES=0
                  -DLLVM_INCLUDE_TESTS=0
                  -DLLVM_INCLUDE_DOCS=0
                  -DLLVM_INCLUDE_UTILS=0
                  -DCLANG_DEFAULT_PIE_ON_LINUX=0
                  -DLLVM_ENABLE_ZSTD=OFF
                  "${_cxx_flag[@]}"
                  "${AOMP_SET_NINJA_GEN[@]}")

   mkdir -p "$BuildDir"
   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running llvm-classic $Cfg cmake ---- "
   echo "$AompCmake $(shquot "${LLVMCMAKEOPTS[@]}") $SrcDir"

   if ! "$AompCmake" "${LLVMCMAKEOPTS[@]}" "$SrcDir" 2>&1; then
      echo "ERROR cmake failed. Cmake flags"
      echo "      $(shquot "${LLVMCMAKEOPTS[@]}")"
      exit 1
   fi
   echo
   popd >& /dev/null || exit
}

task_build() {
   local Cfg=$1
   local BuildDir
   local NinjaBin
   local Jobs
   BuildDir="$(get_build_dir "$Cfg")"
   NinjaBin="$(cfgvar AOMP_NINJA_BIN)"
   Jobs="$(cfgvar AOMP_JOB_THREADS)"

   pushd "$BuildDir" >& /dev/null || exit
   echo " ---  Running $NinjaBin for $BuildDir ---- "
   if ! $NinjaBin -j "$Jobs"; then
      echo " "
      echo "ERROR: $NinjaBin -j $Jobs  FAILED"
      echo "To restart:"
      echo "  cd $BuildDir"
      echo "  $NinjaBin"
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

# List of tasks per config.  llvm-classic is built but not installed.
do_list_tasks() {
  local Cfg=$1
  if valid_config "$Cfg"; then
    echo "clean"
    echo "cmake"
    echo "build"
  else
    echo "Unknown config '$Cfg'"
  fi
}

command_dispatcher "$@"
