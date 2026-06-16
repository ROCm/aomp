#!/bin/bash
# 
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
# 
#  build_flang-classic.sh:  Script to build the flang-classic binary driver
#         This driver will never call flang -fc1, it only calls binaries 
#             clang, flang1, flang2, build elsewhere
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
  get_config_var_string flang-classic "$1"
}

cfgbool() {
  get_config_var_bool flang-classic "$1"
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
   echo "$(cfgvar AOMP_REPOS)/$(cfgvar AOMP_FLANG_REPO_NAME)/flang-classic/$(cfgvar AOMP_LFL_DIR)"
}

# Print the build dir for a given config, passed as $1.
get_build_dir() {
   local Cfg=$1
   local BuildDir
   BuildDir="$(cfgvar BUILD_DIR)"

   case "$Cfg" in
   "default")
     echo -n "$BuildDir/flang-classic/$(cfgvar AOMP_LFL_DIR)"
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

   # The flang-classic build dir is shared with llvm-classic (built first by
   # build_llvm-classic.sh); remove everything except the llvm-classic subdir.
   if [ -d "$BuildDir" ]; then
      local Entry
      for Entry in "${BuildDir:?}"/* "${BuildDir:?}"/.[!.]*; do
         [ -e "$Entry" ] || continue
         case "$(basename "$Entry")" in
            llvm-classic) ;;
            *) echo "rm -rf $Entry"; rm -rf "$Entry" ;;
         esac
      done
   else
      echo "ERROR: Build llvm-classic before flang-classic."
      exit 1
   fi
}

task_cmake() {
   local Cfg=$1
   local SrcDir
   local BuildDir
   local AompCmake
   local UseNinja
   local Standalone
   local osversion
   local -a AOMP_SET_NINJA_GEN
   local -a _cxx_flag
   local -a MYCMAKEOPTS

   SrcDir="$(get_src_dir)"
   BuildDir="$(get_build_dir "$Cfg")"
   AompCmake="$(cfgvar AOMP_CMAKE)"
   UseNinja="$(cfgbool AOMP_USE_NINJA)"
   Standalone="$(cfgbool AOMP_STANDALONE_BUILD)"

   if "$UseNinja"; then
      AOMP_SET_NINJA_GEN=(-G Ninja)
   fi

   osversion=$(grep -e ^VERSION_ID < /etc/os-release)
   if [[ $osversion =~ \"7\. ]] || [[ $osversion =~ \"8\. ]]; then
      _cxx_flag=(-DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0')
   fi

   MYCMAKEOPTS=(-DCMAKE_BUILD_TYPE="$(cfgvar BUILD_TYPE)"
                -DCMAKE_C_COMPILER="$LLVM_INSTALL_LOC/bin/clang"
                -DCMAKE_CXX_COMPILER="$LLVM_INSTALL_LOC/bin/clang++"
                "${_cxx_flag[@]}"
                -DCMAKE_CXX_STANDARD=17
                -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_LOC"
                "${AOMP_SET_NINJA_GEN[@]}")

   if "$Standalone" ; then
      MYCMAKEOPTS+=(-DBUILD_SHARED_LIBS=ON "${AOMP_ORIGIN_RPATH[@]}")
   else
      MYCMAKEOPTS+=(-DBUILD_SHARED_LIBS=OFF "${OPENMP_EXTRAS_ORIGIN_RPATH[@]}")
   fi

   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Running flang-classic $Cfg cmake ---- "
   echo "$AompCmake $(shquot "${MYCMAKEOPTS[@]}") $SrcDir"

   if ! "$AompCmake" "${MYCMAKEOPTS[@]}" "$SrcDir" 2>&1; then
      echo "ERROR cmake failed. Cmake flags"
      echo "      $(shquot "${MYCMAKEOPTS[@]}")"
      exit 1
   fi
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

task_install() {
   local Cfg=$1
   local BuildDir
   local AompCmake
   local Jobs
   BuildDir="$(get_build_dir "$Cfg")"
   AompCmake="$(cfgvar AOMP_CMAKE)"
   Jobs="$(cfgvar AOMP_JOB_THREADS)"

   pushd "$BuildDir" >& /dev/null || exit
   echo " -----Installing to $AOMP_INSTALL_DIR ---- "
   if ! $SUDO "$AompCmake" --build . -j "$Jobs" --target install; then
      echo "ERROR make install failed "
      exit 1
   fi
   echo
   echo "SUCCESSFUL INSTALL to $AOMP_INSTALL_DIR"
   echo
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
