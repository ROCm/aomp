#!/bin/bash
#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
# 
#  setup_srock.sh: Clone and initialize TheRock Repo. 
#
# --- Start standard header to set SROCK environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
# shellcheck disable=1091
. "$thisdir/srock_common_vars"
# --- end standard header ----
#

if [ -d "$SROCK_THEROCK_DIR" ] ; then
   echo " ERROR:  $0 requires that $SROCK_THEROCK_DIR NOT exist"
   echo "         Delete or move that directory to run $0"
   exit 1
fi

_curdir=$PWD
_start_date=$(date)
_start_secs=$(date +%s)

# Print the start banner similar to DONE banner, useful if fails
echo
echo "===== START $0 on $_start_date"
echo "      THEROCK targets:   $_gfxsemicolons"
echo "      THEROCK families:  $_gfamsemicolons"
echo "      ROCm install dir:  $SROCK_INSTALL_DIR"
echo "      TheRock Dir:       $SROCK_THEROCK_DIR"
echo "      TheRock branch:    $SROCK_THEROCK_BRANCH"
echo "      Compiler branch:   $SROCK_COMPILER_BRANCH"
echo "      SROCK config name: $SROCK_CONFIG"
echo "      cmake args:        ${_cmake_args[*]}"

# Run srock prebuild which includes finding suitable cmake
echo
echo "===== Sourcing prebuild_srock.sh"
. "$thisdir/prebuild_srock.sh"
echo "===== DONE Sourcing prebuild_srock.sh"

cd "$SROCK_REPOS" || exit
echo
echo "===== git clone https://github.com/ROCm/TheRock.git -b $SROCK_THEROCK_BRANCH TheRock"
git clone https://github.com/ROCm/TheRock.git -b "$SROCK_THEROCK_BRANCH" TheRock
if [ -f TheRock/version.json ] ; then 
   _quoted=$(cat TheRock/version.json | grep rocm-version | cut -d: -f2)
   _rocm_version=${_quoted//\"/}
   echo " ROCm components version : $_rocm_version"
   echo " SROCK_VERSION_STRING    :  $SROCK_VERSION_STRING (Compiler dev version)"
fi

cd "$SROCK_THEROCK_DIR" || exit

if [ ! -d "$SROCK_THEROCK_DIR/.venv/bin" ] ; then
   echo
   echo "===== Building virtual environment in .venv and updating PATH ====="
   cd "$SROCK_THEROCK_DIR" || exit
   echo "python3 -m venv .venv && source .venv/bin/activate"
   # shellcheck disable=1091
   python3 -m venv .venv && source ".venv/bin/activate"
   echo "pip install -r requirements.txt"
   pip install -r requirements.txt
fi
export PATH=$SROCK_THEROCK_DIR/.venv/bin:$PATH

echo
echo "===== Running python ./build_tools/fetch_sources.py ====="
python ./build_tools/fetch_sources.py
echo "=====  Done running python ./build_tools/fetch_sources.py"

echo "cd $SROCK_THEROCK_DIR" 
cd "$SROCK_THEROCK_DIR" || exit
[ -d build ] && echo "WARNING build directory $SROCK_THEROCK_DIR/build should not exist "

echo 
echo "===== Running build_tools/setup_ccache.py"
eval "$(python3 ./build_tools/setup_ccache.py)"

# Make updates to compiler submodules unless this is native TheRock build 
if [ "$SROCK_COMPILER_BRANCH" != "develop" ] ; then 
   # FIXME: Before wiping out current amd-staging changes, 
   #        to save current changes in the patches directory. 
   #        Otherwise, this is not a real development environment"
   echo
   echo "===== Switch to $SROCK_COMPILER_BRANCH branch for compiler components"
   echo "      --- cd $SROCK_THEROCK_DIR/compiler/hipify"
   cd "$SROCK_THEROCK_DIR/compiler/hipify" || exit
   echo "      --- git checkout ."
   git checkout .
   echo "      --- git checkout $SROCK_COMPILER_BRANCH (WARNING: This may leave commits behind"
   git checkout "$SROCK_COMPILER_BRANCH" 
   echo "      --- git pull (gets most recent updates to $SROCK_COMPILER_BRANCH)"
   git pull

   echo "      --- cd $SROCK_THEROCK_DIR/compiler/spirv-llvm-translator"
   cd "$SROCK_THEROCK_DIR/compiler/spirv-llvm-translator" || exit
   echo "      --- git checkout ."
   git checkout .
   echo "      --- git checkout $SROCK_COMPILER_BRANCH"
   git checkout "$SROCK_COMPILER_BRANCH"
   echo "      --- git pull (gets most recent updates to $SROCK_COMPILER_BRANCH)"
   git pull

   echo "      --- cd $SROCK_THEROCK_DIR/compiler/amd-llvm"
   cd "$SROCK_THEROCK_DIR/compiler/amd-llvm" || exit
   echo "      --- git checkout ."
   git checkout .
   echo "      --- git checkout $SROCK_COMPILER_BRANCH (WARNING: This leaves commits behind for amd-llvm)"
   git checkout "$SROCK_COMPILER_BRANCH"
   echo "      --- git pull (gets most recent updates to $SROCK_COMPILER_BRANCH)"
   git pull

   if [ -d "$thisdir/patches/$SROCK_COMPILER_BRANCH" ] ; then 
      cd "$SROCK_THEROCK_DIR" || exit
         _patch_file=$thisdir/patches/$SROCK_COMPILER_BRANCH/_TheRock.patch
      if [ -f "$_patch_file" ] ; then
         test_apply_patch
      fi
      _tmpfile=/tmp/submod$$
      git submodule > "$_tmpfile"
      echo "tmpfile:$_tmpfile"
      while read -r _line ; do
	 _subdir=$(echo "$_line" | cut -d" " -f2)
	 _subdirfull="$SROCK_THEROCK_DIR/$_subdir"
	 if [ ! -d "$_subdirfull" ] ; then 
            echo "Directory $_subdirfull does not exist "
         else 
            cd "$_subdirfull" || exit
	    _subdirname=$(echo "$_subdir" | tr "/" "_")
            _patch_file=$thisdir/patches/$SROCK_COMPILER_BRANCH/$_subdirname.patch
            if [ -f "$_patch_file" ] ; then
               test_apply_patch
            else
               echo "PATCHFILE $_patch_file DOES NOT EXIST"
            fi
	 fi
      done < $_tmpfile
      rm "$_tmpfile"
   fi

echo "      --- end compiler submodule updates for $SROCK_COMPILER_BRANCH"

cd "$SROCK_THEROCK_DIR" || exit
echo 
echo "===== cmake CMD: $SROCK_CMAKE ${_cmake_args[*]}"
$SROCK_CMAKE "${_cmake_args[@]}"
_rc=$? && [ "$_rc" != 0 ] && cd "$_curdir" && exit "$_rc"
fi

_setup_secs=$(date +%s)
_secs_to_setup=$(( _setup_secs - _start_secs ))

echo
echo "===== DONE $0 on $_start_date"
echo "   THEROCK targets:      $_gfxsemicolons"
echo "   ROCm comp version:    $_rocm_version"
echo "   SROCK_VERSION_STRING: $SROCK_VERSION_STRING (Compiler dev version)"
echo "   THEROCK families:     $_gfamsemicolons"
echo "   ROCm install dir:     $SROCK_INSTALL_DIR"
echo "   TheRock Dir:          $SROCK_THEROCK_DIR"
echo "   TheRock branch:       $SROCK_THEROCK_BRANCH"
echo "   Compiler branch:      $SROCK_COMPILER_BRANCH"
echo "   SROCK config name:    $SROCK_CONFIG"
echo "   Setup time:           $_secs_to_setup (seconds)"
echo "   cmake args:           ${_cmake_args[*]}"
echo 
echo " Next step, run this command: $thisdir/build_srock.sh" 
echo 

