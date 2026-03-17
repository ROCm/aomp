#!/bin/bash
#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
# 
#
#  rebuild_srock_llvm.sh: iterative rebuild of LLVM after successful
#  build_srock.sh.
#
# --- Start standard header to set SROCK environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/srock_common_vars"
# --- end standard header ----
_curdir=$PWD
_start_date=$(date)
_start_secs=$(date +%s)

if [ ! -d "$SROCK_THEROCK_DIR" ] ; then
   echo " ERROR:  $0 requires that $SROCK_THEROCK_DIR exist and has "
   echo "         been setup with $thisdir/setup_srock.sh. Please run"
   echo
   echo "         nohup $thisdir/setup_srock.sh &"
   echo
   exit 1
fi

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

cd "$SROCK_THEROCK_DIR"
echo "===== Enable only LLVM project ====="
build_tools/buildctl.py enable amd-llvm

echo "===== Rebuild LLVM ====="
# The first build picks up any changes from the LLVM project.
cd "$SROCK_THEROCK_DIR"/build/compiler/amd-llvm/build || exit
ninja install || exit

echo "===== Install LLVM ====="
# The second build installs the compiler in the chosen installation directory.
cd "$SROCK_THEROCK_DIR"/build || exit
ninja install || exit

# Gather some build stats
_end_date=$(date)
_end_secs=$(date +%s)
_secs_to_build=$(( _end_secs - _start_secs ))
_filecount=$(find "$SROCK_INSTALL_DIR" -type f | wc -l)
_size=$(du -hs "$SROCK_INSTALL_DIR" | cut -f1)

echo
echo "===== DONE $0 on $_end_date"
echo "      THEROCK targets:   $_gfxsemicolons"
echo "      ROCm install dir:  $SROCK_INSTALL_DIR"
echo "      TheRock Dir:       $SROCK_THEROCK_DIR"
echo "      TheRock branch:    $SROCK_THEROCK_BRANCH"
echo "      Compiler branch:   $SROCK_COMPILER_BRANCH"
echo "      SROCK config name: $SROCK_CONFIG"
echo "      cmake command:     $SROCK_CMAKE"
echo "      Build time:        $_secs_to_build (seconds)"
echo "      Files:             $_filecount"
echo "      Size:              $_size"
echo "      cmake args:        ${_cmake_args[*]}"
echo
echo "      For aomp testing, set AOMP=$SROCK_LINK"
echo "         or AOMP=$SROCK_INSTALL_DIR"
echo
