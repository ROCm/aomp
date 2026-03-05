#!/bin/bash
#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
# 
#
#  post_setup_build_srock.sh: build TheRock after successful setup_srock.sh
#
# --- Start standard header to set SROCK environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/srock_common_vars"
# --- end standard header ----
_curdir=$PWD
_start_date=$(date)
_start_secs=$(date +%s)

# TWO QUICK CHECKS, MUST HAVE SROCK_REPOS and SROCK_THEROCK_DIR MUST EXIST
mkdir -p "$SROCK_REPOS"
if [ ! -d "$SROCK_REPOS" ] ; then
   echo
   echo "ERROR: $0 could not create directory $SROCK_REPOS"
   echo "       Consider setting SROCK_REPOS to use a large fast fileystem."
   echo "       or $HOME/git/srock-repos"
   echo
   exit 1
fi
if [ ! -d "$SROCK_THEROCK_DIR" ] ; then
   echo " ERROR:  $0 requires that $SROCK_THEROCK_DIR exist and has "
   echo "         been setup with $thisdir/setup_srock.sh. Please run"
   echo
   echo "         nohup $thisdir/setup_srock.sh &"
   echo
   exit 1
fi

declare -a _cmake_enable
_cmake_enable=()

# This TheRock config is full build minus failing components
if [ "$SROCK_CONFIG" == "all" ] ; then
   _cmake_enable=("${_cmake_enable[@]}"
      -DTHEROCK_ENABLE_ALL=ON
      -DTHEROCK_ENABLE_MIOPEN=OFF
      -DTHEROCK_ENABLE_COMPOSABLE_KERNEL=OFF
      -DTHEROCK_ENABLE_FFT=OFF
   )
fi

# This is full build which could include failing components
if [ "$SROCK_CONFIG" == "all-debug" ] ; then
   _cmake_enable=("${_cmake_enable[@]}"
           -DTHEROCK_ENABLE_ALL=ON )
fi

# Default is minimal for compiler developers
if [ "$SROCK_CONFIG" == "minimal" ] ; then
   _cmake_enable=("${_cmake_enable[@]}"
      -DTHEROCK_ENABLE_ALL=OFF
      -DTHEROCK_ENABLE_HIP=ON
      -DTHEROCK_ENABLE_HIP_RUNTIME=ON
      -DTHEROCK_ENABLE_HIPIFY=ON
      -DTHEROCK_BUNDLE_SYSDEPS=ON
      -DTHEROCK_ENABLE_COMPILER=ON
   )
fi

_gfxsemicolons=$(echo "$GFXLIST" | tr ' ' ';')
_gfamsemicolons=$(echo "$GFXFAM" | tr ' ' ';')

declare -a _cmake_args
_cmake_args=(-B build -GNinja
   -DTHEROCK_AMDGPU_TARGETS='$_gfxsemicolons'
   -DTHEROCK_AMDGPU_FAMILIES='$_gfamsemicolons'
   -DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=srock
   -DTHEROCK_BACKGROUND_BUILD_JOBS=1
   -DTHEROCK_ENABLE_LLVM_TESTS=1
   "${_cmake_enable[@]}"
   "$SROCK_THEROCK_DIR"
)

#  _cmake_enable and _cmake_args are not used in post_setup_build_srock.sh
#  above is repeated from setup_srock.sh to make the start and end banner useful.

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

cd "$SROCK_THEROCK_DIR" || exit
_cmd="cmake --build build"
echo 
echo "===== build CMD: $_cmd"
$_cmd
_rc=$? && [ "$_rc" != 0 ] && cd "$_curdir" && exit "$_rc"
date

# Usually nothing to do for therock-dist
cd build || exit
_cmd="ninja therock-dist"
echo 
echo "===== dist CMD: $_cmd"
$_cmd
_rc=$? && [ "$_rc" != 0 ] && cd "$_curdir" && exit "$_rc"
date

echo
echo "===== copying ROCm build from $SROCK_THEROCK_DIR/build/dest/rocm to $SROCK_INSTALL_DIR" 
# FIXME: instead of rsync --delete consider using artifact descriptors to copy files to installation
cd "$SROCK_THEROCK_DIR/build" || exit
echo "mkdir -p $SROCK_INSTALL_DIR"
mkdir -p "$SROCK_INSTALL_DIR"
echo "rsync -a --delete dist/rocm/ $SROCK_INSTALL_DIR/"
rsync -a --delete dist/rocm/ "$SROCK_INSTALL_DIR"/
# FileCheck binary not found in dist/rocm, so get it from amd-llvm build
echo cp -p  ./compiler/amd-llvm/build/bin/FileCheck "$SROCK_INSTALL_DIR/lib/llvm/bin/FileCheck"
cp -p  ./compiler/amd-llvm/build/bin/FileCheck "$SROCK_INSTALL_DIR/lib/llvm/bin/FileCheck"

if [ ! -d "${SROCK_REPOS}/hipfort/build" ] ; then 
   echo
   echo "===== Sourcing build_hipfort.sh to build and install hipfort"
   . "$thisdir/build_hipfort.sh"
fi
# FIXME: Add builds for rocdbgapi and rocgdb here 

echo
echo "===== Creating compiler cfg files "
amd_compiler_cfg=("clang" "clang++" "clang-cpp" "clang-${SROCK_MAJOR_VERSION}" "clang-cl" "flang")
echo "--rocm-path='<CFGDIR>/../../..'" >"$SROCK_INSTALL_DIR"/lib/llvm/bin/rocm.cfg
echo "-frtlib-add-rpath" >>"$SROCK_INSTALL_DIR"/lib/llvm/bin/rocm.cfg
for ii in "${amd_compiler_cfg[@]}" ; do
   if [ -f "${SROCK_INSTALL_DIR}/lib/llvm/bin/$ii" ] ; then
      echo "Creating config file: ${ii}.cfg in ${SROCK_INSTALL_DIR}/lib/llvm//bin"
      config_file="${SROCK_INSTALL_DIR}/lib/llvm/bin/${ii}.cfg"
      echo "@rocm.cfg" > "$config_file"
   fi
done

# Gather some build stats
_end_date=$(date)
_end_secs=$(date +%s)
_secs_to_build=$(( _end_secs - _start_secs ))
_filecount=$(find "$SROCK_INSTALL_DIR" -type f | wc -l)
_size=$(du -hs "$SROCK_INSTALL_DIR" | cut -f1)

echo
echo "===== Linking $SROCK_INSTALL_DIR to $SROCK_LINK" 
if [[ -e "$SROCK_LINK" && ! -L "$SROCK_LINK" ]] ; then
   echo "WARNING: $SROCK_LINK exists and is NOT a link"
fi
echo "rm -f $SROCK_LINK"
rm -f "$SROCK_LINK"
echo ln -sf "$SROCK_INSTALL_DIR" "$SROCK_LINK"
ln -sf "$SROCK_INSTALL_DIR" "$SROCK_LINK"

echo
echo "===== DONE $0 on $_end_date"
echo "      THEROCK targets:   $_gfxsemicolons"
echo "      ROCm install dir:  $SROCK_INSTALL_DIR"
echo "      TheRock Dir:       $SROCK_THEROCK_DIR"
echo "      TheRock branch:    $SROCK_THEROCK_BRANCH"
echo "      Compiler branch:   $SROCK_COMPILER_BRANCH"
echo "      SROCK config name: $SROCK_CONFIG"
echo "      cmake command:     $SROCK_CMAKE"
echo "      cmake args:        ${_cmake_args[*]}"
echo "      Build time:        $_secs_to_build (seconds)"
echo "      Files:             $_filecount"
echo "      Size:              $_size"
echo
echo "      For aomp testing, set AOMP=$SROCK_LINK"
echo "         or AOMP=$SROCK_INSTALL_DIR"
echo
