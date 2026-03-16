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

(
cd "$SROCK_THEROCK_DIR" || exit
# reconstruct .amd-llvm.smrev using the current SHA
cd compiler/amd-llvm || exit
smrev="../.amd-llvm.smrev"
git config --get remote.origin.url > "$smrev"
smsha=$(git rev-parse HEAD)
echo "${smsha}${LLVM_SHA_EXTRA}" >> "$smrev"
)

cd "$SROCK_THEROCK_DIR" || exit
_cmd="cmake --build build"
echo 
echo "===== build CMD: $_cmd"
$_cmd
_rc=$? && [ "$_rc" != 0 ] && cd "$_curdir" && exit "$_rc"
date

echo
echo "===== Install into $SROCK_INSTALL_DIR"
cd "$SROCK_THEROCK_DIR/build" || exit
echo "mkdir -p $SROCK_INSTALL_DIR"
mkdir -p "$SROCK_INSTALL_DIR"
_cmd="ninja install"
$_cmd
_rc=$? && [ "$_rc" != 0 ] && cd "$_curdir" && exit "$_rc"
date

# TheRock does not yet build hipfort.
# FIXME:  remove this stanza and the build_hipfort.sh script when it does
if [ ! -d "${SROCK_REPOS}/hipfort/build" ] ; then 
   echo
   echo "===== Sourcing build_hipfort.sh to build and install hipfort"
   . "$thisdir/build_hipfort.sh"
fi

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
echo "      Build time:        $_secs_to_build (seconds)"
echo "      Files:             $_filecount"
echo "      Size:              $_size"
echo "      cmake args:        ${_cmake_args[*]}"
echo
echo "      For aomp testing, set AOMP=$SROCK_LINK"
echo "         or AOMP=$SROCK_INSTALL_DIR"
echo
