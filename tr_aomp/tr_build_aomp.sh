#!/bin/bash
#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
#
#  tr_build_aomp.sh : Build aomp using TheRock 
#
# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
# shellcheck disable=SC1091
source "$thisdir"/tr_aomp_common_vars
# --- end standard header ----
#
_curdir=$PWD

if [ -z "$AOMP_INSTALL_DIR" ] ; then
   echo "ERROR: Env VAR AOMP_INSTALL_DIR is not set "
   cd "$_curdir" || exit
   exit 1
fi

_therockdir=$TR_AOMP_REPOS/TheRock

cd "$_therockdir" || exit
if [ -d "$_therockdir"/.venv/bin ] ; then
   PATH="$_therockdir"/.venv/bin:$PATH
   export PATH
fi
(
# reconstruct .amd-llvm.smrev using the current SHA
cd compiler/amd-llvm || exit
smrev="../.amd-llvm.smrev"
git config --get remote.origin.url > "$smrev"
git rev-parse HEAD >> "$smrev"
)
_start_date=$(date)
_start_secs=$(date +%s)

cd "$_therockdir" || exit
if [ "$1" == "restart" ] ; then
   if [ -d build ] ; then
      echo "----- Restart $0 from a previous build in $_therockdir/build -----"
      _rsync_v="v"
   else
      echo "----- Fresh start $0 -----"
      echo "      No build directory ($_therockdir/build) found, so fresh start"
      [ -d "$AOMP_INSTALL_DIR" ] && echo "      rm -rf $AOMP_INSTALL_DIR" && rm -rf "$AOMP_INSTALL_DIR"
      mkdir -p "$_therockdir/build"
      _rsync_v=""
   fi
else
   echo "----- Fresh start $0 -----"
   if [ -d build ] ; then
      echo "      To avoid this fresh start, in the future use: $0 restart"
      echo "      cd $_therockdir; rm -rf build" && rm -rf build
   fi
   [ -d "$AOMP_INSTALL_DIR" ] && echo "      rm -rf $AOMP_INSTALL_DIR" && rm -rf "$AOMP_INSTALL_DIR"
   [ -L "$AOMP" ] && echo "      rm $AOMP" && rm "$AOMP"
   [ -d "$AOMP_BUILD_LOGS" ]  && echo "      rm -rf $AOMP_BUILD_LOGS"  && rm -rf "$AOMP_BUILD_LOGS"
   mkdir -p "$_therockdir/build"
   _rsync_v=""
fi

if [ "${AOMP_SKIP_RCCL}" == 1 ] ; then
   _rccl_opt="-DTHEROCK_ENABLE_RCCL=OFF"
else
   _rccl_opt="-DTHEROCK_ENABLE_RCCL=ON"
fi
if [ "${AOMP_SKIP_MATH_LIBS}" == 1 ] ; then
   _mathlibs_opt="-DTHEROCK_ENABLE_MATH_LIBS=OFF"
else
   _mathlibs_opt="-DTHEROCK_ENABLE_MATH_LIBS=ON"
fi
if [ "${AOMP_SKIP_ML_LIBS}" == 1 ] ; then
   _mllibs_opt="-DTHEROCK_ENABLE_ML_LIBS=OFF"
else
   _mllibs_opt="-DTHEROCK_ENABLE_ML_LIBS=ON"
fi
if [ "${AOMP_SKIP_COMPOSABLE_KERNEL}" == 1 ] ; then
   _composablek_opt="-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=OFF"
else
   _composablek_opt="-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON"
fi
if [ "${AOMP_SKIP_FFT}" == 1 ] ; then
   _fft_opt="-DTHEROCK_ENABLE_FFT=OFF"
else
   _fft_opt=""
fi

_gfxsemicolons=$(echo "$GFXLIST" | tr ' ' ';')
# shellcheck disable=SC2089
_cmake_cmd="cmake -B build -GNinja -DTHEROCK_AMDGPU_TARGETS='$_gfxsemicolons' -DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=aomp $_composablek_opt $_mathlibs_opt $_mllibs_opt -DTHEROCK_BUNDLE_SYSDEPS=ON -DTHEROCK_BUILD_TESTING=OFF $_rccl_opt $_fft_opt $_therockdir"

#Record config and PATH in AOMP release info file"
"$thisdir"/tr_add_info.sh therock_config "$_cmake_cmd"
"$thisdir"/tr_add_info.sh build_path "$PATH"

eval "$(python3 ./build_tools/setup_ccache.py)"

if [ "$1" == "restart" ] ; then
   echo "----- Skipping TheRock cmake because this is a restart"
else
   echo
   echo "===== CMD:$_cmake_cmd"
   # shellcheck disable=SC2090
   $_cmake_cmd
   _rc=$? && [ "$_rc" != 0 ] && cd "$_curdir" && exit "$_rc"
fi

_cmd="cmake --build build"
echo 
echo "===== CMD:$_cmd"
$_cmd
_rc=$? && [ "$_rc" != 0 ] && cd "$_curdir" && exit "$_rc"
date

# Usually nothing to do for therock-dist
cd build || exit
_cmd="ninja therock-dist"
echo 
echo "===== CMD:$_cmd"
$_cmd
_rc=$? && [ "$_rc" != 0 ] && cd "$_curdir" && exit "$_rc"
date

echo
echo "===== copying ROCm build from $_therockdir/build/dest/rocm to $AOMP_INSTALL_DIR" 
cd "$_therockdir/build" || exit
echo "mkdir -p $AOMP_INSTALL_DIR"
mkdir -p "$AOMP_INSTALL_DIR"
echo "rsync -a$_rsync_v --delete dist/rocm/ $AOMP_INSTALL_DIR/"
rsync -a$_rsync_v --delete dist/rocm/ "$AOMP_INSTALL_DIR"/
# FileCheck binary not found in dist/rocm, so get it from amd-llvm build
echo cp -p  ./compiler/amd-llvm/build/bin/FileCheck "$AOMP_INSTALL_DIR/lib/llvm/bin/FileCheck"
cp -p  ./compiler/amd-llvm/build/bin/FileCheck "$AOMP_INSTALL_DIR/lib/llvm/bin/FileCheck"

echo
echo "===== Linking $AOMP_INSTALL_DIR to $AOMP ====="
if [ -L "$AOMP" ] ; then
   rm "$AOMP"
fi
echo ln -sf "$AOMP_INSTALL_DIR" "$AOMP"
ln -sf "$AOMP_INSTALL_DIR" "$AOMP"

# rocm.cfg content
echo
echo "===== Creating compiler cfg files "
amd_compiler_cfg=("clang" "clang++" "clang-cpp" "clang-${AOMP_MAJOR_VERSION}" "clang-cl" "flang")
echo "--rocm-path='<CFGDIR>/../../..'" >"$AOMP_INSTALL_DIR"/lib/llvm/bin/rocm.cfg
echo "-frtlib-add-rpath" >>"$AOMP_INSTALL_DIR"/lib/llvm/bin/rocm.cfg
for ii in "${amd_compiler_cfg[@]}" ; do
   if [ -f "${AOMP_INSTALL_DIR}/lib/llvm/bin/$ii" ] ; then
      echo "Creating config file: ${ii}.cfg in ${AOMP_INSTALL_DIR}/lib/llvm//bin"
      config_file="${AOMP_INSTALL_DIR}/lib/llvm/bin/${ii}.cfg"
      echo "@rocm.cfg" > "$config_file"
   fi
done

echo
echo "===== Saving TheRock build stats to $AOMP_INFO_FILE"
#echo "rsync -a --delete $_therockdir/build/logs/ $AOMP_BUILD_LOGS/"
#rsync -a --delete "$_therockdir"/build/logs/ "$AOMP_BUILD_LOGS"/
_end_date=$(date)
_end_secs=$(date +%s)
echo
echo "START  : $_start_date"
echo "END    : $_end_date"
_secs_to_build=$(( _end_secs - _start_secs ))
echo "TIME   : $_secs_to_build (seconds)"
_filecount=$(find "$AOMP_INSTALL_DIR" -type f | wc -l)
echo "FILES  : $_filecount"
_size=$(du -hs "$AOMP_INSTALL_DIR" | cut -f1)
echo "SIZE   : $_size"
"$thisdir"/tr_add_info.sh start_build_date "$_start_date"
"$thisdir"/tr_add_info.sh end_build_date "$_end_date"
"$thisdir"/tr_add_info.sh secs_to_build "$_secs_to_build"
"$thisdir"/tr_add_info.sh file_count "$_filecount"
"$thisdir"/tr_add_info.sh size "$_size"
if [ "$1" == "restart" ] ; then
   "$thisdir"/tr_add_info.sh "restart" YES
   echo "RESTART: YES"
else
   echo "RESTART: NO"
   "$thisdir"/tr_add_info.sh "restart" NO
fi
echo
echo "===== DONE $0 for AOMP release $AOMP_VERSION_STRING"
echo
