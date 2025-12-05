#!/bin/bash
#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
# 
#
#  build_srock.sh: Clone and build TheRock with amd-staging compiler.
#
# --- Start standard header to set SROCK environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/srock_common_vars"
# --- end standard header ----
function test_apply_patch() {
   if ! patch -p1 -t -N --merge --dry-run < $_patch_file  >/dev/null; then
      echo "ERROR:  patch --dry-run failed.  Could not apply $_patch_file "
      cd $_curdir
      exit 1
   else
      echo "patch -p1 --no-backup-if-mismatch --merge < $_patch_file"
      patch -p1 --no-backup-if-mismatch --merge < $_patch_file
   fi
}

_curdir=$PWD
_start_date=$(date)
_start_secs=$(date +%s)

# build_srock.sh Modes:
#   The default mode "fullupdate" will
#     - if TheRock repo exists, remove all previous updates 
#       and update TheRock and submodules to the tip
#       else clone TheRock
#     - if not develop, get fresh amd-staging updates to compiler repos
#     - rerun fetch_sources.py
#     - if not develop, apply srock patches/amd-staging/* patches
#     - remove previous build dir completely
#     - Run TheRock cmake configuration. 
#     - do full build 
#   The restart mode
#     - assumes TheRock repo exists
#     - assumes TheRocke/build dir exists
#     - assumes TheRock config has been run
#     - assumes all data preparation has been run
#     - restarts build in TheRock dir with cmake --build build
# FIXME: Add support for build modes: newclone and newbuild
#
_build_srock_mode=${1:-fullupdate} 
_cmake_enable=""

# This TheRock config is full build minus failing components
if [ "$SROCK_CONFIG" == "all" ] ; then
   _cmake_enable="\
-DTHEROCK_ENABLE_ALL=ON \
-DTHEROCK_ENABLE_MIOPEN=OFF \
-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=OFF \
-DTHEROCK_ENABLE_FFT=OFF \
"
fi

# This is full build which could include failing components
if [ "$SROCK_CONFIG" == "all-debug" ] ; then
   _cmake_enable="\
-DTHEROCK_ENABLE_ALL=ON \
"
fi

# Default is minimal for compiler developers
if [ "$SROCK_CONFIG" == "minimal" ] ; then
   _cmake_enable="\
-DTHEROCK_ENABLE_ALL=OFF \
-DTHEROCK_ENABLE_HIP=ON \
-DTHEROCK_ENABLE_HIP_RUNTIME=ON \
-DTHEROCK_ENABLE_HIPIFY=ON \
-DTHEROCK_BUNDLE_SYSDEPS=ON \
-DTHEROCK_ENABLE_COMPILER=ON \
"
fi

_gfxsemicolons=$(echo "$GFXLIST" | tr ' ' ';')
_gfamsemicolons=$(echo "$GFXFAM" | tr ' ' ';')

_cmake_args="-B build -GNinja \
-DTHEROCK_AMDGPU_TARGETS='$_gfxsemicolons' \
-DTHEROCK_AMDGPU_FAMILIES='$_gfamsemicolons' \
-DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=srock \
-DTHEROCK_BACKGROUND_BUILD_JOBS=1 \
$_cmake_enable \
$SROCK_THEROCK_DIR"

# Print the start banner similar to DONE banner, useful if fails
echo
echo "===== START $0 on $_start_date"
echo "      THEROCK targets:   $_gfxsemicolons"
echo "      THEROCK families:  $_gfamsemicolons"
echo "      ROCm install dir:  $SROCK_INSTALL_DIR"
echo "      TheRock Dir:       $SROCK_THEROCK_DIR"
echo "      Compiler branch:   $SROCK_COMPILER_BRANCH"
echo "      Build Mode:        $_build_srock_mode"
echo "      SROCK config name: $SROCK_CONFIG"
echo "      cmake args:        $_cmake_args"

# Run srock prebuild which includes finding suitable cmake
echo
echo "===== Sourcing prebuild_srock.sh"
. $thisdir/prebuild_srock.sh
echo "===== DONE Sourcing prebuild_srock.sh"

if [ "$_build_srock_mode" != "restart" ] ; then 
# ------------  BEGINNING of NOT restart mode ------------

# srock_common_vars ensures that SROCK_REPOS is created
mkdir -p $SROCK_REPOS
if [ ! -d $SROCK_REPOS ] ; then 
   echo
   echo "ERROR: $0 could not create directory $SROCK_REPOS"
   echo "       Consider setting SROCK_REPOS to use a large fast fileystem."
   echo "       or $HOME/git/srock-repos"
   echo
   exit 1
fi

if [ -d $SROCK_THEROCK_DIR ] ; then 
   echo
   echo "===== Updating existing TheRock clone in $SROCK_THEROCK_DIR"
   echo "      --- cd $SROCK_THEROCK_DIR/compiler/amd-llvm"
   cd $SROCK_THEROCK_DIR/compiler/amd-llvm
   echo "      --- git checkout . (clean previous amd-llvm local patches)"
   git checkout .
   echo "      --- cd $SROCK_THEROCK_DIR"
   cd $SROCK_THEROCK_DIR
   echo "      --- git checkout . (clean TheRock repo for update and to be patched)"
   git checkout .
   echo "      --- git pull"
   git pull
   echo "      --- git submodule update --remote --recursive"
   git submodule update --remote --recursive
   echo "      --- cd $SROCK_THEROCK_DIR/rocm-systems"
   cd $SROCK_THEROCK_DIR/rocm-systems
   echo "      --- git submodule update (to pickup changes to external repos)"
   git submodule update 
else
   cd $SROCK_REPOS
   echo
   echo "===== git clone https://github.com/ROCm/TheRock.git -b main TheRock"
   git clone https://github.com/ROCm/TheRock.git -b main TheRock
fi

cd $SROCK_THEROCK_DIR

if [ ! -d $SROCK_THEROCK_DIR/.venv/bin ] ; then
   echo
   echo "===== Building virtual environment in .venv and updating PATH ====="
   cd $SROCK_THEROCK_DIR
   echo "python3 -m venv .venv && source .venv/bin/activate"
   python3 -m venv .venv && source .venv/bin/activate
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
[ -d build ] && echo "rm -rf build" && rm -rf build

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
   cd $SROCK_THEROCK_DIR/compiler/hipify
   echo "      --- git checkout ."
   git checkout .
   echo "      --- git checkout $SROCK_COMPILER_BRANCH (WARNING: This may leave commits behind"
   git checkout $SROCK_COMPILER_BRANCH 
   echo "      --- git pull (gets most recent updates to $SROCK_COMPILER_BRANCH)"
   git pull

   echo "      --- cd $SROCK_THEROCK_DIR/compiler/spirv-llvm-translator"
   cd $SROCK_THEROCK_DIR/compiler/spirv-llvm-translator
   echo "      --- git checkout ."
   git checkout .
   echo "      --- git checkout $SROCK_COMPILER_BRANCH"
   git checkout $SROCK_COMPILER_BRANCH
   echo "      --- git pull (gets most recent updates to $SROCK_COMPILER_BRANCH)"
   git pull

   echo "      --- cd $SROCK_THEROCK_DIR/compiler/amd-llvm"
   cd $SROCK_THEROCK_DIR/compiler/amd-llvm
   echo "      --- git checkout ."
   git checkout .
   echo "      --- git checkout $SROCK_COMPILER_BRANCH (WARNING: This leaves commits behind for amd-llvm)"
   git checkout $SROCK_COMPILER_BRANCH 
   echo "      --- git pull (gets most recent updates to $SROCK_COMPILER_BRANCH)"
   git pull

   if [ -d "$thisdir/patches/$SROCK_COMPILER_BRANCH" ] ; then 
      cd $SROCK_THEROCK_DIR
      _patch_file=$thisdir/patches/$SROCK_COMPILER_BRANCH/_TheRock.patch
      if [ -f "$_patch_file" ] ; then
         test_apply_patch
      fi
      _tmpfile=/tmp/submod$$
      git submodule > $_tmpfile
      while read _line ; do
         _subdir=`echo $_line | cut -d" " -f2`
         cd $SROCK_THEROCK_DIR/$_subdir
         _subdirname=`echo $_subdir | tr "/" "_"`
         _patch_file=$thisdir/patches/$SROCK_COMPILER_BRANCH/$_subdirname.patch
         if [ -f "$_patch_file" ] ; then 
            test_apply_patch
         fi
      done < $_tmpfile
      rm $_tmpfile
   fi
echo "      --- end compiler submodule updates for $SROCK_COMPILER_BRANCH"
fi #  END compiler submodule updates

(
cd $SROCK_THEROCK_DIR
# reconstruct .amd-llvm.smrev using the current SHA
cd compiler/amd-llvm || exit
smrev="../.amd-llvm.smrev"
git config --get remote.origin.url > "$smrev"
smsha=$(git rev-parse HEAD)
echo "${smsha}${LLVM_SHA_EXTRA}" >> "$smrev"
)

cd $SROCK_THEROCK_DIR
echo 
echo "===== cmake CMD: $SROCK_CMAKE $_cmake_args"
# shellcheck disable=SC2090
$SROCK_CMAKE $_cmake_args
_rc=$? && [ "$_rc" != 0 ] && cd "$_curdir" && exit "$_rc"
# ------------  end of NOT restart mode ------------
fi

_prep_secs=$(date +%s)

# restart mode starts here
cd $SROCK_THEROCK_DIR
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
echo "rsync -a$_rsync_v --delete dist/rocm/ $SROCK_INSTALL_DIR/"
rsync -a$_rsync_v --delete dist/rocm/ "$SROCK_INSTALL_DIR"/
# FileCheck binary not found in dist/rocm, so get it from amd-llvm build
echo cp -p  ./compiler/amd-llvm/build/bin/FileCheck "$SROCK_INSTALL_DIR/lib/llvm/bin/FileCheck"
cp -p  ./compiler/amd-llvm/build/bin/FileCheck "$SROCK_INSTALL_DIR/lib/llvm/bin/FileCheck"

if [ ! -d ${SROCK_REPOS}/hipfort/build ] ; then 
   echo
   echo "===== Sourcing build_hipfort.sh to build and install hipfort"
   . $thisdir/build_hipfort.sh
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
_secs_to_prep=$(( _prep_secs - _start_secs ))
_secs_to_build=$(( _end_secs - _prep_secs ))
_filecount=$(find "$SROCK_INSTALL_DIR" -type f | wc -l)
_size=$(du -hs "$SROCK_INSTALL_DIR" | cut -f1)

echo
echo "===== Linking $SROCK_INSTALL_DIR to $SROCK_LINK" 
if [ -L "SROCK_LINK" ] ; then
   rm "$SROCK_LINK"
fi
echo ln -sf "$SROCK_INSTALL_DIR" "$SROCK_LINK"
ln -sf "$SROCK_INSTALL_DIR" "$SROCK_LINK"

echo
echo "===== DONE $0 on $_end_date"
echo "      THEROCK targets:   $_gfxsemicolons"
echo "      ROCm install dir:  $SROCK_INSTALL_DIR"
echo "      TheRock Dir:       $SROCK_THEROCK_DIR"
echo "      Compiler branch:   $SROCK_COMPILER_BRANCH"
echo "      Build Mode:        $_build_srock_mode"
echo "      SROCK config name: $SROCK_CONFIG"
echo "      cmake command:     $SROCK_CMAKE"
echo "      cmake args:        $_cmake_args"
echo "      Data prep time:    $_secs_to_prep (seconds)"
echo "      Build time:        $_secs_to_build (seconds)"
echo "      Files:             $_filecount"
echo "      Size:              $_size"
echo
echo "      For aomp testing, set AOMP=$SROCK_LINK"
echo "         or AOMP=$SROCK_INSTALL_DIR"
echo
