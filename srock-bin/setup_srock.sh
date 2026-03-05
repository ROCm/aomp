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

# TWO QUICK CHECKS, MUST HAVE SROCK_REPOS and SROCK_THEROCK_DIR MUST NOT EXIST
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
   echo " ERROR:  $0 requires that $SROCK_THEROCK_DIR NOT exist"
   echo "         Delete or move that directory to run $0"
   exit 1
fi

_curdir=$PWD
_start_date=$(date)
_start_secs=$(date +%s)

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
-DTHEROCK_ENABLE_LLVM_TESTS=1 \
$_cmake_enable \
$SROCK_THEROCK_DIR"

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
echo "      cmake args:        $_cmake_args"

# Run srock prebuild which includes finding suitable cmake
echo
echo "===== Sourcing prebuild_srock.sh"
. $thisdir/prebuild_srock.sh
echo "===== DONE Sourcing prebuild_srock.sh"

cd $SROCK_REPOS
echo
echo "===== git clone https://github.com/ROCm/TheRock.git -b $SROCK_THEROCK_BRANCH TheRock"
git clone https://github.com/ROCm/TheRock.git -b $SROCK_THEROCK_BRANCH TheRock

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
fi

_setup_secs=$(date +%s)
_secs_to_setup=$(( _setup_secs - _start_secs ))

echo
echo "===== DONE $0 on $_start_date"
echo "      THEROCK targets:   $_gfxsemicolons"
echo "      THEROCK families:  $_gfamsemicolons"
echo "      ROCm install dir:  $SROCK_INSTALL_DIR"
echo "      TheRock Dir:       $SROCK_THEROCK_DIR"
echo "      TheRock branch:    $SROCK_THEROCK_BRANCH"
echo "      Compiler branch:   $SROCK_COMPILER_BRANCH"
echo "      SROCK config name: $SROCK_CONFIG"
echo "      Setup time:        $_secs_to_setup (seconds)"
echo 
echo " The next step is to run: post_setup_build_srock.sh" 
echo 

