#! /usr/bin/env bash

# HPCG_SOURCE_DIR           where to clone sources to. Default: AOMP_REPOS_TEST
# HPCG_BUILD_NUM_THREADS    Number of parallel compile processes. Default: 32

realpath=`realpath $0`
thisdir=`dirname $realpath`
export AOMP_USE_CCACHE=0

. $thisdir/aomp_common_vars

: ${HPCG_SOURCE_DIR:=$AOMP_REPOS_TEST/hpcg}
: ${HPCG_BUILD_NUM_THREADS:=32}

rm -rf ${HPCG_SOURCE_DIR}

if [ ! -d ${HPCG_SOURCE_DIR} ]; then
  mkdir -p ${HPCG_SOURCE_DIR}
fi

cd ${HPCG_SOURCE_DIR} || exit 1

if [ ! -d ./rocHPCG ]; then
  # Get the sources
  git clone --depth 1 --single-branch -b omptarget https://github.com/ROCmSoftwarePlatform/rocHPCG || exit 1
  cd rocHPCG || exit 1
  git checkout omptarget || exit 1
fi

cd omptarget/hpcg || exit 1

mkdir build
cd build || exit 1

../configure LLVM_OMP_TARGET

make --output-sync -j${HPCG_BUILD_NUM_THREADS}

cd bin
./xhpcg 104 104 104 600 >/dev/null 2>&1

# If that is not found, the result is valid.
cat *.txt | grep -e "INVALID"
if [ $? -ne 1 ]; then
  echo "Invalid result produced, exiting."
  exit 1
fi

# The script does full rebuilds every time, so only one .txt file here.
cat HPCG-*.txt

